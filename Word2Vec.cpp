//编译参数：g++ Word2Vec.cpp -o Word2Vec -Wall -Ofast -fopenmp -static-libgcc  -std=c++17 -march=native
//训练词向量：./Word2Vec skip-gram text8 ans.out ./Word2Vec cbow text8 ans.out
//同样配置下，cbow快但质量低，skip-gram慢但是质量高
//cbow多迭代几次（大约是窗口的平均大小），效果和耗时就接近skip-gram了，但是skip-gram对低频词效果还是更好一点
//测试词向量质量： ./Word2Vec ans.out
//输入单词来测试训练结果
//也可输入含+/-号的表达式：例如，输入female-male+men,理论上最接近的应该是women
//调整参数：请见base.h

#include "HuffmanTree.hpp"
#include "Word2VecDetail.hpp"
#include <bits/stdc++.h>
namespace Word2Vec
{
    using namespace std;
    using namespace Util;
    enum class errnum
    {
        succeed,
        trainWayNotFound,
        argFormatError
    };
    void cbow(unordered_map<string, VecRow> &wordVecs, HuffmanTree &tree, vector<string>::const_iterator begin,vector<string>::const_iterator end,const unordered_map<string, int> & stringTable,const int64_t totalTrainWords)
    {
        VecRow vecSum;
        VecCol e;
        int64_t avgWordFreq=totalTrainWords/wordVecs.size();//每个单词平均出现的次数
        for (auto iter = begin; iter != end; ++iter)
        {
            auto path = tree.get(*iter);    //获取树上路径
            if (path.begin() != path.end()) //树上有这个词（没有被过滤）
            { 
                int64_t freq=0;
                vecSum.setZero();
                e.setZero();
                int c=randWindowSize();
                auto [low,up] = setWindowRange(iter,begin,end,c);//设置窗口范围
                //计算累加和
                for (auto iter2=low;iter2!=up;++iter2)
                {
                    if (*iter2==""||iter2==iter) continue;//累加和不包括自己，累加和不包括那些没有的
                    auto &vec=wordVecs.find(*iter2)->second;
                    vecSum += vec;
                    freq+=stringTable.find(*iter2)->second;
                }
                freq/=2*c;//计算窗口内单词的平均词频
                float_t learnRate=calRealLearnRate(freq,avgWordFreq);//计算学习率
                //按叶子到根节点的顺序，遍历树上路径
                for (auto next=path.begin(),now=next++;next!=path.end();now=next++)
                {
                    float_t x=(vecSum * *next),q=sigmond(x);
                    auto g=learnRate*(1-next.isMyLeftChild(now)-q);
                    e+=*next * g;
                    *next+=vecSum* g;
                }
                //将学到的知识迭代给上下文的词向量
                for (auto iter2=low;iter2!=up;++iter2)
                {
                    if (*iter2==""||iter2==iter) continue;//累加和不包括自己，累加和不包括那些没有的
                    wordVecs.find(*iter2)->second += e;
                }
            }
        }
    }
    void skipGram(unordered_map<string, VecRow> &wordVecs, HuffmanTree &tree, vector<string>::const_iterator begin,vector<string>::const_iterator end,const unordered_map<string, int> &stringTable,int64_t totalTrainWords)
    {
        VecRow vecSum;
        VecCol e;
        int64_t avgWordFreq=totalTrainWords/wordVecs.size();//每个单词平均出现的次数
        for (auto iter = begin; iter != end; ++iter)        //遍历区间内所有的单词
        {
            auto pos=wordVecs.find(*iter);
            if (pos != wordVecs.end()) //树上有这个词（没有被过滤）
            { 
                auto &vecSum=pos->second;
                auto [low,up] = setWindowRange(iter,begin,end);//设置窗口范围
                int64_t freq=stringTable.find(*iter)->second;
                float_t learnRate=calRealLearnRate(freq,avgWordFreq);//计算学习率
                //按叶子到根节点的顺序，遍历huffman树上路径
                for (auto iter2=low;iter2!=up;++iter2)
                {
                    if (*iter2==""||iter2==iter) continue;
                    e.setZero();
                    auto path = tree.get(*iter2);
                    for (auto next=path.begin(),now=next++;next!=path.end();now=next++)
                    {
                        float_t x=(vecSum * *next),q=sigmond(x);
                        auto g=learnRate*(1-next.isMyLeftChild(now)-q);
                        e+=*next * g;
                        *next+=vecSum* g;
                    }
                    vecSum += e;
                }
            }
        }
    }

    const static unordered_map<string, function<decltype(cbow)>> algoTable = {{"cbow", cbow}, {"skip-gram", skipGram}};

    errnum parrelTrain(function<decltype(cbow)> algo,unordered_map<string, VecRow> &wordVecs, HuffmanTree &tree, const vector<string> &wordsList,const unordered_map<string, int> &stringTable,int64_t totalTrainWords)
    {
        for (int k=0;k<trainCount;++k)
        {
            //不加锁的并行（非常疯狂），尽可能提高性能
            auto threadCount=std::thread::hardware_concurrency()/(DisableHT?2:1);//超线程下反而跑的更慢了
            threadCount=max(threadCount,1u);//至少保证1个线程
            vector<thread> threads;
            threads.reserve(threadCount);
            auto dis=(wordsList.end()-wordsList.begin())/threadCount;
            for (size_t i=0;i<threadCount;)
            {
                auto iterBegin=wordsList.begin()+dis*i;
                auto iterEnd=(++i==threadCount)?wordsList.end():iterBegin+dis;
                threads.emplace_back(algo,ref(wordVecs),ref(tree),iterBegin,iterEnd,ref(stringTable),totalTrainWords);
            }
            for (auto &&e:threads)
                e.join();
        }
        return errnum::succeed;
    }
    
    errnum showVec(const unordered_map<string,VecRow> & wordVecs);
    
    errnum train(const char *trainWay, const char *rawDatainputPath, const char *trainResultoutputPath)
    {
        auto algoiter=algoTable.find(trainWay);
        if (algoiter == algoTable.end())
            return errnum::trainWayNotFound;             //不存在对应的训练算法
        srand(time(nullptr));                            //随机初始化
        auto wordsList = parseRawData(rawDatainputPath); //解析原文
        auto stringTable = bulidWordsTable(wordsList);   //计算词频，略去低频和高频词
        replace_if(wordsList.begin(),wordsList.end(),[&](auto & e){
            return stringTable.find(e)==stringTable.end();
        },string(""));
        HuffmanTree tree(stringTable);                   //建立霍夫曼树
        unordered_map<string, VecRow> wordsVecs;
        wordsVecs.reserve(stringTable.size());
        int64_t totalTrainWords=0;
        for (auto &&e : stringTable)
        {
            totalTrainWords+=e.second;
            wordsVecs.emplace(e.first, VecRow::Random() / (2 * VecDim)); //生成填充了[-0.5/维度,0.5/维度]的随机元素的向量
        }
        parrelTrain(algoiter->second,wordsVecs, tree, wordsList,stringTable,totalTrainWords);//训练模型
        for(auto &&e:wordsVecs)
            e.second.normalize();                                       //词向量归一化
        saveWordVec(wordsVecs,trainResultoutputPath);                   //保存词向量                              
        return errnum::succeed;
    }
    errnum show(const char *trainResultInputPath)
    {
        auto temp=readWordVec(trainResultInputPath);
        showVec(temp);
        return errnum::succeed;
    }
    errnum showVec(const unordered_map<string,VecRow> & wordVecs)
    {
        string queryWord;
        fmt::print("Please input the query word.(input :q to quit)\n");
        vector<pair<float_t,string>> dis;
        dis.reserve(queryWord.size());
        std::regex reg ("[^+-]+");
        while (getline(cin,queryWord))
        {
            if (queryWord==":q"&&queryWord!="") break;
            smatch result;  
            bool flag=0;
            VecRow vec;
            vec.setZero();  
            auto start=queryWord.cbegin();
            while (regex_search(start, queryWord.cend(), result, reg ))
            {
                auto iter=wordVecs.find(result[0]);
                if (iter==wordVecs.end()) 
                {
                    fmt::print("The query word is not exist because the frequency is too high or too low in data.\n");
                    flag=1;
                    break;
                }
                else
                {
                    if (*result[0].second=='-')
                        vec-=iter->second;
                    else
                        vec+=iter->second;
                }
                start = result[0].second;	//更新搜索起始位置,搜索剩下的字符串
            }
            if (flag) continue;
            transform(wordVecs.begin(),wordVecs.end(),back_inserter(dis),[&](const auto&e)
            {
                return make_pair((vec-e.second).norm(),e.first);
            });
            auto iter=dis.size()>=21?dis.begin()+21:dis.end();
            nth_element(dis.begin(),iter,dis.end());
            dis.erase(iter,dis.end());
            sort(dis.begin(),dis.end());
            dis.erase(dis.begin(),dis.begin()+1);
            fmt::print("{}\n",fmt::join(dis,"\n"));
            dis.clear();
        }
        return errnum::succeed;
    }
    unordered_map<errnum, string> errmsg =
    {
        {errnum::succeed, "Run succeed"},
        {errnum::trainWayNotFound, "Error: trainWayNotFound"},
        {errnum::argFormatError, "Error: argFormatError"}
    };
} // namespace Word2Vec
int main(int argc, char **argv)
{
    std::chrono::high_resolution_clock clk;
    auto beg = clk.now();
    Word2Vec::errnum err = Word2Vec::errnum::argFormatError;
    switch (argc)
    {
    case 2: //测试词向量
        err = Word2Vec::show(argv[1]);
        break;
    case 4: //训练词向量
        err = Word2Vec::train(argv[1], argv[2], argv[3]);
        break;
    default:
        fmt::print
        (
            "┌{0:─^{3}}┐\n"
            "│{1: ^{3}}│\n"
            "│{2: ^{3}}│\n"
            "└{0:─^{3}}┘\n",
            "",
            "train data: {{cbow | skip-gram},inputPath,resultOutputpath}",
            "test train result: {resultOutputpath}", 70
        );
    }
    fmt::print("{}\n", Word2Vec::errmsg[err]);
    fmt::print("used time: {}\n", std::chrono::duration_cast<std::chrono::milliseconds>(clk.now() - beg));
    return 0;
}