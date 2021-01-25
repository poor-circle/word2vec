#include <bits/stdc++.h>
#include "base.h"
namespace Word2Vec::Util
{
    using namespace std;
    constexpr auto sigmondRealCal(float_t x) noexcept
    {
        return 1.0/(1.0+exp(-x));
    }
    constexpr decltype(auto) sigmondPrecal() noexcept
    {
        array<float_t,precalAccuracy> ar{};
        for (size_t i=0;i<ar.size();++i)
            ar[i]=sigmondRealCal(i*precalRange/precalAccuracy);
        return ar;
    }
    constexpr auto sigmondPreCalResult=sigmondPrecal();
    constexpr auto sigmond(float_t x) noexcept//预处理sigmond
    {
        bool bit=(x>=0);
        x=bit?x:-x;
        float_t ans=(x>=precalRange)?(1.0):(sigmondPreCalResult[x*precalAccuracy/precalRange]);
        return bit?ans:1.0-ans;
    }
    decltype(auto) parseRawData(const char* rawDatainputPath)
    {   
        vector<string> ret;
        ifstream file;
        file.open(rawDatainputPath);
        ios::sync_with_stdio(false);
        while (true)
        {
            string temp;
            file>>temp;
            if (temp!="")
                ret.push_back(move(temp));
            else break;
        }
        ios::sync_with_stdio(true);
        file.close();
        return ret;
    }
    void filterWordsTable(unordered_map<string, int>& wordsList,size_t totalCount)
    {   
        for (auto iter=wordsList.begin();iter!=wordsList.end();)
        {
            if (iter->second<minWordCount)
                iter=wordsList.erase(iter);
            else 
            {
                float_t temp=maxWordFrequency*totalCount/iter->second;
                float_t lim=sqrt(temp)+temp,rnd=(float)rand()/RAND_MAX;
                if (lim<rnd)
                {
                    iter=wordsList.erase(iter);
                }
                else ++iter;
            }
        }
        return;
    }
    decltype(auto) bulidWordsTable(const vector<string>& wordsList)
    {   
        unordered_map<string, int> ret;
        for (auto &&e:wordsList)
            ++ret[e];
        filterWordsTable(ret,wordsList.size());
        return ret;
    }
    int randWindowSize() noexcept
    {
        return rand()%(WindowMax-WindowMin)+WindowMin+1;
    }
    auto setWindowRange(vector<string>::const_iterator iter,vector<string>::const_iterator beg,vector<string>::const_iterator end,int width=randWindowSize()) noexcept
    {
        auto ret = make_tuple(iter - width, iter + width + 1);
        get<0>(ret) = get<0>(ret) < beg ? beg : get<0>(ret);
        get<1>(ret) = get<1>(ret) > end ? end : get<1>(ret);
        return ret;
    }
    int saveWordVec(unordered_map<string, VecRow> &wordVecs,const char *trainResultoutputPath)
    {
        auto sz=wordVecs.size();
        FILE *fp=fopen(trainResultoutputPath,"wb");
        fwrite(&sz,sizeof(sz),1,fp);
        for (auto &e:wordVecs)
        {
            fwrite(e.second.data(),sizeof(float_t)*VecDim,1,fp);
        }
        for (auto &e:wordVecs)
            fmt::print(fp,"{} ",e.first);
        fclose(fp);
        return 0;
    }
    decltype(auto) readWordVec(const char * filePath)
    {
        unordered_map<string,VecRow> ret;
        ifstream file;
        file.open(filePath,ios::binary|ios::in);
        ios::sync_with_stdio(false);
        size_t total;
        file.read((char *)&total,sizeof(total));
        ret.reserve(total);
        file.seekg(sizeof(float_t)*VecDim*total,ios::cur);
        vector<VecRow*> order_temp;
        order_temp.reserve(total);
        while (true)
        {
            string temp;
            file>>temp;
            if (temp!="")
                order_temp.push_back(&(ret.emplace(move(temp),VecRow()).first->second));
            else break;
        }
        file.clear();
        file.seekg(sizeof(total),ios::beg);
        for (auto &&e:order_temp)
            file.read((char *)e->data(),sizeof(float_t)*VecDim);
        ios::sync_with_stdio(true);
        file.close();
        return ret;
    }
    constexpr float_t calRealLearnRate(int64_t freq,int64_t avgWordFreq) noexcept
    {
        float_t ret=rawLearnRate*(1-(float_t)freq/(avgWordFreq+1));
        return ret<minLearnRate?minLearnRate:ret;
    }
}
