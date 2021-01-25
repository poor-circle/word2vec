# word2vec

一个简单的基于c++17和softmax模型实现的word2vec算法,支持多线程计算：

这个程序需要依赖fmtlib和eigen才能运行，如果没有安装这两个库，可以直接将lib.7z解压到目录中

训练数据位于text8.7z中，请解压获得测试数据。

编译参数：g++ Word2Vec.cpp -o Word2Vec -Wall -Ofast -static-libgcc -std=c++17 -march=native

需要较新的g++版本以支持c++17

训练词向量：./Word2Vec skip-gram text8 ans.out 或 ./Word2Vec cbow text8 ans.out 

测试词向量质量： ./Word2Vec ans.out 

可输入一个单词，也可输入多个单词组成的表达式，例如，输入female-male+men,最接近的单词很可能是women

参数设定：位于base.h

同样配置下，cbow速度快但质量低，skip-gram速度慢但是质量高
cbow多迭代几次（倍数约等于窗口的平均大小），效果和耗时就接近skip-gram了，但是skip-gram对低频词效果还是更好一点

理论知识可阅读这篇链接：![https://www.cnblogs.com/peghoty/p/3857839.html]
