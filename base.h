#ifndef BASE_H
#define BASE_H
#define FMT_HEADER_ONLY
#include <fmt/printf.h>
#include <fmt/ranges.h>
#include <fmt/chrono.h>
#include <fmt/compile.h>
#include "Eigen/Dense"
namespace Word2Vec
{
    constexpr bool DisableHT=true;//禁用超线程（只使用一半的CPU核心）以提高性能
                                  //没有开启超线程的机器可以设为false
    constexpr int VecDim=300;//向量维度
    constexpr int minWordCount=5;//罕见词过滤
    constexpr float_t maxWordFrequency=0.001;//词频过滤
    constexpr int precalAccuracy=10000;//预处理sigmond函数
    constexpr float_t precalRange=20.0;//预处理的范围[-20,20]
    constexpr int WindowMin=1;//窗口大小
    constexpr int WindowMax=5;
    constexpr int trainCount=5;//迭代次数
    constexpr float_t rawLearnRate=0.010;//原始学习率
    constexpr float_t minLearnRate=0.0005;//最小学习率
    using VecCol=Eigen::Matrix<float_t,VecDim,1>;
    using VecRow=Eigen::Matrix<float_t,1,VecDim>;
}
#endif