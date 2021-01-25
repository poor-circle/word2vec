#ifndef HUFFMANTREE_H
#define HUFFMANTREE_H

#include <vector>
#include <algorithm>
#include <map>
#include <stdlib.h>
#include <string>
#include <queue>
#include <stack>
#include <memory>


#include "base.h"
namespace Word2Vec::Util
{
	using namespace std;
	
	struct HuffmanNode
	{
	private:
		friend class HuffmanTree;
		friend class HuffmanPath;
		int frequency;       // 字符出现的频率
		unique_ptr<HuffmanNode> leftChild,rightChild;
		unique_ptr<VecCol> syn1;
		HuffmanNode *fa;
		string_view word;   
	public:
		HuffmanNode(unique_ptr<HuffmanNode>&& left,unique_ptr<HuffmanNode>&& right):
			frequency(left->frequency+right->frequency),
			leftChild(move(left)),
			rightChild(move(right)),
			syn1(make_unique<VecCol>()),
			fa(nullptr),
			word()
		{
			leftChild->fa=rightChild->fa=this;
			syn1->setZero();
		}
		HuffmanNode(string_view word, int frequency) :
			frequency(frequency),
			leftChild(),
			rightChild(),
			syn1(),
			fa(nullptr),
			word(word)
		{}
		friend bool operator <(const HuffmanNode& a, const HuffmanNode& b) noexcept
		{
			return a.frequency < b.frequency;
		}
	};
	class HuffmanPath
	{
	public:
		struct iterator : public std::iterator<std::forward_iterator_tag, HuffmanNode> //前向迭代器定义
		{
			friend class HuffmanTree;
			friend class HuffmanPath;
		private:
			HuffmanNode* address; //指向节点的地址,被封装在内部
			void next(void) {address = address->fa;}
			iterator(HuffmanNode *address):address(address){}
		public:
			iterator(const iterator &other):address(other.address){}
			iterator operator++(int) //重载迭代器后置++运算符
			{
				iterator temp = *this;
				next();
				return temp;
			}
			bool isMyLeftChild(const iterator& child)
			{
				return address->leftChild.get()==child.address;
			}
			bool isMyChild(const iterator& child)
			{
				return child.address->fa==address;
			}
			friend bool operator==(const iterator& add, const iterator& add2)
			{
				return add.address == add2.address;
			}
			friend bool operator!=(const iterator& add, const iterator& add2)
			{
				return add.address != add2.address;
			}
			decltype(auto) operator*(void) //重载迭代器*运算符
			{
				return *(address->syn1);
			}
		};
	private:
		iterator _begin;
		HuffmanPath(const iterator &_begin):_begin(_begin){}
	public:
		friend class HuffmanTree;
		iterator end() noexcept {return nullptr;} 
		iterator begin()noexcept {return _begin;} 
		HuffmanPath(const HuffmanPath& other):_begin(other._begin){}
	};
	class HuffmanTree
	{
	public:
		HuffmanTree(const unordered_map<string, int>& frequencyMap) :root() 
		{
			if (frequencyMap.empty()) //如果输入文件为空
				return;
			auto comp=[](const unique_ptr<HuffmanNode> &a,const unique_ptr<HuffmanNode> &b)
			{
				return a->frequency > b->frequency;
			};
			priority_queue<unique_ptr<HuffmanNode>,vector<unique_ptr<HuffmanNode>>,decltype(comp)> que(comp);//优先队列
			for (auto& e: frequencyMap)
			{
				auto leaf=make_unique<HuffmanNode>(e.first, e.second);
				tab.emplace(leaf->word,*leaf.get());
				que.push(move(leaf)); //构造优先队列
			}
			while (que.size() >= 2) //当队列中超过两个节点时
			{
				unique_ptr<HuffmanNode> a,b;
				a.swap(const_cast<unique_ptr<HuffmanNode>&>(que.top())); //危险操作
				que.pop();
				b.swap(const_cast<unique_ptr<HuffmanNode>&>(que.top())); //危险操作
				que.pop();
				que.push(make_unique<HuffmanNode>(move(a),move(b)));  //取出头部两个节点，构造一个新的节点塞入优先队列
			}
			this->root.swap(const_cast<unique_ptr<HuffmanNode>&>(que.top()));  //最后剩下的节点就是huffman树的根节点
			return;
		}
		auto get(const string_view word)
		{
			auto iter=tab.find(word);
			if (iter!=tab.end())
				return HuffmanPath(&(iter->second));
			else
			  	return HuffmanPath(nullptr);
		}
	private:
		unique_ptr<HuffmanNode> root;
		unordered_map<string_view,HuffmanNode&> tab;
	};
}


#endif
