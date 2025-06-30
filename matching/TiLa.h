#ifndef TREE_TILA_H
#define TREE_TILA_H

#endif //TREE_TILA_H

#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <iostream>
#include <boost/dynamic_bitset.hpp>
#include "../graph/graph.h"
using boost::dynamic_bitset;

//private List<Map<Integer, Set<Node>>> TiLa;

struct TiLa_Edge {
    int nei_id;
    dynamic_bitset<> lifespan;
    TiLa_Edge(){};
    TiLa_Edge(int nei_) {
        nei_id = nei_;
        lifespan = {};
    }
    bool operator<(const uint r) const {
        return nei_id < r;
    }
};

struct TiLa_Node {
    int id;
    std::map<int, dynamic_bitset<>> labels;
    TiLa_Node(){ id = -1; };
    TiLa_Node(int id_) {
        id = id_;
        labels = {};
    }
};

class TiLa_Graph {
public:
    int T;
    std::vector<TiLa_Node> nodes;
    uint node_nums;
    std::vector<std::vector<TiLa_Edge>> TiLaEdge;
    // std::vector<std::map<int, std::set<TiLa_Node>>> TiLa;

    void build(int K, std::string query_path);
    void addEdge(int a, int b, int time);
    void addEdge(int a, int b, int la, int lb, int time);

    TiLa_Graph() {
        node_nums = 0;
        nodes = {};
        TiLaEdge = {};
        // TiLa = {};
    }

};

