#ifndef MATCHING_DELTAGRAPH
#define MATCHING_DELTAGRAPH

#include <queue>
#include <map>
#include <vector>

#include "../graph/graph.h"
#include "../matching/matching.h"

class DeltaGraph {
public:
    Graph cur_graph {};
    std::vector<Node> skeleton;
    int k;
    int leafs;
    int depth;
    int node_nums;
    int pool_bias;
    int pooll, poolr;

    std::string fdif;
    std::vector<std::vector<unsigned long long>> dist;
    std::vector<std::vector<uint>> path;
    std::vector<Edge> pool_vertex;
    std::vector<std::vector<uint>> gpvertex;
    std::vector<uint> p;

    DeltaGraph(int k_, int n);
//    ~DeltaGraph(){};
    void build(std::string path, int n);
    void dif(int l, int r);
    void cha(int big, int small, int store, int big_idx=-1, int store_idx=-1);
    void pool_cha(int big, int small, int store, int big_idx=-1, int store_idx=-1);
    void pool_cup(int big, int small, int store, int big_idx=-1, int store_idx=-1);
    void cha(Node& skeleton, Node& store, int idx);
    void cap(Node&, Node&);
    void graphpool(int level);
    void cup(Node&, Node&, int idx=-1);
    void GetDist();
    std::vector<std::vector<uint>> Prim(int l, int r);
    void skewed();
    void balence();
    void Rskewed();
    void Lskewed();
    void mixed();
    void empty();
    int find(int u);
    void get_snap();
};

#endif //MATCHING_DELTAGRAPH
