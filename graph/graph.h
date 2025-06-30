#ifndef GRAPH_GRAPH
#define GRAPH_GRAPH

#include <queue>
#include <tuple>
#include <vector>
#include <unordered_map>
#include <map>
#include "../utils/types.h"
#include "../utils/utils.h"

struct Edge {
    uint val;
    std::vector<uint> csr;
    std::vector<std::string> bm;
    Edge()=default;
    Edge(uint a, std::vector<uint> b, std::vector<std::string> c={}): val(a), csr(b), bm(c) {};
    bool operator<(const int r) const {
        return val < r;
    }
};

struct Node {
    uint id;
    uint fa;
    int isMat=0;
    int ednums=0;
    std::map<int, std::vector<Edge>> Neighbors;
    std::map<int, int> Neighbor_delta;
    std::vector<uint> son;
};

class Graph
{
public:
    bool BS;
    uint vlabel_count_;
    uint elabel_count_;
    std::vector<std::vector<uint>> neighbors_;
    std::vector<std::vector<uint>> elabels_;
    std::vector<std::pair<int, int>> BSEdge;
    std::queue<InsertUnit> updates_;
    std::vector<uint> vlabels_;

    uint edge_count_;

    Graph();

    uint NumVertices() const { return vlabels_.size(); }
    int NumEdges() const { return edge_count_; }
    uint NumVLabels() const { return vlabel_count_; }
    uint NumELabels() const { return elabel_count_; }

    void AddVertex(uint id, uint label);
    void RemoveVertex(uint id);
    void AddEdge(uint v1, uint v2);
    void RemoveEdge(uint v1, uint v2);

    uint GetVertexLabel(uint u) const;
    const std::vector<uint>& GetNeighborLabels(uint v) const;

    void LoadFromFile(const std::string &path, bool bs = false);
    void LoadUpdateStream(const std::string &path);
    void PrintMetaData() const;
    ~Graph();
};

#endif //GRAPH_GRAPH
