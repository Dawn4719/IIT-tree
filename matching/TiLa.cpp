#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <iostream>

#include "TiLa.h"

void TiLa_Graph::build(int K, std::string query_path) {
    T = K;
    for (int snap_i = 0; snap_i < K; ++snap_i) {
        Graph snap{};
        snap.LoadFromFile(query_path + std::to_string(snap_i));
        // std::cout << snap.edge_count_ << std::endl;
        for (int i = 0; i < snap.neighbors_.size(); ++i) {
            auto v1 = snap.GetVertexLabel(i);
            for (auto j : snap.neighbors_[i]) {
                uint v2 = snap.GetVertexLabel(j);
                addEdge(v1, v2, snap_i);
            }
        }
    }
}


void TiLa_Graph::addEdge(int a, int b, int time) {
    if (a + 1 > TiLaEdge.size()) TiLaEdge.resize(a + 1);
    if (TiLaEdge[a].empty()) {
        TiLaEdge[a] = {b};
        TiLaEdge[a][0].lifespan.resize(T + 1);
        TiLaEdge[a][0].lifespan.set(time);
    }
    else {
        auto lower = std::lower_bound(TiLaEdge[a].begin(), TiLaEdge[a].end(), b);
        int idx = lower - TiLaEdge[a].begin();
        if (lower != TiLaEdge[a].end() && lower->nei_id == b) {
            TiLaEdge[a][idx].lifespan.set(time);
        }
        else {
            TiLaEdge[a].insert(lower, TiLa_Edge(b));
            TiLaEdge[a][idx].lifespan.resize(T + 1);
            TiLaEdge[a][idx].lifespan.set(time);
        }
    }
}

void TiLa_Graph::addEdge(int a, int b, int la, int lb, int time) {
    if (a + 1 > nodes.size()) nodes.resize(a + 1);
    if (nodes[a].id == -1) {
        node_nums++;
        nodes[a] = TiLa_Node(node_nums);
    }
    if (b + 1 > nodes.size()) nodes.resize(b + 1);
    if (nodes[b].id == -1) {
        node_nums++;
        nodes[b] = TiLa_Node(node_nums);
    }
    if (time >= nodes[a].labels[la].size()) {
        nodes[a].labels[la].resize(time + 1);
    }
    nodes[a].labels[la].set(time);
    if (time >= nodes[b].labels[lb].size()) {
        nodes[b].labels[lb].resize(time + 1);
    }
    nodes[b].labels[lb].set(time);
}
