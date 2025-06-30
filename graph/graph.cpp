#include <algorithm>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <tuple>
#include <vector>
#include "../utils/types.h"
#include "../utils/utils.h"
#include "../graph/graph.h"

Graph::Graph()
: edge_count_(0)
, vlabel_count_(0)
, elabel_count_(0)
, neighbors_{}
, elabels_{}
, updates_{}
, vlabels_{}
{}

void Graph::AddVertex(uint id, uint label)
{
    if (id >= vlabels_.size())
    {
        vlabels_.resize(id + 1, NOT_EXIST);
        vlabels_[id] = label;
        neighbors_.resize(id + 1);
        elabels_.resize(id + 1);
    }
    else if (vlabels_[id] == NOT_EXIST)
    {
        vlabels_[id] = label;
    }
    
    vlabel_count_ = std::max(vlabel_count_, label + 1);
    /*std::cout << "labels: ";
    for (uint i = 0; i < vlabels_.size(); i++)
    {
        std::cout << i << ":" << vlabels_[i] << " (";
        for (uint j = 0; j < neighbors_[i].size(); j++)
        {
            std::cout << neighbors_[i][j] << ":" << elabels_[i][j] << " ";
        }
        std::cout << ")" << std::endl;
    }*/
}

void Graph::RemoveVertex(uint id)
{
    vlabels_[id] = NOT_EXIST;
    neighbors_[id].clear();
    elabels_[id].clear();
}

void Graph::AddEdge(uint v1, uint v2)
{
    if (!BS) {
        auto lower = std::lower_bound(neighbors_[v1].begin(), neighbors_[v1].end(), v2);
        if (lower != neighbors_[v1].end() && *lower == v2) return;

        size_t dis = std::distance(neighbors_[v1].begin(), lower);
        neighbors_[v1].insert(lower, v2);
    }
    else {
        BSEdge.resize(BSEdge.size() + 1);
        if (vlabels_[v1] > vlabels_[v2])
            BSEdge.back() = {vlabels_[v2], vlabels_[v1]};
        else
            BSEdge.back() = {vlabels_[v1], vlabels_[v2]};
    }
    // elabels_[v1].insert(elabels_[v1].begin() + dis, label);
    
//    lower = std::lower_bound(neighbors_[v2].begin(), neighbors_[v2].end(), v1);
//    dis = std::distance(neighbors_[v2].begin(), lower);
//    neighbors_[v2].insert(lower, v1);
//    elabels_[v2].insert(elabels_[v2].begin() + dis, label);

    edge_count_++;
    // elabel_count_ = std::max(elabel_count_, label + 1);
    // print graph
    /*std::cout << "labels: ";
    for (uint i = 0; i < vlabels_.size(); i++)
    {
        std::cout << i << ":" << vlabels_[i] << " (";
        for (uint j = 0; j < neighbors_[i].size(); j++)
        {
            std::cout << neighbors_[i][j] << ":" << elabels_[i][j] << " ";
        }
        std::cout << ")" << std::endl;
    }*/
}

void Graph::RemoveEdge(uint v1, uint v2)
{
    auto lower = std::lower_bound(neighbors_[v1].begin(), neighbors_[v1].end(), v2);
    if (lower == neighbors_[v1].end() || *lower != v2)
    {
        std::cout << "deletion error" << std::endl;
        exit(-1);
    }
    neighbors_[v1].erase(lower);
    elabels_[v1].erase(elabels_[v1].begin() + std::distance(neighbors_[v1].begin(), lower));
    
    lower = std::lower_bound(neighbors_[v2].begin(), neighbors_[v2].end(), v1);
    if (lower == neighbors_[v2].end() || *lower != v1)
    {
        std::cout << "deletion error" << std::endl;
        exit(-1);
    }
    neighbors_[v2].erase(lower);
    elabels_[v2].erase(elabels_[v2].begin() + std::distance(neighbors_[v2].begin(), lower));

    edge_count_--;
}

uint Graph::GetVertexLabel(uint u) const
{
    return vlabels_[u];
}

const std::vector<uint>& Graph::GetNeighborLabels(uint v) const
{
    return elabels_[v];
}

void Graph::LoadFromFile(const std::string &path, bool bs)
{
    BS = bs;
    if (!io::file_exists(path.c_str()))
    {
        std::cout << "Failed to open: " << path << std::endl;
        exit(-1);
    }
    std::ifstream ifs(path);

    std::string type;
    while (ifs >> type)
    {
        if (type == "t")
        {
            uint temp1;
            uint temp2;
            ifs >> temp1 >> temp2;
        }
        else if (type == "v")
        {
            uint vertex_id, label;
            ifs >> vertex_id >> label;
            AddVertex(vertex_id, label);
        }
        else
        {
            uint from_id, to_id;
            ifs >> from_id >> to_id;
            AddEdge(from_id, to_id);
        }
    }
    ifs.close();
}

void Graph::LoadUpdateStream(const std::string &path)
{
    if (!io::file_exists(path.c_str()))
    {
        std::cout << "Failed to open: " << path << std::endl;
        exit(-1);
    }
    std::ifstream ifs(path);

    std::string type;
    while (ifs >> type)
    {
        if (type == "v" || type == "-v")
        {
            uint vertex_id, label;
            ifs >> vertex_id >> label;
            updates_.emplace('v', type == "v", vertex_id, 0u, label);
        }
        else
        {
            uint from_id, to_id, label;
            ifs >> from_id >> to_id >> label;
            updates_.emplace('e', type == "e", from_id, to_id, label);
        }
    }
    ifs.close();
}

void Graph::PrintMetaData() const
{
    std::cout << "# vertices = " << NumVertices() <<
        "\n# edges = " << NumEdges() << std::endl;
}

Graph::~Graph() {
    for (auto& i : neighbors_)
        i.clear();
    neighbors_.clear();
    neighbors_.shrink_to_fit();
    for (auto& i : elabels_)
        i.clear();
    elabels_.clear();
    elabels_.shrink_to_fit();
    vlabels_.clear();
    vlabels_.shrink_to_fit();
}
