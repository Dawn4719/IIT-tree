#include <algorithm>
#include <iostream>
#include <queue>
#include <map>
#include <unordered_set>
#include <vector>
#include <assert.h>
#include "../graph/graph.h"
#include "deltagraph.h"

#include <vector>

DeltaGraph::DeltaGraph(int k_, int n){
    k = k_;
    leafs = n;
    int nums = n;
    depth = 1;
    pool_bias = 0;
    while (n > 1) {
        nums += n / k;
        bool ok = false;
        if (n % k && n / k) {
            nums += 1;
            ok = true;
        }
        if (n / k == 0 && n % k != 0 && n != 1)
            nums++;
        n /= k;
        if (ok) n++;
        depth++;
    }
    node_nums = nums;

    skeleton.resize(nums + 1);
    for (int i = 0; i <= nums; ++i) {
        skeleton[i].id = i;
    }
    skeleton[nums].son.emplace_back(nums - 1);
    skeleton[nums].fa = 0;
    skeleton[nums - 1].fa = nums;
    std::cout << "nums: " << nums << std::endl;
    std::cout << "depth: " << depth << std::endl;
}

void DeltaGraph::build(std::string path, int n){
    int cnt = 0;
    int i;
    for (i = k; i <= n; i += k) {
        for (int j = i - k; j < i; j++) {
            Graph snap{};
            snap.LoadFromFile(path + std::to_string(j));
            std::cout << path + std::to_string(j) << std::endl;
            skeleton[n + cnt].son.emplace_back(j);
            skeleton[j].fa = n + cnt;
            int ednums = 0;
            for (int kk = 0; kk < snap.neighbors_.size(); ++kk) {
                auto v1 = snap.GetVertexLabel(kk);
                auto lower = std::lower_bound(skeleton[j].Neighbors[-1].begin(), skeleton[j].Neighbors[-1].end(), v1);
                uint idx = lower - skeleton[j].Neighbors[-1].begin();
                if (lower == skeleton[j].Neighbors[-1].end() || lower->val != v1) {
                    skeleton[j].Neighbors[-1].insert(lower, Edge (v1, std::vector<uint>{}));
                }
                skeleton[j].Neighbors[-1][idx].csr = snap.neighbors_[kk];
                for (auto& l : skeleton[j].Neighbors[-1][idx].csr) {
                    l = snap.GetVertexLabel(l);
                }
                skeleton[j].ednums += skeleton[j].Neighbors[-1][idx].csr.size();
                ednums += skeleton[j].Neighbors[-1][idx].csr.size();
                std::sort(skeleton[j].Neighbors[-1][idx].csr.begin(), skeleton[j].Neighbors[-1][idx].csr.end());
            }
            std::cout << j << " edge nums: " << ednums << std::endl;
        }

        dif( std::max(0, i - k), i);
        std::cout << "dif: " << mem::getValue() / 1024 << " mb" << std::endl;

        int sum = 0;

        for (int j = std::max(0, i - k); j < i; j++) {
            if (fdif == "intersection")
                cha(j, skeleton[j].fa, skeleton[j].fa, -1, j);
            else
                cha(skeleton[j].fa, j, skeleton[j].fa, -1, j);
        }
        for (auto j : skeleton[skeleton[std::max(0, i - k)].fa].Neighbors) {
            for (auto ll : j.second) {
                skeleton[skeleton[std::max(0, i - k)].fa].ednums += ll.csr.size();
            }
        }

        for (int j = std::max(0, i - k - 1); j < i - 1; j++) {
            if (fdif == "intersection") {
                cha(j + 1, j, j + 1, -1, j);
                cha(j, j + 1, j, -1, j + 1);
            }
            else {
                cha(j + 1, j, j, -1, j + 1);
                cha(j, j + 1, j + 1, -1, j);
            }
            skeleton[j].Neighbors.erase(-1);
//            std::cout << "cha: " << mem::getValue() / 1024 << " mb" << std::endl;
        }
//        std::cout << "build one: " << mem::getValue() / 1024 << " mb" << std::endl;
        cnt ++;
    }

    if (i - k != n) {
        for (int j = i - k; j < n; j++) {
            Graph snap{};
            snap.LoadFromFile(path + std::to_string(j));
            std::cout << path + std::to_string(j) << std::endl;
            skeleton[n + cnt].son.emplace_back(j);
            skeleton[j].fa = n + cnt;
            skeleton[j].id = j;
            for (int kk = 0; kk < snap.neighbors_.size(); ++kk) {
                auto v1 = snap.GetVertexLabel(kk);
                auto lower = std::lower_bound(skeleton[j].Neighbors[-1].begin(), skeleton[j].Neighbors[-1].end(), v1);
                uint idx = lower - skeleton[j].Neighbors[-1].begin();
                if (lower == skeleton[j].Neighbors[-1].end() || lower->val != v1) {
                    skeleton[j].Neighbors[-1].insert(lower, Edge (v1, std::vector<uint>{}));
                }
                skeleton[j].Neighbors[-1][idx].val = v1;
                skeleton[j].Neighbors[-1][idx].csr = snap.neighbors_[kk];
                for (auto& l : skeleton[j].Neighbors[-1][idx].csr) {
                    l = snap.GetVertexLabel(l);
                }
                std::sort(skeleton[j].Neighbors[-1][idx].csr.begin(), skeleton[j].Neighbors[-1][idx].csr.end());
            }
        }

        dif( i - k, n);

        for (int j = i - k; j < n; j++) {
            if (fdif == "intersection")
                cha(j, skeleton[j].fa, skeleton[j].fa, -1, j);
            else
                cha(skeleton[j].fa, j, skeleton[j].fa, -1, j);
        }
        for (int j = i - k - 1; j < n - 1; j++) {
            if (fdif == "intersection") {
                cha(j + 1, j, j + 1, -1, j + 1);
                cha(j, j + 1, j, -1, j);
            }
            else {
                cha(j + 1, j, j, -1, j + 1);
                cha(j, j + 1, j + 1, -1, j);
            }
            skeleton[j].Neighbors.erase(-1);
        }
//        std::cout << "build one: " << mem::getValue() / 1024 << " mb" << std::endl;
        cnt++;
    }
    skeleton[n - 1].Neighbors.erase(-1);
    for (int i = 0; i < n - 1; ++i) {
        skeleton[i].son.emplace_back(i + 1);
    }
    for (int i = 1; i < n; ++i) {
        skeleton[i].son.emplace_back(i - 1);
    }
//    for (int i = 0; i < n; ++i)
//        skeleton[i].Neighbors.erase(-1);
    n += cnt;
//    std::cout << mem::getValue() / 1024 << " mb" << std::endl;
    int cur_depth = 2;

    while (cur_depth < depth) {
        std::cout << "cur_depth_node_num:" << cnt << std::endl;
        i = n - cnt + k;
        cnt = 0;
        for (; i <= n; i += k) {
            for (int j = i - k; j < i; j++) {
                skeleton[n + cnt].son.emplace_back(j);
//                std::cout << n + cnt << ' ' << j << std::endl;
                skeleton[j].fa = n + cnt;
                skeleton[j].id = j;
            }
//            std::cout << i << "---" << std::endl;
            dif( i - k, i);

            for (int j = i - k; j < i; j++) {
                if (fdif == "intersection")
                    cha(j, skeleton[j].fa, skeleton[j].fa,-1,  j);
                else
                    cha(skeleton[j].fa, j, skeleton[j].fa, -1, j);
                skeleton[j].Neighbors.erase(-1);
            }

            cnt ++;
        }
        if (i - k != n) {
            for (int j = i - k; j < n; j++) {            
                skeleton[n + cnt].son.emplace_back(j);
                skeleton[j].fa = n + cnt;
                skeleton[j].id = j;
            }
            dif( i - k, n);

            for (int j = i - k; j < n; j++) {
                if (fdif == "intersection")
                    cha(j, skeleton[j].fa, skeleton[j].fa,-1,  j);
                else
                    cha(skeleton[j].fa, j, skeleton[j].fa, -1, j);
                skeleton[j].Neighbors.erase(-1);
            }
            cnt ++;
        }
        n += cnt;
        cur_depth++;
//        std::cout << mem::getValue() / 1024 << " mb" << std::endl;
    }

    std::vector<uint> sons;
    sons.emplace_back(skeleton.back().son[0]);
    for (int dep = 0; dep < depth; ++dep) {
        std::vector<uint> tmp_son;
        for (auto idx : sons) {
            for (auto each_son : skeleton[idx].son) {
                size_t delta = 0;
                for (const auto& nei : skeleton[idx].Neighbors[each_son])
                    delta += nei.csr.size();
                skeleton[idx].Neighbor_delta[each_son] = delta;
                tmp_son.emplace_back(each_son);
            }
        }
        sons = std::move(tmp_son);
    }

    std::cout << "Build over: " << mem::getValue() / 1024 << " mb" << std::endl;
}

void DeltaGraph::cha(int big, int small, int store, int big_idx, int store_idx) {
    for (int i = 0; i < skeleton[big].Neighbors[big_idx].size(); ++i) {
        auto v = skeleton[big].Neighbors[big_idx][i].val;
        auto lower2 = std::lower_bound(skeleton[small].Neighbors[big_idx].begin(), skeleton[small].Neighbors[big_idx].end(), v);
        int j = lower2 - skeleton[small].Neighbors[big_idx].begin();
        if (lower2 != skeleton[small].Neighbors[big_idx].end() && lower2->val == v) {
            auto lower = std::lower_bound(skeleton[store].Neighbors[store_idx].begin(), skeleton[store].Neighbors[store_idx].end(), v);
            int idx2 = lower - skeleton[store].Neighbors[store_idx].begin();
            if (lower == skeleton[store].Neighbors[store_idx].end() || lower->val != v) {
                skeleton[store].Neighbors[store_idx].insert(lower, Edge (v, std::vector<uint>{}));
                std::set_difference(skeleton[big].Neighbors[big_idx][i].csr.begin(), skeleton[big].Neighbors[big_idx][i].csr.end(),
                                    skeleton[small].Neighbors[big_idx][j].csr.begin(), skeleton[small].Neighbors[big_idx][j].csr.end(),
                                    std::back_inserter(skeleton[store].Neighbors[store_idx][idx2].csr));
            } else {
                std::vector<uint> tmp;
                std::set_difference(skeleton[big].Neighbors[big_idx][i].csr.begin(), skeleton[big].Neighbors[big_idx][i].csr.end(),
                                    skeleton[small].Neighbors[big_idx][j].csr.begin(), skeleton[small].Neighbors[big_idx][j].csr.end(),
                                    std::back_inserter(tmp));
                skeleton[store].Neighbors[store_idx][idx2].csr = tmp;
            }

        } else {
            auto lower = std::lower_bound(skeleton[store].Neighbors[store_idx].begin(), skeleton[store].Neighbors[store_idx].end(), v);
            skeleton[store].Neighbors[store_idx].insert(lower, Edge(v, skeleton[big].Neighbors[big_idx][i].csr));
        }
    }
    for (auto i = skeleton[store].Neighbors[store_idx].begin(); i != skeleton[store].Neighbors[store_idx].end(); )
        if (i->csr.empty()) {
            i = skeleton[store].Neighbors[store_idx].erase(i);
        } else {
            i ++;
        }
}

void DeltaGraph::pool_cha(int big, int small, int store, int big_idx, int store_idx) {
    for (int i = 0; i < skeleton[big].Neighbors[big_idx].size(); ++i) {
        auto v = skeleton[big].Neighbors[big_idx][i].val;
        auto lower = std::lower_bound(skeleton[big].Neighbors[store_idx].begin(), skeleton[big].Neighbors[store_idx].end(), v);
        int idx2 = lower - skeleton[big].Neighbors[store_idx].begin();
        if (lower == skeleton[big].Neighbors[store_idx].end() || lower->val != v) {
            skeleton[store].Neighbors[-1].resize(skeleton[store].Neighbors[-1].size() + 1);
            skeleton[store].Neighbors[-1].back().val = v;
            skeleton[store].Neighbors[-1].back().csr = skeleton[big].Neighbors[big_idx][i].csr;
        } else {
            skeleton[store].Neighbors[-1].resize(skeleton[store].Neighbors[-1].size() + 1);
            skeleton[store].Neighbors[-1].back().val = v;
            std::set_difference(skeleton[big].Neighbors[big_idx][i].csr.begin(), skeleton[big].Neighbors[big_idx][i].csr.end(),
                                skeleton[big].Neighbors[store_idx][idx2].csr.begin(), skeleton[big].Neighbors[store_idx][idx2].csr.end(),
                                std::back_inserter(skeleton[store].Neighbors[-1].back().csr));
        }
    }
    for (auto i = skeleton[store].Neighbors[-1].begin(); i != skeleton[store].Neighbors[-1].end(); )
        if (i->csr.empty()) {
            i = skeleton[store].Neighbors[-1].erase(i);
        } else {
            i ++;
        }
}

void DeltaGraph::pool_cup(int big, int small, int store, int big_idx, int store_idx) {
    if (skeleton[big].Neighbors[store_idx].size() == 0)
        std::swap(big_idx, store_idx);
    for (int i = 0; i < skeleton[big].Neighbors[store_idx].size(); ++i) {
        auto v = skeleton[big].Neighbors[store_idx][i].val;
        auto lower = std::lower_bound(skeleton[big].Neighbors[big_idx].begin(), skeleton[big].Neighbors[big_idx].end(), v);
        int idx2 = lower - skeleton[big].Neighbors[big_idx].begin();
        if (lower == skeleton[big].Neighbors[big_idx].end() || lower->val != v) {
            skeleton[store].Neighbors[-1].resize(skeleton[store].Neighbors[-1].size() + 1);
            skeleton[store].Neighbors[-1].back().val = v;
            skeleton[store].Neighbors[-1].back().csr = skeleton[big].Neighbors[store_idx][i].csr;
        } else {
            skeleton[store].Neighbors[-1].resize(skeleton[store].Neighbors[-1].size() + 1);
            skeleton[store].Neighbors[-1].back().val = v;
            std::set_union(skeleton[big].Neighbors[store_idx][i].csr.begin(), skeleton[big].Neighbors[store_idx][i].csr.end(),
                                skeleton[big].Neighbors[big_idx][idx2].csr.begin(), skeleton[big].Neighbors[big_idx][idx2].csr.end(),
                                std::back_inserter(skeleton[store].Neighbors[-1].back().csr));
        }
    }
    // for (auto i = skeleton[store].Neighbors[-1].begin(); i != skeleton[store].Neighbors[-1].end(); )
    //     if (i->csr.empty()) {
    //         i = skeleton[store].Neighbors[-1].erase(i);
    //     } else {
    //         i ++;
    //     }
}

void DeltaGraph::cha(Node& skeleton, Node& store, int idx) {
    for (int i = 0; i < store.Neighbors[-1].size(); ++i) {
        auto v = store.Neighbors[-1][i].val;
        auto lower = std::lower_bound(skeleton.Neighbors[idx].begin(), skeleton.Neighbors[idx].end(), v);
        int j = lower - skeleton.Neighbors[idx].begin();
        if (lower != skeleton.Neighbors[idx].end() && lower->val == v) {
            std::vector<uint> tmp;
            std::set_difference(store.Neighbors[-1][i].csr.begin(), store.Neighbors[-1][i].csr.end(),
                                          skeleton.Neighbors[idx][j].csr.begin(), skeleton.Neighbors[idx][j].csr.end(),
                                          std::back_inserter(tmp));
            store.Neighbors[-1][i].csr = tmp;
        }
    }
    for (auto i = store.Neighbors[-1].begin(); i != store.Neighbors[-1].end(); )
        if (i->csr.empty()) {
            i = store.Neighbors[-1].erase(i);
        } else {
            i ++;
        }
}

void DeltaGraph::dif(int l, int r) {
    Node res;
    for (int i = l; i < r; ++i) {
        if (fdif == "intersection") {
            if (i == l) {
                res = skeleton[i];
            } else
            cap(res, skeleton[i]);
        }
        else if (fdif == "union") {
            cup(res, skeleton[i]);
        }
        else if (fdif == "skewed") {
            
        }
        else if (fdif == "balence") {
            
        }
        else if (fdif == "rskewed") {
            
        }
        else if (fdif == "lskewed") {
            
        }
        else if (fdif == "mixed") {
            
        }
        else if (fdif == "empty") {

        }
    }
    skeleton[skeleton[l].fa].Neighbors = res.Neighbors;
    skeleton[skeleton[l].fa].ednums = 0;
    for (auto i : skeleton[skeleton[l].fa].Neighbors[-1]) {
        skeleton[skeleton[l].fa].ednums += i.csr.size();
    }
}

void DeltaGraph::cap(Node& res, Node& snap){
    std::map<int, bool> st;
    for (const auto& i : snap.Neighbors[-1]) {
        auto v = i.val;
        auto lower = std::lower_bound(res.Neighbors[-1].begin(), res.Neighbors[-1].end(), v);
        if (lower != res.Neighbors[-1].end() && lower->val == v) {
            st[v] = true;
            int idx2 = lower - res.Neighbors[-1].begin();
            auto tmp = std::move(res.Neighbors[-1][idx2].csr);
            std::set_intersection(tmp.begin(), tmp.end(),
                i.csr.begin(), i.csr.end(),
                std::back_inserter(res.Neighbors[-1][idx2].csr));
        }
    }
    for (auto i = res.Neighbors[-1].begin(); i != res.Neighbors[-1].end(); ) {
        if (i->csr.empty() || st.find(i->val) == st.end() || st[i->val] == false) {
            i = res.Neighbors[-1].erase(i);
        }
        else i++;
    }
}

void DeltaGraph::cup(Node& res, Node& snap, int idx){
    for (auto & i : snap.Neighbors[idx]) {
        auto v = i.val;
        auto lower = std::lower_bound(res.Neighbors[-1].begin(), res.Neighbors[-1].end(), v);
        auto idx2 = lower - res.Neighbors[-1].begin();
        if (lower != res.Neighbors[-1].end() && lower->val == v) {
            auto tmp = std::move(res.Neighbors[-1][idx2].csr);
            std::set_union(tmp.begin(), tmp.end(),
                           i.csr.begin(), i.csr.end(),
                           std::back_inserter(res.Neighbors[-1][idx2].csr));
        } else {
            res.Neighbors[-1].insert(lower, Edge(v, i.csr));
        }
    }
}

void DeltaGraph::graphpool(int level) {
    std::cout << "GraphPool start" << std::endl;

    std::vector<uint> sons;
    sons.emplace_back(skeleton.back().son[0]);
    for (int i = 0; i < level; ++i) {
        std::vector<uint> tmp_son;
        for (auto idx : sons) {
            for (auto each_son : skeleton[idx].son) {
                if (fdif == "union") pool_cha(idx, idx, each_son, -1, each_son);
                if (fdif == "intersection") pool_cup(idx, idx, each_son, -1, each_son);
                tmp_son.emplace_back(each_son);
                skeleton[each_son].isMat= 1;
                skeleton[each_son].ednums = 0;
                for (const auto& j : skeleton[each_son].Neighbors[-1]) {
                    skeleton[each_son].ednums += j.csr.size();
                }
            }
            skeleton[idx].Neighbors.clear();
            skeleton[idx].isMat= -1;
        }
        sons = std::move(tmp_son);
        if (i + 1 == level) {
            skeleton.back().son = sons;
            skeleton.back().Neighbors.clear();
            skeleton.back().Neighbor_delta.clear();
            for (auto j : sons) {
                skeleton[j].fa = skeleton.size() - 1;
                skeleton.back().Neighbor_delta[j] = skeleton[j].ednums;
                // for (auto k : skeleton[j].Neighbors) {
                //     for (auto l : k.second) {
                //         skeleton.back().Neighbor_delta[j] += l.csr.size();
                //     }
                // }
            }
        }
    }

    std::cout << "GraphPool end" << std::endl;
}

// void dfs(bool* st, std::vector<Node>& skeleton, uint u, uint begin, uint end, int D, size_t d,
//          std::map<uint, std::map<uint, size_t>>& dist, uint depth,
//          std::map<uint, std::map<uint, std::vector<uint>>>& path, std::vector<uint> pa) {
//     if (D > depth) return;
//     if (d > dist[begin][end]) return;
//     if (skeleton[u].isMat == 1) return;
//     if (u == end) {
//         // dist[end][begin] = dist[begin][end];
//         // path[end][begin] = path[begin][end];
//         return;
//     }
//
//     // for (auto i : skeleton[u].son) {
//     //     if (!st[i]) {
//     //         st[i] = true;
//     //         path[begin][end].emplace_back(i);
//     //         dist[begin][end] += skeleton[u].Neighbor_delta[i];
//     //
//     //         dfs(st, skeleton, i, begin, end, D + 1, 0, dist, depth, path, pa);
//     //         // pa.pop_back();
//     //         st[i] = false;
//     //     }
//     // }
//     auto i = skeleton[u].fa;
//     if (!st[i]) {
//         st[i] = true;
//         // pa.emplace_back(i);
//         path[begin][end].emplace_back(i);
//         dist[begin][end] += skeleton[i].Neighbor_delta[u];
//         dfs(st, skeleton, i, begin, end, D + 1, d + skeleton[i].Neighbor_delta[u], dist, depth, path, pa);
//         // path[begin][end].pop_back();
//         // pa.pop_back();
//         st[i] = false;
//     }
// }

void getpath(std::vector<std::vector<uint>>& path, int i, int j, std::vector<uint>& pp) {
    if(i==j) return;
    if(path[i][j]==0) {
        pp.emplace_back(i);
    }
    else{
        getpath(path, i,path[i][j], pp);
        getpath(path, path[i][j],j, pp);
    }
};

void DeltaGraph::GetDist() {
    uint super_root = skeleton.size() - 1;

    std::queue<uint> q;
    std::vector<bool> vis(skeleton.size(), false);
    q.push(super_root);
    dist.assign(skeleton.size(), std::vector<unsigned long long>(skeleton.size(), LONG_LONG_MAX));
    path.assign(skeleton.size(), std::vector<uint>(skeleton.size()));
    while (!q.empty()) {
        auto p = q.front();
        q.pop();
        if (vis[p]) continue;
        vis[p] = true;
        for (auto idx : skeleton[p].son) {
            dist[p][idx] = skeleton[p].Neighbor_delta[idx];
            dist[idx][p] = dist[p][idx];
            q.push(idx);
        }
    }

    // for (int k = 0; k < skeleton.size(); ++k) {
    //     for (int i = k + 1; i < skeleton.size(); ++i) {
    //         std::cout << "[" << k << " " << i << "] " << dist[i][k] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    for (int k = 0; k < skeleton.size(); ++k) {
        for (int i = 0; i < skeleton.size(); ++i) {
            for (int j = 0; j < skeleton.size(); ++j) {
                if (dist[i][j] > dist[i][k] + dist[k][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                    path[i][j] = k;
                }
            }
        }
    }

    // for (int i = 0; i < leafs; i++) {
    //     for (int j = i + 1; j < leafs; j++) {
    //         std::cout << "[" << i << " " << j << "] " << dist[i][j] << ": ";
    //         getpath(path, i, j);
    //         std::cout << std::endl;
    //         // int tmp = j;
    //         // while (path[i][tmp]) {
    //         //     std::cout << path[i][tmp] << " ";
    //         //     tmp = path[i][tmp];
    //         // }
    //         // std::cout << std::endl;
    //     }
    // }

    // std::cout << "Get Dist leafs" << std::endl;
    // for (auto i : skeleton.back().son) {
    //     dist[i][super_root] = 0;
    //     path[i][super_root] = std::move(std::vector<uint>({i, super_root}));
    // }
    //
    // std::cout << "super_root = " << super_root << std::endl;
    // for (int i = 0; i < leafs; ++i) {
    //     path[i][super_root].emplace_back(i);
    //     dfs(st, skeleton, i, i, super_root, 0, 0, dist, depth, path, pa);
    // }
    // delete[] st;
}

int DeltaGraph::find(int u) {
    if (p[u] != u) p[u] = find(p[u]);
    return p[u];
}

std::vector<std::vector<uint>> DeltaGraph::Prim(int l, int r) {
    std::vector<uint> ve;
    ve.emplace_back(l);
    ve.emplace_back(r);
    int ll = skeleton[l].fa, rr = skeleton[r].fa;
    while (skeleton[ll].isMat != 1)
        ll = skeleton[ll].fa;
    while (skeleton[rr].isMat != 1)
        rr = skeleton[rr].fa;

    ve.emplace_back(skeleton.size() - 1);
    for (auto son : skeleton.back().son) {
        if (ll < son && son < rr)
            ve.emplace_back(son);
    }
    p.resize(ve.size());
    for (int i = 0; i < (int)ve.size(); ++i) p[i] = i;

    struct MST_EDGE {
        int a, b, w;
        MST_EDGE(int a, int b, int w) : a(a), b(b), w(w) {}
        bool operator< (const MST_EDGE& W) const {
            return w < W.w;
        }
    };

    std::vector<MST_EDGE> mst_eg;
    for (int i = 0; i < ve.size(); ++i) {
        for (int j = i + 1; j < ve.size(); ++j) {
            if (ve[i] < ve[j]) {
                mst_eg.emplace_back(i, j, dist[ve[i]][ve[j]]);
            }
            else {
                mst_eg.emplace_back(j, i, dist[ve[j]][ve[i]]);
            }
        }
    }
    sort(mst_eg.begin(), mst_eg.end());

    std::vector<std::vector<uint>> v;
    for (auto i : mst_eg) {
        int a = i.a, b = i.b;
        a = find(a);
        b = find(b);
        if (a != b) {
            std::vector<uint> pp ;
            getpath(path, ve[a], ve[b], pp);
            pp.emplace_back(ve[b]);
            p[a] = b;

            reverse(pp.begin(), pp.end());
            v.emplace_back(pp);
        }
    }
    sort(v.begin(), v.end(), [](std::vector<uint>& a, std::vector<uint>& b) {
        if (a[0] != b[0])
            return a[0] > b[0];
        return a[1] > b[1];
    });
    for (auto i : v) {
        for (auto j : i)
            std::cout << j << " ";
        std::cout << std::endl;
    }
    // std::cout << mst_eg[0].a << "->" << mst_eg[0].b << std::endl;
    // for (auto i : v[0])
    //     std::cout << i << " ";
    // std::cout << std::endl;
    // std::cout << mst_eg[1].a << "->" << mst_eg[1].b << std::endl;
    // for (auto i : v[1])
    //     std::cout << i << " ";
    // std::cout << std::endl;
    // v[2] = std::vector<uint>({skeleton.size() - 2, skeleton.size() - 1});
    // sort(v.begin(), v.end(), [](std::vector<uint>& a, std::vector<uint>& b) {
    //     return a.back() > b.back();
    // });

    // path.clear();
    return v;
}

void DeltaGraph::get_snap() {

}

void DeltaGraph::skewed(){}
void DeltaGraph::balence(){}
void DeltaGraph::Rskewed(){}
void DeltaGraph::Lskewed(){}
void DeltaGraph::mixed(){}
void DeltaGraph::empty(){}