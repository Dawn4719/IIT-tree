#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <bitset>
#include <cmath>
#include <algorithm>
#include <bit>
#include "../utils/CLI11.hpp"
#include "../utils/globals.h"
#include "../utils/types.h"
#include <boost/dynamic_bitset.hpp>
#include "../graph/graph.h"
#include "matching.h"
#include "deltagraph.h"
#include "TiLa.h"
#include <fstream>
#include <random>
#include <ctime>
#include <immintrin.h>
#include <omp.h>
#include "../utils/pod.h"

#include "staticcore.hpp"
#include "reach.hpp"

using namespace std;
int K = 18;
int level;
bool fun;
size_t EDGE_MAX = 3e4 * 256;
const int N = 100000000;
std::string query_path, dataset, input_file;
bool CHECK = true;
size_t all_edges;

struct TNode {
    TNode() : lson(-1), rson(-1), fa(-1) {
        //        bv.resize(N, false);
        dep = 0;
        id = 0;
        prex = 0;
    };
    dynamic_bitset<uint32_t> bv;
    int id;
    int prex;
    int lson;
    int rson;
    int fa;
    int dep;

    ~TNode()=default;
};

size_t edge_idx;
std::vector<std::vector<std::pair<uint, uint> > > CSR;
// std::vector<std::vector<uint>> CSR;
struct egg{
    unsigned l, r;
    int id;
    egg() = default;
    egg(unsigned l, unsigned r, int id) : l(l), r(r), id(id) {}
};
// std::vector<egg> id2edge;
std::vector<pair<int, int>> id2edge;

struct Tree {
    std::vector<TNode> tr;
};

std::vector<Tree> tree;
std::vector<uint> snap_idx;
std::vector<uint> leaf_size;
size_t sum;

auto lb(std::vector<std::pair<uint, uint> > &V, uint val) {
    size_t L = 0, R = V.size();
    while (L < R) {
        size_t MID = (L + R) >> 1;
        if (V[MID].first >= val) R = MID;
        else L = MID + 1;
    }
    return V.begin() + L;
}

// auto lb(std::vector<uint> &V, uint val) {
//     size_t L = 0, R = V.size();
//     while (L < R) {
//         size_t MID = (L + R) >> 1;
//         if (V[MID] >= val) R = MID;
//         else L = MID + 1;
//     }
//     return V.begin() + L;
// }

map<string, size_t> get_index_mem() {
    FILE *fp = fopen("/proc/self/status", "r");
    char line[128];
    map<string, size_t> res;
    while (fgets(line, 128, fp) != NULL) {
        //        if (strncmp(line, "VmPeak", 2) == 0)
        //        {
        //            cout << line << endl;
        ////            printf("当前进程占用虚拟内存大小为：%d KB\n", atoi(line + 6));
        //        }
        if (strncmp(line, "VmRSS:", 6) == 0) {
            string p = line;
            res["now"] = size_t(stoull(p.substr(6)));
            cout << line;
        }
        if (strncmp(line, "VmPeak:", 7) == 0) {
            string p = line;
            res["pk"] = size_t(stoull(p.substr(7)));
            cout << line;
        }
    }
    fclose(fp);
    return res;
}
bool useRoaring;
void build(Tree &tre, bool f) {
    auto &tr = tre.tr;
    int depth = log2(tr.size());
    if ((tr.size() & (tr.size() - 1)) == 0)
        depth++;
    else
        depth += 2;
    size_t nums = 0;
    size_t nn = tr.size();
    while (nn) {
        nums += nn;
        if (nn == 1)
            break;
        nn = (nn + 1) / 2;
    }
    //cout << "depth:" << depth << "nums:" << nums << endl;
    int cur_depth = 1;
    int n = tr.size();
    int half = depth - 2;
    if (half == -1)
        half = 0;
    leaf_size.emplace_back(1 << half);
    sum += n;
    tr.resize(nums);
    for (size_t i = 0; i < nums; ++i) tr[i].id = i, tr[i].dep = 0;
    tr[nums - 1].fa = nums - 1;
    int cnt = n;

    while (cur_depth < depth) {
        //// std::cout << "cur_depth_node_num:" << cnt << std::endl;
        int i = n - cnt + 2;
        cnt = 0;
        for (; i <= n; i += 2) {
            tr[n + cnt].lson = i - 2;
            tr[n + cnt].rson = i - 1;
            tr[n + cnt].dep = cur_depth;
            tr[i - 2].fa = n + cnt;
            tr[i - 1].fa = n + cnt;

            if (fun) {
                if (tr[i - 1].prex * 32 + tr[i - 1].bv.size() >= tr[i - 2].prex * 32 + tr[i - 2].bv.size()) {
                    tr[n + cnt].bv = tr[i - 1].bv;
                    tr[n + cnt].prex = min(tr[i - 1].prex, tr[i - 2].prex);
                    int commonSize = tr[i - 1].prex - tr[i - 2].prex;
                    if (commonSize > 0)
                        tr[n + cnt].bv.resize(tr[n + cnt].bv.size() + 32 * commonSize);

                    for (int l = tr[i - 2].bv.num_blocks() - 1, j = tr[n + cnt].bv.num_blocks() - 1 + min(0, commonSize); ~l && ~j; l--, j--)
                        tr[n + cnt].bv.m_bits[j] |= tr[i - 2].bv.m_bits[l];

                    assert(tr[n + cnt].bv.m_bits.back() != 0);
                }
                else{
                    tr[n + cnt].bv = tr[i - 2].bv;
                    tr[n + cnt].prex = min(tr[i - 1].prex, tr[i - 2].prex);
                    int commonSize = tr[i - 1].prex - tr[i - 2].prex;
                    if (commonSize > 0)
                        tr[n + cnt].bv.resize(tr[n + cnt].bv.size() + 32 * commonSize);

                    for (int l = tr[i - 1].bv.num_blocks() - 1, j = tr[n + cnt].bv.num_blocks() - 1 + min(0, commonSize); ~l && ~j; l--, j--)
                        tr[n + cnt].bv.m_bits[j] |= tr[i - 1].bv.m_bits[l];

                    assert(tr[n + cnt].bv.m_bits.back() != 0);
                }
            }
            else {
                if (tr[i - 1].prex * 32 + tr[i - 1].bv.size() <= tr[i - 2].prex * 32 + tr[i - 2].bv.size()) {
                    tr[n + cnt].bv = tr[i - 1].bv;
                    tr[n + cnt].prex = min(tr[i - 1].prex, tr[i - 2].prex);
                    int commonSize = tr[i - 2].prex - tr[i - 1].prex;
                    int l = tr[i - 2].bv.num_blocks() - 1 + min(commonSize, 0), j = tr[n + cnt].bv.num_blocks() - 1;
                    tr[n + cnt].bv.resize(tr[n + cnt].bv.size() - min(0, commonSize) * 32);
                    int zeroCnt = 0;
                    commonSize = max(0, commonSize);
                    for (; commonSize > 0; j--, commonSize--) {
                        tr[n + cnt].bv.m_bits[j] = 0;
                        zeroCnt++;
                    }

                    for (; ~l && ~j; l--, j--)
                        tr[n + cnt].bv.m_bits[j] &= tr[i - 2].bv.m_bits[l];
                    for (j = tr[n + cnt].bv.num_blocks() - 1; ~j; j--) {
                        if (tr[n + cnt].bv.m_bits[j] != 0) {
                            tr[n + cnt].prex += tr[n + cnt].bv.num_blocks() - j - 1;
                            tr[n + cnt].bv.resize((j + 1) * 32);
                            tr[n + cnt].bv.shrink_to_fit();
                            break;
                        }
                    }
                    // assert(tr[n + cnt].bv.m_bits.back() != 0);
                }
                else {
                    tr[n + cnt].bv = tr[i - 2].bv;
                    tr[n + cnt].prex = min(tr[i - 1].prex, tr[i - 2].prex);
                    int commonSize = tr[i - 1].prex - tr[i - 2].prex;
                    int l = tr[i - 1].bv.num_blocks() - 1 + min(commonSize, 0), j = tr[n + cnt].bv.num_blocks() - 1;
                    tr[n + cnt].bv.resize(tr[n + cnt].bv.size() - min(0, commonSize) * 32);
                    // tr[n + cnt].prex -= min(0, commonSize);
                    int zeroCnt = 0;
                    commonSize = max(0, commonSize);
                    for (; commonSize > 0; j--, commonSize--) {
                        tr[n + cnt].bv.m_bits[j] = 0;
                        zeroCnt++;
                    }
                    for (; ~l && ~j; l--, j--)
                        tr[n + cnt].bv.m_bits[j] &= tr[i - 1].bv.m_bits[l];

                    for (j = tr[n + cnt].bv.num_blocks() - 1; ~j; j--) {
                        if (tr[n + cnt].bv.m_bits[j] != 0) {
                            tr[n + cnt].prex += tr[n + cnt].bv.num_blocks() - j - 1;
                            tr[n + cnt].bv.resize((j + 1) * 32);
                            tr[n + cnt].bv.shrink_to_fit();
                            break;
                        }
                    }
                    // assert(tr[n + cnt].bv.m_bits.back() != 0);
                }
            }
            // cout << n + cnt << " " << tr[n + cnt].bv.count() << endl;

            cnt++;
        }
        if (i - 2 != n) {
            if (tr[n + cnt].lson == -1) {
                tr[n + cnt].lson = n - 1;
                tr[n + cnt].dep = cur_depth;
                tr[n + cnt].bv = tr[n - 1].bv;
                tr[n + cnt].prex = tr[n - 1].prex;
                tr[n - 1].fa = n + cnt;
            } else {
                tr[n + cnt].rson = n - 1;
                tr[n + cnt].dep = cur_depth;
                tr[n - 1].fa = n + cnt;
                if (fun) {
                    if (tr[n - 1].prex * 32 + tr[n - 1].bv.size() >= tr[tr[n + cnt].lson].prex * 32 + tr[tr[n + cnt].lson].bv.size()) {
                        tr[n + cnt].bv = tr[n - 1].bv;
                        tr[n + cnt].prex = min(tr[n - 1].prex, tr[tr[n + cnt].lson].prex);
                        int commonSize = tr[n - 1].prex - tr[tr[n + cnt].lson].prex;
                        tr[n + cnt].bv.resize(tr[n + cnt].bv.size() + 32 * commonSize);
                        for (int l = tr[tr[n + cnt].lson].bv.num_blocks() - 1, j = tr[n + cnt].bv.num_blocks() - 1; ~l && ~j; l--, j--)
                            tr[n + cnt].bv.m_bits[j] |= tr[tr[n + cnt].lson].bv.m_bits[l];
                        assert(tr[n + cnt].bv.m_bits.back() != 0);
                    }
                    else if (tr[n - 1].prex * 32 + tr[n - 1].bv.size() < tr[tr[n + cnt].lson].prex * 32 + tr[tr[n + cnt].lson].bv.size()) {
                        tr[n + cnt].bv = tr[tr[n + cnt].lson].bv;
                        tr[n + cnt].prex = min(tr[n - 1].prex, tr[tr[n + cnt].lson].prex);
                        int commonSize = tr[tr[n + cnt].lson].prex - tr[n - 1].prex;
                        tr[n + cnt].bv.resize(tr[n + cnt].bv.size() + 32 * commonSize);
                        for (int l = tr[n - 1].bv.num_blocks() - 1, j = tr[n + cnt].bv.num_blocks() - 1; ~l && ~j; l--, j--)
                            tr[n + cnt].bv.m_bits[j] |= tr[n - 1].bv.m_bits[l];
                        assert(tr[n + cnt].bv.m_bits.back() != 0);
                    }
                }
                else {
                    if (tr[n - 1].prex * 32 + tr[n - 1].bv.size() <= tr[tr[n + cnt].lson].prex * 32 + tr[tr[n + cnt].lson].bv.size()) {
                        tr[n + cnt].bv = tr[n - 1].bv;
                        tr[n + cnt].prex = max(tr[n - 1].prex, tr[tr[n + cnt].lson].prex);
                        int commonSize = tr[n - 1].prex - tr[tr[n + cnt].lson].prex;

                        int l = tr[n - 1].bv.num_blocks() - 1 + max(0, commonSize), j = tr[n + cnt].bv.num_blocks() - 1;
                        tr[n + cnt].bv.resize(tr[n + cnt].bv.size() - min(0, commonSize) * 32);
                        int zeroCnt = 0;
                        commonSize = max(0, commonSize);
                        for (; commonSize > 0; j--, commonSize--) {
                            tr[n + cnt].bv.m_bits[j] = 0;
                            zeroCnt++;
                        }

                        for (; ~l && ~j; l--, j--)
                            tr[n + cnt].bv.m_bits[j] &= tr[tr[n + cnt].lson].bv.m_bits[l];
                        for (j = tr[n + cnt].bv.num_blocks() - 1; ~j; j--) {
                            if (tr[n + cnt].bv.m_bits[j] != 0) {
                                tr[n + cnt].prex += tr[n + cnt].bv.num_blocks() - j - 1;
                                tr[n + cnt].bv.resize((j + 1) * 32);
                                tr[n + cnt].bv.shrink_to_fit();
                                break;
                            }
                        }
                        // assert(tr[n + cnt].bv.m_bits.back() != 0);
                    }
                    else {
                        tr[n + cnt].bv = tr[tr[n + cnt].lson].bv;
                        tr[n + cnt].prex = max(tr[n - 1].prex, tr[tr[n + cnt].lson].prex);
                        int commonSize = tr[tr[n + cnt].lson].prex - tr[n - 1].prex;

                        int l = tr[n - 1].bv.num_blocks() - 1 + min(0, commonSize), j = tr[n + cnt].bv.num_blocks() - 1;
                        tr[n + cnt].bv.resize(tr[n + cnt].bv.size() - min(0, commonSize) * 32);
                        int zeroCnt = 0;
                        commonSize = max(0, commonSize);
                        for (; commonSize > 0; j--, commonSize--) {
                            tr[n + cnt].bv.m_bits[j] = 0;
                            zeroCnt++;
                        }
                        for (; ~l && ~j; l--, j--)
                            tr[n + cnt].bv.m_bits[j] &= tr[n - 1].bv.m_bits[l];
                        for (j = tr[n + cnt].bv.num_blocks() - 1; ~j; j--) {
                            if (tr[n + cnt].bv.m_bits[j] != 0) {
                                tr[n + cnt].prex += tr[n + cnt].bv.num_blocks() - j - 1;
                                tr[n + cnt].bv.resize((j + 1) * 32);
                                tr[n + cnt].bv.shrink_to_fit();
                                break;
                            }
                        }
                        // assert(tr[n + cnt].bv.m_bits.back() != 0);
                    }
                }
            }
            // cout << n + cnt << " " << tr[n + cnt].bv.count() << endl;
            cnt++;
        }
        n += cnt;
        cur_depth++;
    }
}

void DG_method(bool fun) {
    auto start = Get_Time();
    Graph query_graph{};
    // query_graph.LoadFromFile(query_path);
    // query_graph.PrintMetaData();
    int k = 4;
    // K--;
    DeltaGraph DG(k, K);
    if (fun) DG.fdif = "union";
    else DG.fdif = "intersection";
    std::cout << "----------- Building DeltaGraph ------------" << std::endl;
    DG.build(query_path, K);

    cout << "Build Time: " << Duration(start) / 1000 << endl;
    std::cout << "Memory: " << mem::getValue() / 1024 << "mb" << endl;

    level = 2;
    if (DG.depth <= 2) level = 1;
    if (level != 0)
        DG.graphpool(level);

    get_index_mem();
    cout << input_file << endl;
    fstream input(input_file, ios::in);
    fstream CSV("../result.csv", ios::app);
    uint L, R;
    int CNT = 0;

    auto getdist_time = Get_Time();
    std::cout << "----------- GetDist ------------" << std::endl;
    DG.GetDist();
    Print_Time("getDistTime: ", getdist_time);

    CSV << "DG" << ',' << fun << ',' << dataset << ',' << K << ',' << Duration(start) / 1000 << ',' << mem::getValue() / 1024 << ',';
    std::cout << "Build Over: " << mem::getValue() / 1024 << "Mb" << std::endl;

    size_t intSz = 0;
    for (const auto& i : DG.skeleton) {
        intSz += i.son.size();
        intSz += 3;
        intSz += i.Neighbors.size(); // map
        intSz += 17 * 1.0 / 4 * i.Neighbor_delta.size();
        for (const auto& j : i.Neighbors) {
            intSz += j.second.size();
            for (auto k : j.second) {
                intSz += k.csr.size();
            }
        }
    }

    auto query_start_time = Get_Time();
    size_t asd = 0;
    while (input >> L >> R) {
        auto begin_q = Get_Time();
        cout << "Query: " << L << " " << R << endl;
        CNT += 1;

        // auto prim_time = Get_Time();
        // std::cout << "----------- Prim ------------" << std::endl;
        // auto p = DG.Prim(L, R);
        // // for (const auto &i: p) {
        // //     for (auto j: i) {
        // //         cout << j << " ";
        // //     }
        // //     cout << endl;
        // // }
        // Print_Time("primTime: ", prim_time);

        double time1 = 0;
        double time2 = 0;
        Node res;
        if (level == 0)
            res.Neighbors[-1] = DG.skeleton[DG.skeleton.back().son[0]].Neighbors[-1];

        if (DG.dist[L][DG.skeleton.size() - 1] < DG.dist[R][DG.skeleton.size() - 1]) {
            auto& p = DG.path[L][DG.skeleton.size() - 1];
            for (size_t j = p.size() - 1; j; j--) {
                if (level != 0 && DG.skeleton[p[j]].isMat == 1) {
                    res.Neighbors[-1] = DG.skeleton[p[j]].Neighbors[-1];
                }
                if (fun) {DG.cha(DG.skeleton[p[j]], res, p[j - 1]);}
                else {DG.cup(res, DG.skeleton[p[j]], p[j - 1]);}
            }
        }
        else {
            auto& p = DG.path[R][DG.skeleton.size() - 1];
            for (size_t j = p.size() - 1; j; j--) {
                if (level != 0 && DG.skeleton[p[j]].isMat == 1) {
                    res.Neighbors[-1] = DG.skeleton[p[j]].Neighbors[-1];
                }
                if (fun) {DG.cha(DG.skeleton[p[j]], res, p[j - 1]);}
                else {DG.cup(res, DG.skeleton[p[j]], p[j - 1]);}
            }
        }
        auto& p = DG.path[L][R];
        for (size_t j = L; j < R - 1; j++) {
            if (fun) DG.cup(res, DG.skeleton[p[j]], p[j + 1]);
            else DG.cha(DG.skeleton[p[j]], res, p[j + 1]);
        }

        // cout << "time1: " << time1 << "ms" << endl;
        // cout << "time2: " << time2 << "ms" << endl;

        //        fstream f("../res.txt", ios::app);
        for (const auto &i: res.Neighbors[-1]) all_edges += i.csr.size();

        Print_Time("queryTime: ", begin_q);
        std::cout << "----------------------------------------------Res: " << all_edges << std::endl;
        cout << "peek memoty: " << mem::getValue() / 1024 << "Mb" << std::endl;

        all_edges = 0;
        if (CNT == 15) {
            CSV << Duration(query_start_time) / 1000 << '\t';
        }
    }
    input.close();
    std::cout << "Peak Mem: " << mem::getValue() / 1024 << "mb" << '\t' << "Query time:" << Duration(query_start_time) / 1000
            << "s" << '\t' << "ALL Time: " << Duration(start) / 1000 << std::endl;
    CSV << Duration(query_start_time) / 1000 << ',' << mem::getValue() / 1024;
    // if (level != 0)
    //     CSV << ',' << level;
    CSV << "," << intSz << std::endl;
    CSV.close();
}

void Base_method(bool fun) {
    auto start = Get_Time();
    std::vector<Graph> snaps(K);
    size_t csr_edge_count = 0;
    //    Graph* snap = new Graph[K];
    for (int snap_i = 0; snap_i < K; ++snap_i) {
        snaps[snap_i].LoadFromFile(query_path + std::to_string(snap_i), true);
        sort(snaps[snap_i].BSEdge.begin(), snaps[snap_i].BSEdge.end());
    }
    cout << "Build Time: " << Duration(start) / 1000 << endl;
    std::cout << "Memory: " << mem::getValue() / 1024 << "mb" << endl;
    get_index_mem();

    uint L, R;
    cout << input_file << endl;
    fstream input(input_file, ios::in);
    fstream CSV("../result.csv", ios::app);

    int CNT = 0;
    CSV << "BS" << ',' << fun << ',' <<dataset << ',' << K << ',' << Duration(start) / 1000 << ',' << mem::getValue() / 1024 << endl;

    size_t intSz = 0;
    for (int snap_i = 0; snap_i < K; ++snap_i) {
        intSz += snaps[snap_i].BSEdge.size() * 2;
    }

    auto query_time = Get_Time();
    auto query_start_time = Get_Time();
    while (input >> L >> R) {
        cout << "Query: " << L << " " << R;
        CNT++;
        std::vector<std::pair<int, int>> res;
        for (int snap_i = L; snap_i <= R; ++snap_i) {
            if (fun) {
                for (int i = 0; i < snaps[snap_i].neighbors_.size(); ++i) {
                    std::vector<pair<int, int>> tmp;
                    std::set_union(res.begin(), res.end(),
                                   snaps[snap_i].BSEdge.begin(), snaps[snap_i].BSEdge.end(),
                                   std::back_inserter(tmp));
                    res = move(tmp);
                }
            }
            else {
                for (int i = 0; i < snaps[snap_i].neighbors_.size(); ++i) {
                    if (snap_i == L) {
                        res = snaps[snap_i].BSEdge;
                    } else {
                        vector<pair<int, int>> tmp;
                        std::set_intersection(res.begin(), res.end(),
                                              snaps[snap_i].BSEdge.begin(), snaps[snap_i].BSEdge.end(),
                                              std::back_inserter(tmp));
                        res = move(tmp);
                    }
                }
            }
        }

        for (const auto &i: res) {
            all_edges++;
        }

        std::cout << "  ----------------------------------------------Res: " << all_edges << std::endl;
        //        CSV << "TR" << ',' << L << ',' << R << ',' << all_edges << endl;
        all_edges = 0;
        // cout << "peek memoty: " << mem::getValue() / 1024 << "Mb" << std::endl;

        if (CNT == 15) {
            CSV << Duration(query_start_time) / 1000 << ',';
            // cout << "BS" << '\t' << mem::getValue() / 1024 << "mb" << '\t' << Duration(query_start_time) / 1000 << "s"
            //         << std::endl;
        }
    }
    input.close();

    std::cout << "Peak Mem: " << mem::getValue() / 1024 << "mb" << '\t' << "Query time:" << Duration(query_time) / 1000
            << "s" << '\t' << "ALL Time: " << Duration(start) / 1000 << std::endl;
    CSV << Duration(query_start_time) / 1000 << ',' << mem::getValue() / 1024 << ',' << intSz * 4 << std::endl;

    CSV.close();
    std::cout << "Peak Mem: " << mem::getValue() / 1024 << "mb";
}

void TiLa_method(bool fun) {
    auto start = Get_Time();
    TiLa_Graph tila;
    tila.build(K, query_path);
    cout << "Build Time: " << Duration(start) / 1000 << endl;
    std::cout << "Memory: " << mem::getValue() / 1024 << "mb" << endl;
    get_index_mem();

    uint L, R;
    fstream input(input_file, ios::in);
    fstream CSV("../result.csv", ios::app);

    CSV << "TiLa" << ',' << fun << ',' << dataset << ',' << K << ',' << Duration(start) / 1000 << ',' << mem::getValue() / 1024 << ',';

    int CNT = 0;
    size_t intSz = 0;
    for (auto& i : tila.TiLaEdge) {
        for (auto& j: i) {
            intSz ++;
            intSz += j.lifespan.num_blocks();
        }
    }

    auto query_time = Get_Time();
    auto query_start_time = Get_Time();
    cout << "begin Query" << endl;
    vector<pair<int, int> > res;
    res.reserve(10000);
    map<int, vector<int>> vmp;
    while (input >> L >> R) {
        cout << "Query: " << L << " " << R;
        CNT++;
        res.clear();
        res.reserve(100000);
        bool pas = true;
        dynamic_bitset<uint32_t> bit(R - L + 1);
        for (int mp = 0; mp < tila.TiLaEdge.size(); mp++) {
            for (const auto &ed: tila.TiLaEdge[mp]) {
                string s;
                to_string(ed.lifespan, s);
                s = s.substr(s.size() - 1 - R, R - L + 1);
                // s = s.substr(L, R - L + 1);
                if (fun && s.find('1') != string::npos) {
                    res.emplace_back(mp, ed.nei_id);
                }
                if (!fun && s.find('0') == string::npos) {
                    res.emplace_back(mp, ed.nei_id);
                    pas = false;
                }
            }
        }
        // sort(res.begin(), res.end());
        // res.erase(unique(res.begin(), res.end()), res.end());
        int all_edges = 0;
        for (auto& i : res )
            all_edges++;
        std::cout << "  ----------------------------------------------Res: " << all_edges << std::endl;

        if (CNT == 15) {
            CSV << Duration(query_start_time) / 1000 << ',';
            // cout << "TiLa" << ',' << mem::getValue() / 1024 << "mb" << ',' << Duration(query_start_time) / 1000 << "s"
            //         << std::endl;
        }
    }
    input.close();

    std::cout << "Peak Mem: " << mem::getValue() / 1024 << "mb" << '\t' << "Query time:" << Duration(query_time) / 1000
            << "s" << '\t' << "ALL Time: " << Duration(start) / 1000 << std::endl;
    CSV << Duration(query_start_time) / 1000 << ',' << mem::getValue() / 1024 << "," << intSz * 4 << std::endl;

    CSV.close();
    std::cout << "Peak Mem: " << mem::getValue() / 1024 << "mb";
}

mt19937 rd(time(nullptr));

void getK() {
    if (dataset == "mo1H") K = 34920;
        if (dataset == "mo12H") K = 4693;
        if (dataset == "mo1D") K = 2350;
        if (dataset == "mo3D") K = 784;
        if (dataset == "mo7D") K = 336;
        if (dataset == "mo15D") K = 157;
        if (dataset == "mo1M") K = 79;
        if (dataset == "mo2M") K = 40;
        if (dataset == "mo4M") K = 20;
        if (dataset == "au1H") K = 41804;
        if (dataset == "au12H") K = 4080;
        if (dataset == "au1D") K = 2046;
        if (dataset == "au3D") K = 684;
        if (dataset == "au7D") K = 294;
        if (dataset == "au15D") K = 138;
        if (dataset == "au1M") K = 70;
        if (dataset == "au2M") K = 36;
        if (dataset == "au4M") K = 19;
        if (dataset == "su1H") K = 56232;
        if (dataset == "su12H") K = 4892;
        if (dataset == "su1D") K = 2493;
        if (dataset == "su3D") K = 866;
        if (dataset == "su7D") K = 387;
        if (dataset == "su15D") K = 185;
        if (dataset == "su1M") K = 93;
        if (dataset == "su2M") K = 47;
        if (dataset == "su4M") K = 24;
        if (dataset == "so1H") K = 66445;
        if (dataset == "so12H") K = 5548;
        if (dataset == "so1D") K = 2775;
        if (dataset == "so3D") K = 925;
        if (dataset == "so7D") K = 397;
        if (dataset == "so15D") K = 185;
        if (dataset == "so1M") K = 93;
        if (dataset == "so2M") K = 47;
        if (dataset == "so4M") K = 24;
}

inline void cal(dynamic_bitset<uint32_t>& lres, int& lresPrex, dynamic_bitset<uint32_t>& bt, int& btPrex) {
    dynamic_bitset<uint32_t> res;
    int resPrex = 0;
    if (lresPrex * 32 + lres.size() >= btPrex * 32 + bt.size()) {
        res = lres;
        resPrex = min(lresPrex, btPrex);
        int commonSize = lresPrex - btPrex;
        if (commonSize > 0)
            res.resize(res.size() + 32 * commonSize);

        for (int l = bt.num_blocks() - 1, j = res.num_blocks() - 1 + min(0, commonSize); ~l && ~j; l--, j--)
            res.m_bits[j] |= bt.m_bits[l];

        // assert(res.m_bits.back() != 0);
    }
    else{
        res = bt;
        resPrex = min(lresPrex, btPrex);
        int commonSize = lresPrex - btPrex;
        if (commonSize > 0)
            res.resize(res.size() + 32 * commonSize);

        for (int l = lres.num_blocks() - 1, j = res.num_blocks() - 1 + min(0, commonSize); ~l && ~j; l--, j--)
            res.m_bits[j] |= lres.m_bits[l];

        // assert(res.m_bits.back() != 0);
    }
    lres = res;
    lresPrex = resPrex;
}

int main(int argc, char *argv[]) {
    /*dynamic_bitset<uint32_t> asd;
    dynamic_bitset<uint32_t> asd2;
    asd.resize(320);
    asd2.resize(320 - 32 - 32);

    asd.set(320 - 32 + 1);
    asd.set(320 - 32 - 32);
    // asd.set(320 - 32 - 32 - 32 + 3);

    asd2.set(320 - 32 - 32 - 30);
    asd2.set(320 - 32 - 32 - 30 - 30);
    asd2.set(320 - 32 - 32 - 32 - 32 - 29);

    // 2 1 8  0 0 0 0 0 0 0
    //     4 16 0 0 0 0 0 0

    cout << "asd: " << asd.size() << " " << asd.m_num_bits << " " << asd.num_blocks() << " " << asd.m_bits.size() << endl;
    cout << "asd2: " << asd2.size() << " " << asd2.m_num_bits << " " << asd2.num_blocks() << " " << asd2.m_bits.size() << endl;

    reverse(asd.m_bits.begin(), asd.m_bits.end());
    for (int i = 0; i < asd.num_blocks(); i++) {
        cout << asd.m_bits[i] << " ";
    }
    cout << endl;

    reverse(asd2.m_bits.begin(), asd2.m_bits.end());
    for (int i = 0; i < asd2.num_blocks(); i++) {
        cout << asd2.m_bits[i] << " ";
    }
    cout << endl;
    int asdprex1, asdprex2;
    for (int j = asd.num_blocks() - 1; ~j; j--) {
        if (asd.m_bits[j] != 0) {
            cout << "prex1: " << asd.num_blocks() - j - 1 << endl;
            asdprex1 = asd.num_blocks() - j - 1;
            asd.resize((j + 1) * sizeof(uint32_t) * 8);
            asd.shrink_to_fit();
            break;
        }
    }
    for (int j = asd2.num_blocks() - 1; ~j; j--) {
        if (asd2.m_bits[j] != 0) {
            cout << "prex2: " << asd2.num_blocks() - j - 1 << endl;
            asdprex2 = asd2.num_blocks() - j - 1;
            asd2.resize((j + 1) * sizeof(uint32_t) * 8);
            asd.shrink_to_fit();
            break;
        }
    }

    cout << "asd: " << asd.size() << " " << asd.num_blocks() << " " << asd.m_bits.size() << endl;
    cout << "asd2: " << asd2.size() << " " << asd2.num_blocks() << " " << asd2.m_bits.size() << endl;

    for (int i = 0; i < asd.num_blocks(); i++) {
        cout << asd.m_bits[i] << " ";
    } cout << endl;

    for (int i = 0; i < asd2.num_blocks(); i++) {
        cout << asd2.m_bits[i] << " ";
    } cout << endl;

    dynamic_bitset<uint32_t> res;
    int resPrex;
    if (asdprex1 >= asdprex2) {
        // 2 1 0  0 0 0 0 0 0 0
        //     4 16 8 0 0 0 0 0
        res = asd;
        int commonSize = asdprex1 - asdprex2;
        cout << "commonSize: " << commonSize << endl;
        res.resize(res.size() + 32 * commonSize);
        for (int i = 0; i < res.num_blocks(); i++) {
            cout << res.m_bits[i] << " ";
        }
        cout << endl;
        for (int i = asd2.num_blocks() - 1, j = res.num_blocks() - 1; ~i && ~j; i--, j--) {
            cout << i << " " << j << endl;
            res.m_bits[j] |= asd2.m_bits[i];
        }
        cout << endl;

        resPrex = asdprex2;
    }
    else if (asdprex1 < asdprex2) {
        //     4 16 0 0 0 0 0 0
        // 2 1 0  0 0 0 0 0 0 0
        res = asd2;
        int commonSize = asdprex2 - asdprex1;
        cout << "commonSize: " << commonSize << endl;
        res.resize(res.size() + 32 * commonSize);
        for (int i = 0; i < res.num_blocks(); i++) {
            cout << res.m_bits[i] << " ";
        }
        cout << endl;
        for (int i = asd.num_blocks() - 1, j = res.num_blocks() - 1; ~i && ~j; i--, j--) {
            cout << i << " " << j << endl;
            res.m_bits[j] |= asd.m_bits[i];
        }
        cout << endl;

        resPrex = asdprex1;
    }

    for (int i = 0; i < res.num_blocks(); i++) {
        cout << res.m_bits[i] << " ";
    }
    for (int i = 0; i < resPrex; ++i)
        cout << 0 << " ";
    cout << endl;
    cout << res.num_blocks() + resPrex << endl;

    return 0;*/
    auto memst = mem::getValue();
    cout << memst << endl;

    time_t timep;
    time(&timep);
    printf("%s", ctime(&timep));
    std::cout << "Peak Mem: " << mem::getValue() / 1024 << "mb" << std::endl;
    CLI::App app{"App description"};
    string method = "1";
    dataset = "mo4M";
    level = 0;
    fun = 1;

    app.add_option("-d,--dataset", dataset, "query graph path")->required();
    app.add_option("-m,--method", method, "method")->required();
    app.add_option("-f,--fun", fun, "method")->required();
    // app.add_option("-k,--K", K, "K");
    // app.add_option("-l,--mat", level, "level");
    // app.add_option("-e,--edgemax", EDGE_MAX, "edgemax");
    CLI11_PARSE(app, argc, argv);

    getK();
    EDGE_MAX = 1e5;

    if (dataset[0] == 's' && dataset[1] == 'o') {
        EDGE_MAX = 6e6;
    }

    query_path = "../dataset/" + dataset + "/q";
    input_file = "../dataset/" + dataset + "/input.txt";

    // input_file = "../dataset/" + dataset + "/inputcore.txt";

    std::chrono::high_resolution_clock::time_point start, lstart;
    cout << query_path << " " << input_file << " " << method << " K=" << K << " " << EDGE_MAX << endl;
    start = Get_Time();
    std::cout << "----------- Loading graphs ------------" << std::endl;
    if (method == "2") {
        std::cout << "delta graph" << std::endl;
        DG_method(fun);
        return 0;
    }
    if (method == "0") {
        std::cout << "base" << std::endl;
        Base_method(fun);
        return 0;
    }
    if (method == "3") {
        std::cout << "TiLa" << std::endl;
        TiLa_method(fun);
        return 0;
    }
    std::cout << "tree" << std::endl;
    std::cout << "Peak Mem: " << mem::getValue() / 1024 << "mb" << std::endl;

    int cnt = 1;
    size_t cur_edges = 0;
    size_t csr_edge_count = 0;
    size_t snap_sum = 0;

    snap_idx.emplace_back(0);
    tree.resize(1);
    for (int snap_i = 0; snap_i < K; ++snap_i, ++cnt) {
        Graph snap;
        // snap.LoadFromFile(query_path + std::to_string(snap_i));
        string path = query_path + std::to_string(snap_i);
        if (!io::file_exists(path.c_str()))
        {
            std::cout << "Failed to open: " << path << std::endl;
            exit(-1);
        }
        vector<int> vlabels_;
        // vector<vector<int>> neighbors_;
        tree.back().tr.resize(cnt);
        auto &b = tree.back().tr.back().bv;
        /*b.resize(snap.edge_count_);*/
        tree.back().tr.back().lson = -1;
        tree.back().tr.back().rson = -1;
        tree.back().tr.back().fa = -1;
        size_t edge_count_ = 0;
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
                uint id, label;
                ifs >> id >> label;
                if (id >= vlabels_.size())
                {
                    vlabels_.resize(id + 1, NOT_EXIST);
                    vlabels_[id] = label;
                    // neighbors_.resize(id + 1);
                }
            }
            else
            {
                uint v1, v2;
                ifs >> v1 >> v2;
                // AddEdge(from_id, to_id);
                v1 = vlabels_[v1];
                v2 = vlabels_[v2];
                cur_edges++;
                edge_count_++;
                if (v1 >= CSR.size()) CSR.resize(v1 + 1);

                auto lower = lb(CSR[v1], v2);
                if (lower != CSR[v1].end() && (*lower).first == v2) {
                    if ((*lower).second > b.size()) b.resize((*lower).second);
                        // b.resize(((*lower).second + 31) / 32 * (*lower).second);
                    b.set((*lower).second - 1);
                } else {
                    CSR[v1].insert(lower, {v2, ++csr_edge_count});
                    if (csr_edge_count > b.size()) b.resize(csr_edge_count);
                        // b.resize((csr_edge_count + 31) / 32 * csr_edge_count);
                    b.set(csr_edge_count - 1);
                    id2edge.resize(id2edge.size() + 1);
                    id2edge.back() = {v2, v1};
                }

                // auto lower = lower_bound(CSR[v1].begin(), CSR[v1].end(), v2);
                // if (lower != CSR[v1].end() && (*lower) == v2) {
                //     if ( > b.size()) b.resize((*lower));
                //     b.set((*lower) - 1);
                // } else {
                //     csr_edge_count++;
                //     if (csr_edge_count > b.size()) b.resize(csr_edge_count);
                //     b.set(csr_edge_count - 1);
                //     id2edge.resize(id2edge.size() + 1);
                //     // id2edge.reserve(id2edge.size() + 1);
                //     // auto p = lower - CSR[v1].begin();
                //     // cout << lower - CSR[v1].begin() << endl;
                //     id2edge.back() = {v1, static_cast<unsigned>((size_t)(lower - CSR[v1].begin())), (int)csr_edge_count};
                //     CSR[v1].insert(lower, v2);
                // }
            }
        }
        ifs.close();

        b.resize((b.size() + 31) / 32 * 32);
        // cout << edge_count_ << " " << b.count() << " " << b.num_blocks() << "->";

        reverse(b.m_bits.begin(), b.m_bits.end());
        for (int j = b.num_blocks() - 1; ~j; j--) {
            if (b.m_bits[j] != 0) {
                tree.back().tr.back().prex = b.num_blocks() - j - 1;
                b.resize((j + 1) * sizeof(uint32_t) * 8);
                b.shrink_to_fit();
                break;
            }
        }
        // cout << b.num_blocks() << " " << b.count() << " " << tree.back().tr.back().prex << endl;

        /*for (int i = 0; i < snap.neighbors_.size(); ++i) {
            uint v1 = snap.GetVertexLabel(i);
            for (auto j: snap.neighbors_[i]) {
                uint v2 = snap.GetVertexLabel(j);
                if (v1 >= CSR.size()) CSR.resize(v1 + 1);

                auto lower = lb(CSR[v1], v2);
                if (lower != CSR[v1].end() && (*lower).first == v2) {
                    if ((*lower).second > b.size()) b.resize((*lower).second);
                    b.set((*lower).second - 1);
                } else {
                    CSR[v1].insert(lower, {v2, ++csr_edge_count});
                    if (csr_edge_count > b.size()) b.resize(csr_edge_count);
                    b.set(csr_edge_count - 1);
                    id2edge.resize(id2edge.size() + 1);
                    id2edge.back() = {v2, v1};
                }
            }
        }*/
        std::cout << "Peak Mem: " << mem::getValue() / 1024 << "mb" << std::endl;
        if (cur_edges >= EDGE_MAX) {
            std::cout << "----------- Building tree ------------" << std::endl;
            cout << "tree size" << " " << tree.back().tr.size() << endl;
            snap_sum += tree.back().tr.size();
            snap_idx.resize(snap_idx.size() + 1);
            snap_idx.back() = (snap_sum);
            build(tree.back(), fun);
            cnt = 0;
            cur_edges = 0;
            if (snap_i != K) tree.resize(tree.size() + 1);
            std::cout << "Peak Mem: " << mem::getValue() / 1024 << "mb" << std::endl;
        }
    }
    if (cur_edges) {
        std::cout << "----------- Building tree ------------" << std::endl;
        cout << "tree size" << " " << tree.back().tr.size() << endl;
        snap_sum += tree.back().tr.size();
        snap_idx.resize(snap_idx.size() + 1);
        snap_idx.back() = (snap_sum);
        build(tree.back(), fun);
        cur_edges = 0;
        std::cout << "Peak Mem: " << mem::getValue() / 1024 << "mb" << std::endl;
    }
    // sort(id2edge.begin(), id2edge.end(), [](egg& a, egg& b) {
    //     return a.id < b.id;
    // });

    std::cout << "Peak Mem: " << mem::getValue() / 1024 << "mb" << std::endl;
    fstream CSV2("../result.csv", ios::app);
    CSV2 << "TR" << ',' << fun << ',' << dataset << ',' << K << ',' << Duration(start) / 1000 << ',' << mem::getValue() / 1024 << endl;
    CSV2.close();

    // return 0;
    // id2edge.resize(id2edge.size() + 1);
    // id2edge.back() = {v2, v1};

    cout << "Build Time: " << Duration(start) / 1000 << endl;
    std::cout << "Memory: " << mem::getValue() / 1024 << "mb" << endl;

    std::cout << "----------- Snap Query ------------" << std::endl;
    uint L, R;

    cout << input_file << endl;
    fstream input(input_file, ios::in);
    fstream CSV("../core.csv", ios::app);

    int CNT = 0;
    CSV << "TR" << ',' << fun << ',' << dataset << ',' << K << ',' << Duration(start) / 1000 << ',' << mem::getValue() / 1024 << ",";
    size_t intSz = 0;
    for (auto& i : tree) {
        for (auto& j : i.tr) {
            intSz += 5;
            intSz += j.bv.num_blocks();
        }
    }

    auto query_time = Get_Time();
    auto query_start_time = Get_Time();
    size_t one;
    int* ls = new int[4000000];
    int k;
    while (input >> L >> R) {
        cout << "Query: " << L << " " << R << endl;;
        assert(L < K);
        assert(R < K);
        CNT++;
        uint Ltree, Rtree; // L,R: snap idx Ltree,Rtree: tree idx about snap
        auto lower1 = std::lower_bound(snap_idx.begin(), snap_idx.end(), L);
        if (*lower1 != L) Ltree = lower1 - snap_idx.begin() - 1;
        else Ltree = lower1 - snap_idx.begin();
        auto lower2 = std::lower_bound(snap_idx.begin(), snap_idx.end(), R);
        if (*lower2 != R) Rtree = lower2 - snap_idx.begin() - 1;
        else Rtree = lower2 - snap_idx.begin();
        cout << L << " " << Ltree << " " << R << " " << Rtree << " ";
        dynamic_bitset<uint32_t> lres;
        int lresPrex = 0;
        if (Ltree == Rtree) { // 一起往上跳
            auto &tr = tree[Ltree].tr;
            int lidx = L - snap_idx[Ltree], ridx = R - snap_idx[Rtree];
            // cout << snap_idx[Ltree] << " " << snap_idx[Rtree] << endl;
            if (ridx - lidx + 1 == snap_idx[Rtree + 1] - snap_idx[Ltree]) {
                lres = tr.back().bv;
                // cout << "quick " << tr.back().id << endl;
                goto RES;
            }

            while (true) {
                if (lidx + 1 == ridx) {
                    if (tr[lidx].fa != tr[ridx].fa) {
                        if (lres.empty()) {
                            lres = tr[lidx].bv;
                            lresPrex = tr[lidx].prex;
                            int sz = -1;
                            if (fun) {
                                // if (lres.size() > tr[ridx].bv.size()) sz = tr[ridx].bv.size(), tr[ridx].bv.resize(lres.size());
                                // else if (lres.size() < tr[ridx].bv.size()) lres.resize(tr[ridx].bv.size());
                                // lres |= tr[ridx].bv;
                                cal(lres, lresPrex, tr[ridx].bv, tr[ridx].prex);
                            }
                            else {
                                lres.resize(tr[ridx].bv.size());
                                lres &= tr[ridx].bv;
                            }
                            if (sz != -1) tr[ridx].bv.resize(sz);
                        }
                        else {
                            int sz = -1;
                            if (fun) {
                                // if (lres.size() > tr[lidx].bv.size()) sz = tr[lidx].bv.size(), tr[lidx].bv.resize(lres.size());
                                // else if (lres.size() < tr[lidx].bv.size()) lres.resize(tr[lidx].bv.size());
                                // lres |= tr[lidx].bv;
                                cal(lres, lresPrex, tr[lidx].bv, tr[lidx].prex);
                            }
                            else {
                                lres.resize(tr[lidx].bv.size());
                                lres &= tr[lidx].bv;
                            }
                            if (sz != -1) tr[lidx].bv.resize(sz);

                            sz = -1;
                            if (fun) {
                                // if (lres.size() > tr[ridx].bv.size()) sz = tr[ridx].bv.size(), tr[ridx].bv.resize(lres.size());
                                // else if (lres.size() < tr[ridx].bv.size()) lres.resize(tr[ridx].bv.size());
                                // lres |= tr[ridx].bv;
                                cal(lres, lresPrex, tr[ridx].bv, tr[ridx].prex);
                            }
                            else {
                                lres.resize(tr[ridx].bv.size());
                                lres &= tr[ridx].bv;
                            }
                            if (sz != -1) tr[ridx].bv.resize(sz);
                        }
                        break;
                    }
                }
                if (lidx == ridx) {
                    if (lres.empty()) {
                        lres = tr[lidx].bv;
                        lresPrex = tr[lidx].prex;
                    }
                    else {
                        if (fun) {
                            // if (lres.size() < tr[lidx].bv.size()) lres.resize(tr[lidx].bv.size());
                            // else if (lres.size() > tr[lidx].bv.size()) tr[lidx].bv.resize(lres.size());
                            // lres |= tr[lidx].bv;
                            cal(lres, lresPrex, tr[lidx].bv, tr[lidx].prex);
                        }
                        else {
                            lres.resize(tr[lidx].bv.size());
                            lres &= tr[lidx].bv;
                        }
                    }
                    break;
                }
                if (tr[tr[lidx].fa].rson == lidx) {
                    if (lres.empty()) {
                        lres = tr[lidx].bv;
                        lresPrex = tr[lidx].prex;
                    }
                    else {
                        if (fun) {
                            // if (lres.size() < tr[lidx].bv.size()) lres.resize(tr[lidx].bv.size());
                            // else if (lres.size() > tr[lidx].bv.size()) tr[lidx].bv.resize(lres.size());
                            // lres |= tr[lidx].bv;
                            cal(lres, lresPrex, tr[lidx].bv, tr[lidx].prex);
                        }
                        else {
                            lres.resize(tr[lidx].bv.size());
                            lres &= tr[lidx].bv;
                        }
                    }
                    lidx = tr[lidx].fa + 1;
                }
                else lidx = tr[lidx].fa;
                if (tr[tr[ridx].fa].lson == ridx) {
                    if (lres.empty()) {
                        lres = tr[ridx].bv;
                        lresPrex = tr[lidx].prex;
                    }
                    else {
                        if (fun) {
                            // if (lres.size() < tr[ridx].bv.size()) lres.resize(tr[ridx].bv.size());
                            // else if (lres.size() > tr[ridx].bv.size()) tr[ridx].bv.resize(lres.size());
                            // lres |= tr[ridx].bv;
                            cal(lres, lresPrex, tr[ridx].bv, tr[ridx].prex);
                        }
                        else {
                            lres.resize(tr[ridx].bv.size());
                            lres &= tr[ridx].bv;
                        }
                    }
                    ridx = tr[ridx].fa - 1;
                }
                else {
                    ridx = tr[ridx].fa;
                }
            }

            goto RES;
        }

        // 左边
        // if (L != snap_idx[Ltree])
        {
            auto &tr = tree[Ltree].tr;
            uint idx = L - snap_idx[Ltree];
            uint bias = 0;
            uint half = leaf_size[Ltree];
            uint root = tr.back().id;
            if (idx == 0) {
                lres = tr[root].bv;
                // cout << "Lcal: " << idx << endl;
            } else if (idx == snap_idx[Ltree + 1] - snap_idx[Ltree] - 1) {
                lres = tr[idx].bv;
                // cout << "Lcal: " << idx << endl;
            } else {
                while (1) {
                    bool brk = true;
                    bool brk2 = false;
                    if (tr[idx].fa <= tr.back().id && tr[tr[idx].fa].rson == tr[idx].id && tr[idx].id != tr.back().id && tr[tr[idx].fa + 1].dep == tr[tr[idx].fa].dep) {
                        while (tr[idx].fa <= tr.back().id && tr[tr[idx].fa].rson == tr[idx].id && tr[idx].id != tr.back().id) {
                            if (lres.empty()) {lres = tr[idx].bv; lresPrex = tr[idx].prex;}
                            else {
                                if (fun) {
                                    // if (lres.size() < tr[idx].bv.size()) lres.resize(tr[idx].bv.size());
                                    // else if (lres.size() > tr[idx].bv.size()) tr[idx].bv.resize(lres.size());
                                    // lres |= tr[idx].bv;
                                    cal(lres, lresPrex, tr[idx].bv, tr[idx].prex);
                                }
                                else {
                                    lres.resize(tr[idx].bv.size());
                                    lres &= tr[idx].bv;
                                }
                            }
                            // all_edges = 0;
                            // one = lres.find_first();
                            // while (one != lres.npos) {
                            //     all_edges++;
                            //     one = lres.find_next(one);
                            // }
                            // cout << all_edges << endl;
                            // cout << "Lcal: " << idx << endl;
                            if (tr[tr[idx].fa + 1].dep != tr[tr[idx].fa].dep) {
                                brk2 = true;
                                break;
                            }
                            idx = tr[idx].fa + 1;
                        }
                        brk = false;
                    }
                    if (tr[idx].fa < tr.back().id && tr[tr[idx].fa].lson == tr[idx].id && tr[idx].id != tr.back().id) {
                        while (tr[idx].fa <= tr.back().id && tr[tr[idx].fa].lson == tr[idx].id && tr[idx].id != tr.back().id) {
                            idx = tr[idx].fa;
                        }
                        if (lres.empty()) {lres = tr[idx].bv; lresPrex = tr[idx].prex;}
                        else {
                            if (fun) {
                                // if (lres.size() < tr[idx].bv.size()) lres.resize(tr[idx].bv.size());
                                // else if (lres.size() > tr[idx].bv.size()) tr[idx].bv.resize(lres.size());
                                // lres |= tr[idx].bv;
                                cal(lres, lresPrex, tr[idx].bv, tr[idx].prex);
                            }
                            else {
                                lres.resize(tr[idx].bv.size());
                                lres &= tr[idx].bv;
                            }
                        }
                        brk = false;
                        // cout << "Lcal: " << idx << endl;
                    }
                    if (brk) break;
                    if (brk2) break;
                }
            }
        }
        if (lres.empty())
            goto RES;
        // 中间
        for (uint i = Ltree + 1; i < Rtree; ++i) {
           auto tmp = tree[i].tr.back().bv;

            size_t sz = -1;
            if (fun) {
                // if (lres.size() > tmp.size()) {
                //     sz = tmp.size();
                //     tmp.resize(lres.size());
                // }
                // else if (lres.size() < tmp.size()) lres.resize(tmp.size());
                // lres |= tmp;
                cal(lres, lresPrex, tmp, tree[i].tr.back().prex);
            }
            else {
                lres.resize(tmp.size());
                lres &= tmp;
            }
            if (sz != -1) tmp.resize(sz);
            cout << "cal mid" << endl;
            if (lres.empty())
                goto RES;
        }

        // 右边
        // if (R != snap_idx[Rtree + 1] - 1)
        {
            //            cout << "Right" << endl;
            auto &tr = tree[Rtree].tr;
            uint idx = R - snap_idx[Rtree];
            if (idx == (tr.size() + 1) / 2) {
                if (fun) {
                    // if (lres.size() < tr[idx].bv.size()) lres.resize(tr[idx].bv.size());
                    // else if (lres.size() > tr[idx].bv.size()) tr[idx].bv.resize(lres.size());
                    // lres |= tr[tr.back().id].bv;
                    cal(lres, lresPrex, tr[tr.back().id].bv, tr[tr.back().id].prex);
                }
                else {
                    lres.resize(tr[idx].bv.size());
                    lres &= tr[tr.back().id].bv;
                }
                // cout << "Rcal: " << idx << endl;
            } else if (idx == 0) {
                if (fun) {
                    // if (lres.size() < tr[idx].bv.size()) lres.resize(tr[idx].bv.size());
                    // else if (lres.size() > tr[idx].bv.size()) tr[idx].bv.resize(lres.size());
                    // lres |= tr[idx].bv;
                    cal(lres, lresPrex, tr[idx].bv, tr[idx].prex);
                }
                else {
                    lres.resize(tr[idx].bv.size());
                    lres &= tr[idx].bv;
                }
                // cout << "Rcal: " << idx << endl;
            } else {
                while (1) {
                    bool brk = true;
                    bool brk2 = true;
                    if (tr[idx].fa <= tr.back().id && tr[tr[idx].fa].rson == tr[idx].id && tr[idx].id != tr.back().id) {
                        while (tr[idx].fa <= tr.back().id && tr[tr[idx].fa].rson == tr[idx].id && tr[idx].id != tr.back().id) {
                            idx = tr[idx].fa;
                        }
                        brk = false;
                        if (fun) {
                            // if (lres.size() < tr[idx].bv.size()) lres.resize(tr[idx].bv.size());
                            // else if (lres.size() > tr[idx].bv.size()) tr[idx].bv.resize(lres.size());
                            // lres |= tr[idx].bv;
                            cal(lres, lresPrex, tr[idx].bv, tr[idx].prex);
                        }
                        else {
                            lres.resize(tr[idx].bv.size());
                            lres &= tr[idx].bv;
                        }
                        // cout << "Rcal: " << idx << endl;
                    }
                    if (tr[idx].fa < tr.back().id && tr[tr[idx].fa].lson == tr[idx].id && tr[idx].id != tr.back().id &&
                        tr[tr[idx].fa - 1].dep == tr[tr[idx].fa].dep) {
                        while (tr[idx].fa <= tr.back().id && tr[tr[idx].fa].lson == tr[idx].id && tr[idx].id != tr.back().id) {
                            if (brk) {
                                if (fun) {
                                    // if (lres.size() < tr[idx].bv.size()) lres.resize(tr[idx].bv.size());
                                    // else if (lres.size() > tr[idx].bv.size()) tr[idx].bv.resize(lres.size());
                                    // lres |= tr[idx].bv;
                                    cal(lres, lresPrex, tr[idx].bv, tr[idx].prex);
                                }
                                else {
                                    lres.resize(tr[idx].bv.size());
                                    lres &= tr[idx].bv;
                                }
                                // cout << "Rcal: " << idx << endl;
                            } else
                                brk = true;
                            if (tr[tr[idx].fa - 1].dep != tr[tr[idx].fa].dep) {
                                brk2 = true;
                                break;
                            }
                            idx = tr[idx].fa - 1;
                        }
                        if (lres.size() < tr[idx].bv.size()) lres.resize(tr[idx].bv.size());
                        else if (lres.size() > tr[idx].bv.size()) tr[idx].bv.resize(lres.size());
                        brk = false;
                    }
                    if (brk) break;
                }
            }
        }
    RES:
        all_edges = 0;

        auto queryFinishTime = Duration(query_time);

        auto LoadTime = Get_Time();
        // for (int k = 2; k <= 8; k += 2) {
        all_edges = 0;
        specialsparse* g = (specialsparse *) malloc(sizeof(specialsparse));
        g->e = 0;
        g->n = 0;
        g->edges = (edge *) malloc(sizeof(edge) * lres.cardinality());
        memset(ls, -1, sizeof(int) * 4000000);

        one = lres.find_first();
        while (one != lres.npos) {
            if (ls[id2edge[one].first] == -1)
                ls[id2edge[one].first] = g->n++;
            if (ls[id2edge[one].second] == -1)
                ls[id2edge[one].second] = g->n++;
            g->edges[g->e].s = ls[id2edge[one].first];
            g->edges[g->e].t = ls[id2edge[one].second];
            all_edges++;
            g->e++;
            // assert(one < id2edge.size());
            // assert(id2edge[one].l < CSR.size());
            // g->edges[g->e].s = id2edge[one].l;
            // // assert(id2edge[one].r < CSR[id2edge[one].l].size());
            // g->edges[g->e].t = CSR[id2edge[one].l][id2edge[one].r];
            // g->e++;
            one = lres.find_next(one);
        }
        auto loadTime = Duration(LoadTime);

        vector<vector<int>> N_O(g->n), N_I(g->n);
        for (int egs = 0; egs < g->e; egs++) {
            int u = g->edges[egs].s;
            int v = g->edges[egs].t;
            N_O[u].push_back(v);
            N_I[v].push_back(u);
        }

        int res = 0;
        auto coreTi = Get_Time();
        // res = test(k, g, ls);
        string str = "../dataset/" + dataset + "/q" + to_string(L) + "-" + to_string(R);
        cout << str << endl;

        res = reach(N_O, N_I, g->n, str);
        auto funTime = Duration(coreTi);
        cout << "[ " << L << " " << R << " " << k << " ] ";
        cout << g->n << " " << g->e << endl;
        cout << "Load: " << loadTime << "\t\t";
        cout << "function: " << funTime << "\t\t";
        cout << "|V|: " << res << endl;
        CSV << dataset << "," << k << "," << queryFinishTime << "," << loadTime << "," << funTime << "," << res << endl;

        free(g);
//        std::cout << "  ----------------------------------------------Res: " << all_edges << std::endl;
        //        CSV << "TR" << ',' << L << ',' << R << ',' << all_edges << endl;
        // cout << "peek memoty: " << mem::getValue() / 1024 << "Mb" << std::endl;

        // all_edges = 0;
        if (CNT == 15) {
            CSV << Duration(query_start_time) / 1000 << ',';
            // cout << "TR" << '\t' << mem::getValue() / 1024 << "mb" << '\t' << Duration(query_start_time) / 1000 << "s"
            //         << std::endl;
            // query_start_time = Get_Time();
            // CNT = 0;
        }
    }
    input.close();

    std::cout << "Peak Mem: " << mem::getValue() / 1024 << "mb" << '\t' << "Query time:" << Duration(query_time) / 1000 << "s" << '\t'<< "ALL Time: " << Duration(start) / 1000 << std::endl;
    CSV << Duration(query_start_time) / 1000 << ',' << mem::getValue() / 1024 << ',';

    // if (CHECK) {
    //     size_t sz = 0;
    //     size_t sz2 = 0;
    //     for (auto i: tree) {
    //         for (auto j: i.tr) {
    //             sz += j.bv.size();
    //             sz2 += 4;
    //         }
    //     }
    //     cout << "Tree memory: " << '\t' << (long double) (sz / 8 + sz2 * 4) / 1024 / 1024 << endl;
    //     CSV << (long double) (sz / 8 + sz2 * 4) / 1024 / 1024 << ',';
    //     sz = 0, sz2 = 0;
    //     for (const auto &i: CSR) sz += i.size() * 2 * sizeof(unsigned int);
    //     cout << "CSR memory: " << '\t' << (long double) sz / 1024 / 1024 << endl;
    //     CSV << (long double) sz / 1024 / 1024 << ',';
    //     sz = id2edge.size() * 2 * sizeof(unsigned int);
    //     cout << "2edge memory: " << '\t' << (long double) sz / 1024 / 1024 << endl;
    //     CSV << (long double) sz / 1024 / 1024 << ',';
    //     //        cout << "now memory: " << ',' << get_index_mem()["now"] / 1024 << endl;
    // }
    CSV << intSz*4 << std::endl;
    CSV.close();
    return 0;
}
