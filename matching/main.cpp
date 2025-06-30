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
// #include <rocksdb/db.h>
// #include <rocksdb/options.h>
// #include <rocksdb/slice.h>
#include "staticcore.hpp"
#include "reach.hpp"
// using namespace rocksdb;
using namespace std;
int K = 18;
int level;
bool fun;
size_t EDGE_MAX = 3e4 * 256;
const int N = 100000000;
std::string query_path, dataset, input_file;
bool CHECK = true;
size_t all_edges;

bool USEDB;

// vector<DB*> bv_db;
// DB* tree_db;

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
    vector<int> sons;
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
    int leaf;
};

std::vector<Tree> tree;
std::vector<uint> snap_idx;

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

            if (USEDB) {
                if (fun) {
                    if (tr[i - 2].bv.size() <= tr[i - 1].bv.size()) {
                        tr[n + cnt].bv = tr[i - 2].bv;
                        tr[n + cnt].bv.resize(tr[i - 1].bv.size());
                        tr[n + cnt].bv |= tr[i - 1].bv;
                    }
                    else {
                        tr[n + cnt].bv = tr[i - 1].bv;
                        tr[n + cnt].bv.resize(tr[i - 2].bv.size());
                        tr[n + cnt].bv |= tr[i - 2].bv;
                    }
                }
                else {
                    if (tr[i - 2].bv.size() <= tr[i - 1].bv.size()) {
                        tr[n + cnt].bv = tr[i - 1].bv;
                        tr[n + cnt].bv.resize(tr[i - 2].bv.size());
                        tr[n + cnt].bv &= tr[i - 2].bv;
                    }
                    else {
                        tr[n + cnt].bv = tr[i - 2].bv;
                        tr[n + cnt].bv.resize(tr[i - 1].bv.size());
                        tr[n + cnt].bv &= tr[i - 1].bv;
                    }
                }
                /*if (fun) {
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
                }*/
            }
            cnt++;
        }
        if (i - 2 != n) {
            if (tr[n + cnt].lson == -1) {
                tr[n + cnt].lson = n - 1;
                tr[n + cnt].dep = cur_depth;
                if (USEDB) {
                    tr[n + cnt].bv = tr[n - 1].bv;
                    tr[n + cnt].prex = tr[n - 1].prex;
                }
                tr[n - 1].fa = n + cnt;
            } else {
                tr[n + cnt].rson = n - 1;
                tr[n + cnt].dep = cur_depth;
                tr[n - 1].fa = n + cnt;
                if (USEDB) {
                    if (fun) {
                        if (tr[n - 1].bv.size() >= tr[tr[n + cnt].lson].bv.size()) {
                            tr[n + cnt].bv = tr[tr[n + cnt].lson].bv;
                            tr[n + cnt].bv.resize(tr[n - 1].bv.size());
                            tr[n + cnt].bv |= tr[n - 1].bv;
                        }
                        else {
                            tr[n + cnt].bv = tr[n - 1].bv;
                            tr[n + cnt].bv.resize(tr[tr[n + cnt].lson].bv.size());
                            tr[n + cnt].bv |= tr[tr[n + cnt].lson].bv;
                        }
                    }
                    else {
                        if (fun) {
                            tr[n + cnt].bv = tr[n - 1].bv;
                            tr[n + cnt].bv.resize(tr[tr[n + cnt].lson].bv.size());
                            tr[n + cnt].bv &= tr[tr[n + cnt].lson].bv;
                        }
                        else {
                            tr[n + cnt].bv = tr[tr[n + cnt].lson].bv;
                            tr[n + cnt].bv.resize(tr[n - 1].bv.size());
                            tr[n + cnt].bv &= tr[n - 1].bv;
                        }
                    }
                    /*if (fun) {
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
                    }*/
                }
            }
            // cout << n + cnt << " " << tr[n + cnt].bv.count() << endl;
            cnt++;
        }
        n += cnt;
        cur_depth++;
    }
}

void build(Tree &tre, bool f, int k) {
    auto &tr = tre.tr;
    tre.leaf = tr.size();
    int depth = 0;

    size_t nums = 0;
    size_t nn = tr.size();
    while (nn > k) {
        depth++;
        nums += nn;

        if (nn % k != 0)
            nn = nn / k + 1;
        else
            nn /= k;
    }
    depth += 2;
    nums += nn + 1;
    // cout << "depth:" << depth << "nums:" << nums << endl;
    int cur_depth = 1;
    int n = tr.size();
    int half = depth - 2;
    if (half == -1)
        half = 0;
    // leaf_size.emplace_back(1 << half);
    sum += n;
    tr.resize(nums);
    for (size_t i = 0; i < nums; ++i) tr[i].id = i, tr[i].dep = 0;
    tr[nums - 1].fa = nums - 1;
    int cnt = n;
    int sz = 0;
    while (cur_depth < depth) {
        //// std::cout << "cur_depth_node_num:" << cnt << std::endl;
        int i = n - cnt + k;
        cnt = 0;
        for (; i <= n; i += k) {
            tr[n + cnt].sons.resize(k);
            // cout << "F: " << n + cnt << "- ";
            for (int j = 0; j < k; ++j) {
                tr[n + cnt].sons[j] = i - k + j;
                // cout << i - k + j << " ";
                tr[i - k + j].fa = n + cnt;
            }
            // tr[n + cnt].lson = i - 2;
            // tr[n + cnt].rson = i - 1;
            tr[n + cnt].dep = cur_depth;

            if (!f) {
                tr[n + cnt].bv = tr[i - 1].bv;
                for (int j = i - 2; j >= i - k; j--) {
                    // if (tr[n + cnt].bv.size() < tr[j].bv.size()) {
                        tr[n + cnt].bv.resize(tr[j].bv.size());
                        tr[n + cnt].bv &= tr[j].bv;
                    // }
                    // else {
                    //     sz = tr[j].bv.size();
                    //     tr[j].bv.resize(tr[n + cnt].bv.size());
                    //     tr[n + cnt].bv &= tr[j].bv;
                    //     tr[j].bv.resize(sz);
                    //     tr[j].bv.shrink_to_fit();
                    // }
                }
            }
            else {
                tr[n + cnt].bv = tr[i - k].bv;
                for (int j = i - k + 1; j <= i - 1; j++) {
                    // if (tr[n + cnt].bv.size() < tr[j].bv.size()) {
                        tr[n + cnt].bv.resize(tr[j].bv.size());
                        tr[n + cnt].bv |= tr[j].bv;
                    // }
                    // else {
                    //     sz = tr[j].bv.size();
                    //     tr[j].bv.resize(tr[n + cnt].bv.size());
                    //     tr[n + cnt].bv |= tr[j].bv;
                    //     tr[j].bv.resize(sz);
                    //     tr[j].bv.shrink_to_fit();
                    // }
                }
                // tr[n + cnt].bv = tr[i - 2].bv | tr[i - 1].bv;
            }
            cnt++;
        }
        if (i - k != n) {
            tr[n + cnt].sons.reserve(n - i + k);
            tr[n + cnt].dep = cur_depth;
            // cout << "F: " << n + cnt << "- ";
            for (int j = i - k; j < n; j++) {
                tr[n + cnt].sons.emplace_back(j);
                // cout << j << " ";
                tr[j].fa = n + cnt;
            }

            if (!f) {
                tr[n + cnt].bv = tr[tr[n + cnt].sons.back()].bv;
                for (int j = n - 2; j >= i - k; j--) {
                    // if (tr[n + cnt].bv.size() < tr[j].bv.size()) {
                        tr[n + cnt].bv.resize(tr[j].bv.size());
                        tr[n + cnt].bv &= tr[j].bv;
                    // }
                    // else {
                    //     sz = tr[j].bv.size();
                    //     tr[j].bv.resize(tr[n + cnt].bv.size());
                    //     tr[n + cnt].bv &= tr[j].bv;
                    //     tr[j].bv.resize(sz);
                    //     tr[j].bv.shrink_to_fit();
                    // }
                }
            } else {
                tr[n + cnt].bv = tr[tr[n + cnt].sons[0]].bv;
                for (int j = i - k + 1; j < n; j++) {
                    // if (tr[n + cnt].bv.size() < tr[j].bv.size()) {
                        tr[n + cnt].bv.resize(tr[j].bv.size());
                        tr[n + cnt].bv |= tr[j].bv;
                    // }
                    // else {
                    //     sz = tr[j].bv.size();
                    //     tr[j].bv.resize(tr[n + cnt].bv.size());
                    //     tr[n + cnt].bv |= tr[j].bv;
                    //     tr[j].bv.resize(sz);
                    //     tr[j].bv.shrink_to_fit();
                    // }
                }
                // tr[n + cnt].bv = tr[i - 2].bv | tr[i - 1].bv;
            }
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
    Node res;

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
        intSz += i.son.capacity();
        intSz += 4;
        intSz += 17 * 1.0 / 4 * i.Neighbors.size();
        intSz += 17 * 1.0 / 4 * i.Neighbor_delta.size();
        for (const auto& j : i.Neighbors) {
            intSz += j.second.capacity();
            for (auto k : j.second) {
                intSz += k.csr.capacity();
            }
        }
    }

    auto query_start_time = Get_Time();
    size_t asd = 0;
    int super_root = DG.skeleton.size() - 1;
    while (input >> L >> R) {
        auto begin_q = Get_Time();
        cout << "Query: " << L << " " << R << endl;
        CNT += 1;

        auto prim_time = Get_Time();

        std::cout << "----------- Prim ------------" << std::endl;
        auto p = DG.Prim(L, R);
        // for (const auto &i: p) {
        //     for (auto j: i) {
        //         cout << j << " ";
        //     }
        //     cout << endl;
        // }

        Print_Time("primTime: ", prim_time);

        double time1 = 0;
        double time2 = 0;
        Node res;
        if (level == 0)
            res.Neighbors[-1] = DG.skeleton[DG.skeleton.back().son[0]].Neighbors[-1];
        map<int, bool> st;

        for (int i_ = 0; i_ < p.size(); i_++) {
            auto i = p[i_];
            if (i[0] == super_root) {
                if (i.size() > 2) {
                    Node restmp = move(res);
                    // if (restmp.Neighbors.size() > 0) {
                    //     int ct = 0;
                    //     for (const auto& asd : restmp.Neighbors[-1]) {
                    //         ct += asd.csr.size();
                    //     }
                    //     cout << "776 " << ct << endl;
                    // }
                    if (res.Neighbors.empty() ) {
                        res.Neighbors[-1] = DG.skeleton[i[1]].Neighbors[-1];
                    }

                    // res.Neighbors[-1] = DG.skeleton[i[1]].Neighbors[-1];
                    // cout << " 479->" << i[1] << endl;
                    // int ct = 0;
                    // for (const auto& asd : res.Neighbors[-1])  ct += asd.csr.size();
                    //
                    // cout << "785 " << ct << endl;
                    bool quickL = false, quickR = false;
                    for (int j = 1; j + 1 < i.size(); j++) {
                        // if (fun) {
                            if (i[j] >= K) {
                                if (i_ == p.size() - 1 && DG.skeleton[i[j]].son[0] == i[j + 1] && R >= DG.skeleton[i[j]].son.back()) {
                                    quickL = true;
                                    // cout << "quickL" << endl;
                                    break;
                                }
                                if (i_ == 0 && DG.skeleton[i[j]].son.back() == i[j + 1] && L <= DG.skeleton[i[j]].son[0]) {
                                    quickR = true;
                                    // cout << "quickR" << endl;
                                    break;
                                }

                                if (fun) DG.cha(DG.skeleton[i[j]], res, i[j + 1]);
                                else DG.cup(res, DG.skeleton[i[j]], i[j + 1]);

                                // cout << i[j] << " 484->" << i[j + 1] << endl;

                            }
                            else {
                                if (fun) DG.cup(res, DG.skeleton[i[j]], i[j + 1]);
                                else DG.cha(DG.skeleton[i[j]], res, i[j + 1]);

                                if (DG.skeleton[i[j + 1]].fa != DG.skeleton[i[j]].fa) {
                                    if (fun) DG.cha(DG.skeleton[i[j + 1]], res, i[j]);
                                    else DG.cup(res, DG.skeleton[i[j + 1]], i[j]);
                                }

                                // cout << i[j] << " 487->" << i[j + 1] << endl;
                            }
                        // int ct = 0;
                        // for (auto asd : res.Neighbors[-1]) {
                        //     ct += asd.csr.size();
                        // }
                        // cout << ct << endl;
                        // }
                    }
                    if (i.back() == R && !quickR)   {
                        auto& ve =  DG.skeleton[DG.skeleton[i.back()].fa].son;
                        for (auto j = ve.size() - 1; ~j; j--) {
                            if (ve[j] < R) {
                                if (fun) DG.cup(res, DG.skeleton[ve[j + 1]], ve[j]);
                                else DG.cha(DG.skeleton[ve[j + 1]], res, ve[j]);
                                // cout << ve[j + 1] << " 496->" << ve[j] << endl;
                                // int ct = 0;
                                // for (auto asd : res.Neighbors[-1]) {
                                //     ct += asd.csr.size();
                                // }
                                // cout << ct << endl;
                            }
                        }
                    }
                    if (i.back() == L && !quickL) {
                        auto& ve =  DG.skeleton[DG.skeleton[i.back()].fa].son;
                        for (auto j = 0; j + 1 < ve.size(); j++) {
                            if (L <= ve[j] && ve[j + 1] < R) {
                                if (fun) DG.cup(res, DG.skeleton[ve[j]], ve[j + 1]);
                                else DG.cha(DG.skeleton[ve[j]], res, ve[j + 1]);

                                // cout << ve[j] << " 505->" << ve[j + 1] << endl;
                                // int ct = 0;
                                // for (auto asd : res.Neighbors[-1]) {
                                //     ct += asd.csr.size();
                                // }
                                // cout << ct << endl;
                            }
                        }
                    }
                    if (fun) DG.cup(res, restmp, -1);
                    else {
                        if (!restmp.Neighbors.empty())
                            DG.cap(res, restmp);
                    }
                    // int ct = 0;
                    // for (auto asd : res.Neighbors[-1]) {
                    //     ct += asd.csr.size();
                    // }
                    // cout << ct << endl;
                }
                if (i.size() == 2) {
                    // assert(res.Neighbors.empty());
                    // st[i[1]] = true;
                    if (fun) DG.cup(res, DG.skeleton[i[1]], -1);
                    else DG.cap(res, DG.skeleton[i[1]]);
                    // cout << " 514->" << i[1] << endl;
                    // int ct = 0;
                    // for (auto asd : res.Neighbors[-1]) {
                    //     ct += asd.csr.size();
                    // }
                    // cout << ct << endl;
                }
            }
            else {
                for (int j = 0; j + 1 < i.size(); ++j) {
                    if (DG.skeleton[j].id > K) continue;
                    // st[i[j + 1]] = true;
                    if (fun)
                        DG.cup(res, DG.skeleton[i[j]], i[j + 1]);
                    else {
                        DG.cha(DG.skeleton[i[j]], res, i[j + 1]);
                    }
                    // int ct = 0;
                    // for (auto asd : res.Neighbors[-1]) {
                    //     ct += asd.csr.size();
                    // }
                    // cout << ct << endl;
                    // cout << i[j] << " 523->" << i[j + 1] << endl;
                }
            }
        }

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
inline void calCap(dynamic_bitset<uint32_t>& lres, dynamic_bitset<uint32_t>& bt) {
    if (lres.size() > bt.size()) {
        int sz = bt.size();
        bt.resize(lres.size());
        lres &= bt;
        bt.resize(sz);
        bt.shrink_to_fit();
    }
    else {
        lres.resize(bt.size());
        lres &= bt;
    }
}
inline void calCup(dynamic_bitset<uint32_t>& lres, dynamic_bitset<uint32_t>& bt) {
    if (lres.size() > bt.size()) {
        int sz = bt.size();
        bt.resize(lres.size());
        lres |= bt;
        bt.resize(sz);
        bt.shrink_to_fit();
    }
    else {
        lres.resize(bt.size());
        lres |= bt;
    }
}

int GET_CNT_Read = 0;
double GET_CNT_TIME_Read = 0;
int GET_CNT_Write = 0;
double GET_CNT_TIME_Write = 0;

// inline void GetValFromDB(int DBidx, int key, TNode& tr) {
//     Status s;
//     Options op;
//     // cout << "../../DB/" + dataset + "/bv" + to_string(key) + "_db" << " " << key << endl;
//     // TNode as;
//
//     s = DB::Open(op, "../../DB/" + dataset + "/bv" + to_string(DBidx) + "_db", &bv_db[DBidx]);
//     if (!s.ok()) { cout << "bv_db open failed" << endl; cout << "../../DB/" + dataset + "/bv" + to_string(DBidx) + "_db" << endl; exit(1); }
//     uint l = 0;
//     string value;
//     auto getTi = Get_Time();
//     bv_db[DBidx]->Get(ReadOptions(), to_string(key), &value);
//     GET_CNT_TIME_Read += Duration(getTi);
//     GET_CNT_Read++;
//
// //    cout << value << endl;
//     int idx = 0;
//     // tr.prex = -1;
//     for (int sidx = 0; sidx < value.size(); ++sidx) {
//         if (value[sidx] == ',') {
//             // if (tr.prex == -1) tr.prex = l;
//             // else {
//                 // cout << l << " " << tr.
//                 tr.bv.append(l);
//                 idx++;
//             // }
//             // cout << l << endl;
//             l = 0;
//             continue;
//         }
//         l = l * 10 + value[sidx] - '0';
//     }
//     tr.bv.append(l);
//     tr.bv.shrink_to_fit();
//     bv_db[DBidx]->Close();
// }

int main(int argc, char *argv[]) {
    /*dynamic_bitset<uint32_t> asd;
    dynamic_bitset<uint32_t> asd2;
    // asd.resize(320);
    // asd2.resize(320 - 32 - 32);
    //
    // asd.set(320 - 32 + 1);
    // asd.set(320 - 32 - 32);
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
    dataset = "su4M";
    level = 0;
    fun = 1;
    int k = 2;
    EDGE_MAX = 64;
    app.add_option("-d,--dataset", dataset, "query graph path")->required();
    app.add_option("-m,--method", method, "method")->required();
    app.add_option("-f,--fun", fun, "method")->required();
    app.add_option("-k,--K", k, "K");
    // app.add_option("-l,--mat", level, "level");
    app.add_option("-e,--edgemax", EDGE_MAX, "edgemax");
    CLI11_PARSE(app, argc, argv);

    getK();

    // EDGE_MAX = 1e5;
    // if (dataset[0] == 's' && dataset[1] == 'o') {
    //     EDGE_MAX = 6e6;
    // }
    // bv_db.resize(10);
    query_path = "../dataset/" + dataset + "/q";
    input_file = "../dataset/" + dataset + "/input.txt";

    // input_file = "../dataset/" + dataset + "/inputcore.txt";
    // input_file = "../dataset/" + dataset + "/inputreach.txt";

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

    // Status s;
    // Options op;
    // op.create_if_missing = true;
    // op.IncreaseParallelism();
    // op.OptimizeLevelStyleCompaction();
    // s = DB::Open(op, "../../DB/" + dataset + "/tree_db", &tree_db);
    // if (!s.ok()) { cout << "tree_db " << " open failed" << endl; return 0; }
    //
    // for (int i = 0; i < 3; ++i) {
    //     s = DB::Open(op, "/home/qsl/exp/DB/" + dataset + "/bv" + to_string(i) + "_db", &bv_db[i]);
    //     cout << s.ok() << endl;
    //     if (!s.ok()) { cout << "bv_db " << i << " open failed" << endl; return 0; }
    // }
    // auto it = tree_db->NewIterator(ReadOptions());
    // for (it->SeekToFirst(); it->Valid(); it->Next()) {
    //     int l = 0, r = 0;
    //     for (int j = 0; j < it->value().ToString().size(); j++) {
    //         if (it->value().ToString()[j] == ',') {
    //             l = r;
    //             r = 0;
    //             continue;
    //         }
    //         r = r * 10 + (it->value().ToString()[j] - '0');
    //     }
    //
    //     snap_idx.resize(snap_idx.size() + 1);
    //     snap_idx.back() = l;
    //     tree.resize(tree.size() + 1);
    //     tree.back().tr.resize(r);
    // }
    // tree_db->Close();

    int cnt = 1;
    size_t cur_edges = 0;
    size_t csr_edge_count = 0;
    size_t snap_sum = 0;
    size_t all_edges = 0;
    snap_idx.emplace_back(0);
    tree.resize(1);
    USEDB = true;
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
                // cur_edges++;
                edge_count_++;
                all_edges++;
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
                    id2edge.back() = {v1, v2};
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
        cur_edges++;
        b.resize((b.size() + 31) / 32 * 32);
        // cout << edge_count_ << " " << b.count() << " " << b.num_blocks() << "->";

        // reverse(b.m_bits.begin(), b.m_bits.end());
        // for (int j = b.num_blocks() - 1; ~j; j--) {
        //     if (b.m_bits[j] != 0) {
        //         tree.back().tr.back().prex = b.num_blocks() - j - 1;
        //         b.resize((j + 1) * sizeof(uint32_t) * 8);
        //         b.shrink_to_fit();
        //         break;
        //     }
        // }
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

        if (cur_edges >= EDGE_MAX) {
            // std::cout << "----------- Building tree ------------" << std::endl;
            // cout << "tree size" << " " << tree.back().tr.size() << endl;
            snap_sum += tree.back().tr.size();
            snap_idx.resize(snap_idx.size() + 1);
            snap_idx.back() = (snap_sum);
            build(tree.back(), fun,k);
            cnt = 0;
            cur_edges = 0;
            if (snap_i != K) tree.resize(tree.size() + 1);
            // std::cout << "Peak Mem: " << mem::getValue() / 1024 << "mb" << std::endl;
        }
    }
    if (cur_edges) {
        // std::cout << "----------- Building tree ------------" << std::endl;
        // cout << "tree size" << " " << tree.back().tr.size() << endl;
        snap_sum += tree.back().tr.size();
        snap_idx.resize(snap_idx.size() + 1);
        snap_idx.back() = (snap_sum);
        build(tree.back(), fun,k);
        cur_edges = 0;
        // std::cout << "Peak Mem: " << mem::getValue() / 1024 << "mb" << std::endl;
    }
    // size_t mx = 0;
    // size_t mn = 2e9;
    // size_t avg = 0;
    // size_t s = 0;
    // for (const auto& i : tree) {
    //     for (const auto& j : i.tr) {
    //         mx = max(mx, j.bv.size());
    //         mn = min(mn, j.bv.size());
    //         avg += j.bv.size();
    //         s++;
    //     }
    // }

    // if (USEDB) {
    //     bv_db.resize(tree.size());
    //     op.create_if_missing = true;
    //
    //     s = DB::Open(op, "../../DB/" + dataset + "/tree_db", &tree_db);
    //     if (!s.ok()) { cout << "tree_db " << dataset << " open failed" << endl; return 0; }
    //     for (int i = 0; i < tree.size(); ++i) {
    //         s = DB::Open(op, "../../DB/" + dataset + "/bv" + to_string(i) + "_db", &bv_db[i]);
    //         cout << "../../DB/" + dataset + "/bv" + to_string(i) + "_db" << endl;
    //         if (!s.ok()) { cout << "bv_db " << i << " open failed" << endl; return 0; }
    //
    //         for (int j = 0; j < tree[i].tr.size(); ++j) {
    //             string val = to_string(tree[i].tr[j].bv.m_bits[0]);
    //             for (int k = 1; k < tree[i].tr[j].bv.m_bits.size(); ++k) {
    //                 val += "," + to_string(tree[i].tr[j].bv.m_bits[k]);
    //             }
    //             GET_CNT_Write++;
    //             auto ti = Get_Time();
    //             s = bv_db[i]->Put(WriteOptions(), to_string(j), val);
    //             if (!s.ok()) {cout << 1178 << endl; exit(1); }
    //             GET_CNT_TIME_Write += Duration(ti);
    //         }
    //         bv_db[i]->Close();
    //         GET_CNT_Write++;
    //         auto ti = Get_Time();
    //         tree_db->Put(WriteOptions(), to_string(i), to_string(snap_idx[i + 1]) + "," + to_string(tree[i].tr.size()));
    //         GET_CNT_TIME_Write += Duration(ti);
    //     }
    //     tree_db->Close();
    //     cout << "Store in DB" << endl;
    //     return 0;
    // }
    // cout << bv_db.size() << endl;
    // if (!USEDB)

    // for (int i = 0; i < tree.size(); ++i) {
    //     s = DB::Open(op, "../../DB/" + dataset + "/bv" + to_string(i) + "_db", &bv_db[i]);
    //     if (!s.ok()) { cout << "bv_db open failed" << endl; cout << "../../DB/" + dataset + "/bv" + to_string(i) + "_db" << endl; exit(1); }
    //     for (int j = 0; j < tree[i].tr.size(); ++j) {
    //         // cout << tree[i].tr[j].id << " " << tree[i].tr[j].bv.m_bits.size() << " " << tree[i].tr[j].bv.count() << endl;
    //         tree[i].tr[j].bv.clear();
    //         uint l = 0;
    //         string value;
    //         auto getTi = Get_Time();
    //         bv_db[i]->Get(ReadOptions(), to_string(j), &value);
    //         GET_CNT_TIME_Read += Duration(getTi);
    //         GET_CNT_Read++;
    //
    //         //    cout << value << endl;
    //         int idx = 0;
    //         // tr.prex = -1;
    //         for (int sidx = 0; sidx < value.size(); ++sidx) {
    //             if (value[sidx] == ',') {
    //                 // if (tr.prex == -1) tr.prex = l;
    //                 // else {
    //                 // cout << l << " " << tr.
    //                 tree[i].tr[j].bv.append(l);
    //                 idx++;
    //                 // }
    //                 // cout << l << endl;
    //                 l = 0;
    //                 continue;
    //             }
    //             l = l * 10 + value[sidx] - '0';
    //         }
    //         tree[i].tr[j].bv.append(l);
    //         tree[i].tr[j].bv.shrink_to_fit();
    //     }
    //     bv_db[i]->Close();
    // }
    // cout << "Load from DB" << endl;

    std::cout << "Peak Mem: " << mem::getValue() / 1024 << "mb" << std::endl;
    // fstream CSV2("../result.csv", ios::app);
    // CSV2 << "TR" << ',' << fun << ',' << dataset << ',' << K << ',' << Duration(start) / 1000 << ',' << mem::getValue() / 1024 << endl;
    // CSV2.close();

    cout << "Build Time: " << Duration(start) / 1000 << endl;
    std::cout << "Memory: " << mem::getValue() / 1024 << "mb" << endl;
    size_t intSz = 0;
    // for (auto& i : tree) {
    //     for (auto& j : i.tr) {
    //         intSz += 5;
    //         intSz += j.bv.num_blocks();
    //     }
    // }
    //
    // cout << intSz * 4 / 1024 / 1024 << endl;


    std::cout << "----------- Snap Query ------------" << std::endl;
    uint L, R;

    cout << input_file << endl;
    fstream input(input_file, ios::in);
    fstream CSV("../result.csv", ios::app);

    int CNT = 0;
    CSV << "TR" << ',' << fun << ',' << dataset << ',' << K << ',' << Duration(start) / 1000 << ',' << mem::getValue() / 1024 << ",";

    // size_t intSz = 0;
    for (auto& i : tree) {
        for (auto& j : i.tr) {
            intSz += 5;
            intSz += j.bv.num_blocks();
        }
    }

    auto query_time = Get_Time();
    auto query_start_time = Get_Time();
    size_t one;
    // int* ls = new int[4000000];
    // int k;
    while (input >> L >> R) {
        cout << "Query: " << L << " " << R << endl;
        assert(L < K);
        assert(R < K);
        // query_time = Get_Time();
        CNT++;
        uint Ltree, Rtree; // L,R: snap idx Ltree,Rtree: tree idx about snap
        auto lower1 = std::lower_bound(snap_idx.begin(), snap_idx.end(), L);
        if (*lower1 != L) Ltree = lower1 - snap_idx.begin() - 1;
        else Ltree = lower1 - snap_idx.begin();
        auto lower2 = std::lower_bound(snap_idx.begin(), snap_idx.end(), R);
        if (*lower2 != R) Rtree = lower2 - snap_idx.begin() - 1;
        else Rtree = lower2 - snap_idx.begin();
        // cout << L << " " << Ltree << " " << R << " " << Rtree << " ";
        dynamic_bitset<uint32_t> lres;
        int lresPrex = 0;
        if (Ltree == Rtree) { // 一起往上跳
            auto &tr = tree[Ltree].tr;
            int lidx = L - snap_idx[Ltree], ridx = R - snap_idx[Rtree];
            // cout << lidx << " " << ridx << endl;
            if (ridx - lidx + 1 == snap_idx[Rtree + 1] - snap_idx[Ltree]) {
                lres = tr.back().bv;
                // cout << "quick " << tr.back().id << endl;
                goto RES;
            }
            if (tr[lidx].fa == tr[ridx].fa) {
                auto lower = lower_bound(tr[tr[lidx].fa].sons.begin(), tr[tr[lidx].fa].sons.end(), lidx);
                if (lres.empty())
                    lres = tr[*lower].bv;

                lower++;
                if (fun) {
                    for (; lower != tr[tr[lidx].fa].sons.end(); lower++) {
                        calCup(lres, tr[*lower].bv);
                        if (*lower == ridx)
                            break;
                    }

                }
                else {
                    for (; lower != tr[tr[lidx].fa].sons.end(); lower++) {
                        calCap(lres, tr[*lower].bv);
                        if (*lower == ridx)
                            break;
                    }
                }
                goto RES;
            }
            while (true) {
                if (lidx + 1 == ridx) {
                    if (tr[lidx].fa != tr[ridx].fa) {
                        if (lres.empty()) {
                            lres = tr[lidx].bv;
                            if (fun) {
                                calCup(lres,  tr[ridx].bv);
                            }
                            else {
                                calCap(lres,  tr[ridx].bv);
                            }
                        }
                        else {
                            if (fun) {
                                calCup(lres,  tr[lidx].bv);
                                calCup(lres,  tr[ridx].bv);
                            }
                            else {
                                calCap(lres,  tr[lidx].bv);
                                calCap(lres,  tr[ridx].bv);
                            }
                        }
                    }
                    else {
                        if (fun) {
                            calCup(lres,  tr[lidx].bv);
                            calCup(lres,  tr[ridx].bv);
                        }
                        else {
                            calCap(lres,  tr[lidx].bv);
                            calCap(lres,  tr[ridx].bv);
                        }
                    }
                    break;
                }

                if (tr[lidx].fa + 1 == tr[ridx].fa) {
                    if (tr[tr[lidx].fa].sons[0] == lidx) {
                        if (lres.empty())
                            lres = tr[tr[lidx].fa].bv;
                        else {
                            if (fun)
                                calCup(lres,  tr[tr[lidx].fa].bv);
                            else
                                calCap(lres,  tr[tr[lidx].fa].bv);
                        }
                    }
                    else {
                        auto lower = lower_bound(tr[tr[lidx].fa].sons.begin(), tr[tr[lidx].fa].sons.end(), lidx);
                        if (lres.empty())
                            lres = tr[*lower].bv;
                        else {
                            if (fun)
                                calCup(lres, tr[*lower].bv);
                            else
                                calCap(lres, tr[*lower].bv);
                        }
                        lower++;
                        if (fun) {
                            for (; lower != tr[tr[lidx].fa].sons.end(); lower++)
                                calCup(lres, tr[*lower].bv);
                        }
                        else {
                            for (; lower != tr[tr[lidx].fa].sons.end(); lower++)
                                calCap(lres, tr[*lower].bv);
                        }
                    }

                    if (tr[tr[ridx].fa].sons.back() == ridx) {
                        if (fun) {
                            calCup(lres, tr[tr[ridx].fa].bv);
                        }
                        else {
                            calCap(lres, tr[tr[ridx].fa].bv);
                        }
                        break;
                    }
                    auto lower = lower_bound(tr[tr[ridx].fa].sons.begin(), tr[tr[ridx].fa].sons.end(), ridx);
                    if (fun) {
                        for (; lower >= tr[tr[ridx].fa].sons.begin(); lower--)
                            calCup(lres, tr[*lower].bv);
                    }
                    else {
                        for (; lower >= tr[tr[ridx].fa].sons.begin(); lower--)
                            calCap(lres, tr[*lower].bv);
                    }
                    break;
                }
                if (lidx == ridx) {
                    if (lres.empty()) {
                        lres = tr[lidx].bv;
                    }
                    else {
                        if (fun) {
                            calCup(lres,  tr[lidx].bv);
                        }
                        else {
                            calCap(lres,  tr[lidx].bv);
                        }
                    }
                    break;
                }
                if (tr[lidx].fa == tr[ridx].fa) {
                    if (fun) {
                        for (; lidx <= ridx; lidx++) {
                            calCup(lres,  tr[lidx].bv);
                        }
                    }
                    else {
                        for (; lidx <= ridx; lidx++) {
                            calCap(lres,  tr[lidx].bv);
                        }
                    }
                    goto RES;
                }
                if (tr[tr[lidx].fa].sons[0] != lidx) {
                    if (lres.empty()) {
                        lres = tr[lidx].bv;
                        auto lower = upper_bound(tr[tr[lidx].fa].sons.begin(), tr[tr[lidx].fa].sons.end(), lidx);
                        if (fun) {
                            for (; lower < tr[tr[lidx].fa].sons.end(); lower++) {
                                calCup(lres, tr[*lower].bv);
                            }
                        }
                        else {
                            for (; lower < tr[tr[lidx].fa].sons.end(); lower++) {
                                calCap(lres, tr[*lower].bv);
                            }
                        }
                    }
                    else {
                        auto lower = lower_bound(tr[tr[lidx].fa].sons.begin(), tr[tr[lidx].fa].sons.end(), lidx);

                        if (fun) {
                            for (; lower < tr[tr[lidx].fa].sons.end(); lower++) {
                                calCup(lres, tr[*lower].bv);
                            }
                        }
                        else {
                            for (; lower < tr[tr[lidx].fa].sons.end(); lower++) {
                                calCap(lres, tr[*lower].bv);
                            }
                        }
                    }

                    lidx = tr[lidx].fa + 1;
                }
                else lidx = tr[lidx].fa;
                if (tr[tr[ridx].fa].sons.back() != ridx) {
                    if (lres.empty()) {
                        lres = tr[ridx].bv;
                        auto lower = lower_bound(tr[tr[ridx].fa].sons.begin(), tr[tr[ridx].fa].sons.end(), ridx);
                        lower--;
                        if (fun) {
                            for (; lower >= tr[tr[ridx].fa].sons.begin(); lower--) {
                                calCup(lres, tr[*lower].bv);
                            }
                        }
                        else {
                            for (; lower >= tr[tr[ridx].fa].sons.begin(); lower--) {
                                calCap(lres, tr[*lower].bv);
                            }
                        }
                    }
                    else {
                        auto lower = lower_bound(tr[tr[ridx].fa].sons.begin(), tr[tr[ridx].fa].sons.end(), ridx);

                        if (fun) {
                            for (; lower >= tr[tr[ridx].fa].sons.begin(); lower--) {
                                calCup(lres, tr[*lower].bv);
                            }
                        }
                        else {
                            for (; lower >= tr[tr[ridx].fa].sons.begin(); lower--) {
                                calCap(lres, tr[*lower].bv);
                            }
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
        // 左边
        // if (L != snap_idx[Ltree])
        {
            auto &tr = tree[Ltree].tr;
            uint idx = L - snap_idx[Ltree];
            uint root = tr.back().id;
            if (idx == 0) {
                lres = tr[root].bv;
                // cout << "Lcal: " << idx << endl;
            }
            else if (tr[idx].fa == tr[snap_idx[Ltree + 1] - snap_idx[Ltree] - 1].fa) {
                if (tr[tr[idx].fa].sons[0] == idx) {
                    if (fun)
                        lres = tr[tr[idx].fa].bv;
                    else
                        lres = tr[tr[idx].fa].bv;
                }
                else {
                    auto lower = upper_bound(tr[tr[idx].fa].sons.begin(), tr[tr[idx].fa].sons.end(), idx);
                    lres = tr[idx].bv;
                    for (; lower != tr[tr[idx].fa].sons.end(); lower++) {
                        if (fun)
                            calCup(lres, tr[*lower].bv);
                        else
                            calCap(lres, tr[*lower].bv);
                    }
                }
            }
            else {
                while (1) {
                    bool brk = true;
                    bool brk2 = false;
                    if (tr[idx].fa <= tr.back().id && tr[tr[idx].fa].sons[0] != tr[idx].id && tr[idx].id != tr.back().id && tr[tr[idx].fa + 1].dep == tr[tr[idx].fa].dep) {
                        while (tr[idx].fa <= tr.back().id && tr[tr[idx].fa].sons[0] != tr[idx].id && tr[idx].id != tr.back().id) {
                            if (lres.empty()) {
                                lres = tr[idx].bv;
                                auto lower = lower_bound(tr[tr[idx].fa].sons.begin(), tr[tr[idx].fa].sons.end(), idx);
                                lower++;
                                if (fun) {
                                    for (; lower < tr[tr[idx].fa].sons.end(); lower++)
                                        calCup(lres, tr[*lower].bv);
                                }
                                else {
                                    for (; lower < tr[tr[idx].fa].sons.end(); lower++)
                                        calCap(lres, tr[*lower].bv);
                                }
                            }
                            else {
                                auto lower = lower_bound(tr[tr[idx].fa].sons.begin(), tr[tr[idx].fa].sons.end(), idx);
                                if (fun) {
                                    for (; lower < tr[tr[idx].fa].sons.end(); lower++)
                                        calCup(lres, tr[*lower].bv);
                                }
                                else {
                                    for (; lower < tr[tr[idx].fa].sons.end(); lower++)
                                        calCap(lres, tr[*lower].bv);
                                }
                            }

                            if (tr[tr[idx].fa + 1].dep != tr[tr[idx].fa].dep) {
                                brk2 = true;
                                break;
                            }
                            idx = tr[idx].fa + 1;
                        }
                        brk = false;
                    }
                    if (tr[idx].fa < tr.back().id && tr[tr[idx].fa].sons[0] == tr[idx].id && tr[idx].id != tr.back().id) {
                        while (tr[idx].fa <= tr.back().id && tr[tr[idx].fa].sons[0] == tr[idx].id && tr[idx].id != tr.back().id) {
                            idx = tr[idx].fa;
                        }
                        if (lres.empty()) {
                            lres = tr[idx].bv;
                            auto lower = upper_bound(tr[tr[idx].fa].sons.begin(), tr[tr[idx].fa].sons.end(), idx);

                            if (fun) {
                                for (; lower < tr[tr[idx].fa].sons.end(); lower++)
                                    calCup(lres, tr[*lower].bv);
                            }
                            else {
                                for (; lower < tr[tr[idx].fa].sons.end(); lower++)
                                    calCap(lres, tr[*lower].bv);
                            }
                        }
                        else {
                            auto lower = lower_bound(tr[tr[idx].fa].sons.begin(), tr[tr[idx].fa].sons.end(), idx);
                            if (fun) {
                                for (; lower < tr[tr[idx].fa].sons.end(); lower++)
                                    calCup(lres, tr[*lower].bv);
                            }
                            else {
                                for (; lower < tr[tr[idx].fa].sons.end(); lower++)
                                    calCap(lres, tr[*lower].bv);
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
        // 中间
        for (uint i = Ltree + 1; i < Rtree; ++i) {
           auto tmp = tree[i].tr.back().bv;

            /*int fl = 0;
            size_t sz = 0;
            if (lres.size() > tmp.size()) {
                fl = 1;
                sz = tmp.size();
                tmp.resize(lres.size());
            } else if (lres.size() < tmp.size()) lres.resize(tmp.size());*/
            if (fun) calCup(lres,  tmp);
            else calCap(lres,  tmp);
            /*if (fl == 1) tmp.resize(sz);*/
            // cout << "cal mid" << endl;
        }
        // 右边
        // if (R != snap_idx[Rtree + 1] - 1)
        {
            //            cout << "Right" << endl;
            auto &tr = tree[Rtree].tr;
            uint idx = R - snap_idx[Rtree];
            if (idx == snap_idx[Rtree + 1] - snap_idx[Rtree] - 1) {
                if (fun) {
                    calCup(lres,  tr[tr.back().id].bv);
                }
                else {
                    calCap(lres,  tr[tr.back().id].bv);
                }
                // cout << "Rcal: " << idx << endl;
            } else if (idx == 0) {
                if (fun) {
                    calCup(lres,  tr[idx].bv);
                }
                else {
                    calCap(lres,  tr[idx].bv);
                }
                // cout << "Rcal: " << idx << endl;
            }
            else if (tr[idx].fa == snap_idx[Rtree + 1] - snap_idx[Rtree]) {
                if (tr[tr[idx].fa].sons.back() == idx) {
                    if (fun)
                        calCup(lres,  tr[tr[idx].fa].bv);
                    else
                        calCap(lres,  tr[tr[idx].fa].bv);
                }
                else {
                    auto lower = lower_bound(tr[tr[idx].fa].sons.begin(), tr[tr[idx].fa].sons.end(), idx);
                    for (; lower >= tr[tr[idx].fa].sons.begin(); lower--) {
                        if (fun)
                            calCup(lres, tr[*lower].bv);
                        else
                            calCap(lres, tr[*lower].bv);
                    }
                }
            }
            else {
                while (1) {
                    bool brk = true;
                    bool brk2 = true;
                    // 是最右点
                    if (tr[idx].fa <= tr.back().id && tr[tr[idx].fa].sons.back() == tr[idx].id && tr[idx].id != tr.back().id) {
                        while (tr[idx].fa <= tr.back().id && tr[tr[idx].fa].sons.back() == tr[idx].id && tr[idx].id != tr.back().id) {
                            idx = tr[idx].fa;
                        }

                        brk = false;
                        auto lower = lower_bound(tr[tr[idx].fa].sons.begin(), tr[tr[idx].fa].sons.end(), idx);

                        if (fun) {
                            for (; lower >= tr[tr[idx].fa].sons.begin(); lower--)
                                calCup(lres, tr[*lower].bv);
                        }
                        else {
                            for (; lower >= tr[tr[idx].fa].sons.begin(); lower--)
                                calCap(lres, tr[*lower].bv);
                        }
                        // cout << "Rcal: " << idx << endl;
                    }
                    if (tr[idx].fa < tr.back().id && tr[tr[idx].fa].sons.back() != tr[idx].id && tr[idx].id != tr.back().id &&
                        tr[tr[idx].fa - 1].dep == tr[tr[idx].fa].dep) {
                        while (tr[idx].fa <= tr.back().id && tr[tr[idx].fa].sons.back() != tr[idx].id && tr[idx].id != tr.back().id) {
                            if (brk) {
                                auto lower = lower_bound(tr[tr[idx].fa].sons.begin(), tr[tr[idx].fa].sons.end(), idx);

                                if (fun) {
                                    for (; lower >= tr[tr[idx].fa].sons.begin(); lower--)
                                        calCup(lres, tr[*lower].bv);
                                }
                                else {
                                    for (; lower >= tr[tr[idx].fa].sons.begin(); lower--)
                                        calCap(lres, tr[*lower].bv);
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
                        brk = false;
                    }
                    if (brk) break;
                }
            }
        }
    RES:
        all_edges = 0;

        // auto queryFinishTime = Duration(query_time);

        // auto LoadTime = Get_Time();
        // for (int k = 2; k <= 8; k += 2) {
        all_edges = 0;
        // specialsparse* g = (specialsparse *) malloc(sizeof(specialsparse));
        // g->e = 0;
        // g->n = 0;
        // g->edges = (edge *) malloc(sizeof(edge) * lres.size());
        // memset(ls, -1, sizeof(int) * 4000000);

        one = lres.find_first();
        while (one != lres.npos) {
            // if (ls[id2edge[one].first] == -1)
            //     ls[id2edge[one].first] = g->n++;
            // if (ls[id2edge[one].second] == -1)
            //     ls[id2edge[one].second] = g->n++;
            // g->edges[g->e].s = ls[id2edge[one].first];
            // g->edges[g->e].t = ls[id2edge[one].second];
            // g->e++;

            all_edges++;

            // assert(one < id2edge.size());
            // assert(id2edge[one].l < CSR.size());
            // g->edges[g->e].s = id2edge[one].l;
            // // assert(id2edge[one].r < CSR[id2edge[one].l].size());
            // g->edges[g->e].t = CSR[id2edge[one].l][id2edge[one].r];
            // g->e++;
            one = lres.find_next(one);
        }
        // auto loadTime = Duration(LoadTime);

        // vector<vector<int>> N_O(g->n), N_I(g->n);
        // for (int egs = 0; egs < g->e; egs++) {
        //     int u = g->edges[egs].s;
        //     int v = g->edges[egs].t;
        //     N_O[u].push_back(v);
        //     N_I[v].push_back(u);
        // }

        // int res = 0;
        // auto coreTi = Get_Time();
        // res = test(k, g, ls);
        // string str = "../dataset/" + dataset + "/q" + to_string(L) + "-" + to_string(R);
        // cout << str << endl;

        // res = reach(N_O, N_I, g->n, str);
        // auto funTime = Duration(coreTi);
        // cout << "[ " << L << " " << R << " " << k << " ] ";
        // cout << g->n << " " << g->e << endl;
        // cout << "Load: " << loadTime << "\t\t";
        // cout << "function: " << funTime << "\t\t";
        // cout << "|V|: " << res << endl;
        // CSV << dataset << "," << k << "," << queryFinishTime << "," << loadTime << "," << funTime << "," << res << endl;

        // free(g);
        std::cout << "  ----------------------------------------------Res: " << all_edges << std::endl;


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
    Print_Time("Query Time", query_start_time);
    // CSV << Duration(query_start_time) << endl;

    // std::cout << "Peak Mem: " << mem::getValue() / 1024 << "mb" << '\t' << "Query time:" << Duration(query_start_time) / 1000 << "s" << '\t'<< "ALL Time: " << Duration(start) / 1000 << std::endl;
    CSV << Duration(query_start_time) / 1000 << ',' << mem::getValue() / 1024 << ',';

    CSV << intSz*4 <<
            "," <<
        GET_CNT_Read << "," << GET_CNT_TIME_Read << "," << GET_CNT_Read / GET_CNT_TIME_Read * 1000 << "," <<
        GET_CNT_Write << "," << GET_CNT_TIME_Write << "," << GET_CNT_Write / GET_CNT_TIME_Write * 1000<< std::endl;
    CSV.close();
    return 0;
}
