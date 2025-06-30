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
#include "../roaring/roaring.h"
#include "../roaring/roaring.hh"
#include "staticcore.hpp"
#include "../utils/pod.h"
//#include "Mem.h"
//#include "deltagraph.h"
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
    };
    // dynamic_bitset<uint32_t> bv;
    roaring::Roaring bv;
    int id;
    vector<int> sons;
    int lson;
    int rson;
    int fa;
    int dep;

    ~TNode()=default;
};

size_t edge_idx;
std::vector<std::vector<std::pair<uint, uint> > > CSR;
std::vector<std::pair<uint, uint> > id2edge;

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

            if (!f) {
                tr[n + cnt].bv = tr[i - 2].bv & tr[i - 1].bv;
            } else {
                tr[n + cnt].bv = tr[i - 2].bv | tr[i - 1].bv;
            }
            tr[n + cnt].bv.runOptimize();
            cnt++;
        }
        if (i - 2 != n) {
            if (tr[n + cnt].lson == -1) {
                tr[n + cnt].lson = n - 1;
                tr[n + cnt].dep = cur_depth;

                tr[n + cnt].bv = tr[n - 1].bv;
                tr[n + cnt].bv.runOptimize();
                tr[n - 1].fa = n + cnt;
            } else {
                tr[n + cnt].rson = n - 1;
                tr[n + cnt].dep = cur_depth;
                tr[n - 1].fa = n + cnt;

                if (!f) {
                    tr[n + cnt].bv = tr[n - 1].bv & tr[tr[n + cnt].lson].bv;
                } else {
                    tr[n + cnt].bv = tr[n - 1].bv | tr[tr[n + cnt].lson].bv;
                }
                tr[n + cnt].bv.runOptimize();
            }
            cnt++;
        }
        n += cnt;
        cur_depth++;
    }
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

void getkcore(roaring::Roaring& lres) {
    specialsparse* g = (specialsparse *) malloc(sizeof(specialsparse));;
    g->e = lres.cardinality();
    g->n = 0;
    g->edges = (edge *)malloc(g->e * sizeof(edge));

    int* ls = new int[1000000];
    memset(ls, -1, sizeof(int) * 1000000);

    all_edges = 0;
    for (auto i : lres) {
        if (ls[id2edge[i - 1].first] == -1) {
            ls[id2edge[i - 1].first] = g->n++;
        }
        if (ls[id2edge[i - 1].second] == -1) {
            ls[id2edge[i - 1].second] = g->n++;
        }
        g->edges[all_edges].s = ls[id2edge[i - 1].first];
        g->edges[all_edges].t = ls[id2edge[i - 1].second];
        // cout << i - 1 << " " << ls[id2edge[i - 1].first] << " " << ls[id2edge[i - 1].second] << endl;
        all_edges++;
    }
    // cout << g->n << " " << g->e << endl;
    // test(6, g);
    // Print_Time("k-core: ", query_start_time);
}

int main(int argc, char *argv[]) {
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

    // if (dataset == "mo4M") K = 18, EDGE_MAX = 3e4 * 4;
    // if (dataset == "mo2") K = 5, EDGE_MAX = 50 * 4;
    // if (dataset == "su") K = 31, EDGE_MAX = 3e4 * 8;
    // if (dataset == "wi") K = 106, EDGE_MAX = 3e4 * 8;
    // if (dataset == "so") K = 93, EDGE_MAX = 3e4 * 256;
    // if (dataset == "test") K = 4, EDGE_MAX = 24;
    // if (dataset == "test2") K = 12, EDGE_MAX = 200;
    // if (dataset == "so2") K = 2773, EDGE_MAX = 3e4 * 256;
    // K = 8;
    // int k;
    app.add_option("-d,--dataset", dataset, "query graph path")->required();
    app.add_option("-m,--method", method, "method")->required();
    app.add_option("-f,--fun", fun, "fun")->required();
    // app.add_option("-K,--K", K, "K");
    // app.add_option("-k,--k", k, "k");
    app.add_option("-e,--edgemax", EDGE_MAX, "edgemax");
    CLI11_PARSE(app, argc, argv);

    getK();

    // EDGE_MAX = 1e5;
    //
    // if (dataset[0] == 's' && dataset[1] == 'o') {
    //     EDGE_MAX = 6e6;
    // }

    query_path = "../dataset/" + dataset + "/q";
    input_file = "../dataset/" + dataset + "/input.txt";
    // input_file = "../dataset/" + dataset + "/inputcore.txt";
    std::chrono::high_resolution_clock::time_point start, lstart;
    cout << query_path << " " << input_file << " " << method << " K=" << K << " " << EDGE_MAX << endl;
    start = Get_Time();
    std::cout << "----------- Loading graphs ------------" << std::endl;

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
                    // if ((*lower).second > b.size()) b.resize((*lower).second);
                        // b.resize(((*lower).second + 31) / 32 * (*lower).second);
                    b.add((*lower).second - 1);
                } else {
                    CSR[v1].insert(lower, {v2, ++csr_edge_count});
                    // if (csr_edge_count > b.size()) b.resize(csr_edge_count);
                        // b.resize((csr_edge_count + 31) / 32 * csr_edge_count);
                    b.add(csr_edge_count - 1);
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

        // b.resize((b.size() + 31) / 32 * 32);

        b.runOptimize();

        if (cur_edges >= EDGE_MAX) {
            // std::cout << "----------- Building tree ------------" << std::endl;
            // cout << "tree size" << " " << tree.back().tr.size() << endl;
            snap_sum += tree.back().tr.size();
            snap_idx.emplace_back(snap_sum);
            build(tree.back(), fun);

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
        snap_idx.emplace_back(snap_sum);
        build(tree.back(), fun);
        cur_edges = 0;
        // std::cout << "Peak Mem: " << mem::getValue() / 1024 << "mb" << std::endl;
    }

    cout << "Build Time: " << Duration(start) / 1000 << endl;
    std::cout << "Memory: " << mem::getValue() / 1024 << "mb" << endl;

    std::cout << "Peak Mem: " << mem::getValue() / 1024 << "mb" << std::endl;
    // fstream CSV2("../result.csv", ios::app);
    // CSV2 << "RR" << ',' << fun << ',' << dataset << ',' << K << ',' << Duration(start) / 1000 << ',' << mem::getValue() / 1024 << endl;
    // CSV2.close();
    // for (auto i : tree) {
    //     for (auto j : i.tr)
    //         cout << j.bv.cardinality() << endl;
    // }


    std::cout << "----------- Snap Query ------------" << std::endl;
    uint L, R;
    fstream input(input_file, ios::in);
    fstream CSV("../resultRR.csv", ios::app);

    int CNT = 0;
    CSV << "RR" << ',' << fun << ',' << dataset << ',' << K << ',' << Duration(start) / 1000 << ',' << mem::getValue() / 1024 << ",";

    size_t intSz = 0;
    for (auto& i : tree) {
        for (auto& j : i.tr) {
            intSz += 5 * 4;
            intSz += j.bv.getSizeInBytes();
        }
    }

    auto query_time = Get_Time();
    auto query_start_time = Get_Time();

    while (input >> L >> R) {
        // cout << "Query: " << L << " " << R << " ";
        // assert(L < K);
        // assert(R < K);
        CNT++;
        uint Ltree, Rtree; // L,R: snap idx Ltree,Rtree: tree idx about snap
        auto lower1 = std::lower_bound(snap_idx.begin(), snap_idx.end(), L);
        if (*lower1 != L) Ltree = lower1 - snap_idx.begin() - 1;
        else Ltree = lower1 - snap_idx.begin();
        auto lower2 = std::lower_bound(snap_idx.begin(), snap_idx.end(), R);
        if (*lower2 != R) Rtree = lower2 - snap_idx.begin() - 1;
        else Rtree = lower2 - snap_idx.begin();
        // cout << L << " " << Ltree << " " << R << " " << Rtree << endl;
        /*dynamic_bitset<uint32_t> lres;*/
        roaring::Roaring lres;
        lres.runOptimize();
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
                        if (lres.cardinality() == 0) {
                            lres = tr[lidx].bv;
                            // cout << lres.cardinality() << endl;
                            if (fun) {
                                lres |= tr[ridx].bv;
                            }
                            else {
                                lres &= tr[ridx].bv;
                            }
                        }
                        else {
                            if (fun) {
                                lres |= tr[lidx].bv;
                                lres |= tr[ridx].bv;
                            }
                            else {
                                lres &= tr[lidx].bv;
                                lres &= tr[ridx].bv;
                            }
                        }
                        // cout << lres.cardinality() << endl;
                        break;
                    }
                }
                if (lidx == ridx) {
                    if (lres.cardinality() == 0) {
                        lres = tr[lidx].bv;
                        // cout << lres.cardinality() << endl;
                    }
                    else {
                        if (fun) {
                            lres |= tr[lidx].bv;
                        }
                        else {
                            lres &= tr[lidx].bv;
                        }
                    }
                    // cout << lres.cardinality() << endl;
                    break;
                }
                if (tr[tr[lidx].fa].rson == lidx) {
                    if (lres.cardinality() == 0) {
                        lres = tr[lidx].bv;
                        // cout << lres.cardinality() << endl;
                    }
                    else {
                        if (fun) {
                            lres |= tr[lidx].bv;
                        }
                        else {
                            lres &= tr[lidx].bv;
                        }
                    }
                    // cout << lres.cardinality() << endl;
                    lidx = tr[lidx].fa + 1;
                }
                else lidx = tr[lidx].fa;
                if (tr[tr[ridx].fa].lson == ridx) {
                    if (lres.cardinality() == 0) {
                        lres = tr[ridx].bv;
                        // cout << lres.cardinality() << endl;
                    }
                    else {
                        if (fun) {
                            lres |= tr[ridx].bv;
                        }
                        else {
                            lres &= tr[ridx].bv;
                        }
                    }
                    // cout << lres.cardinality() << endl;
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
                            if (lres.cardinality() == 0) {lres = tr[idx].bv;}
                            else {
                                if (fun) {
                                    lres |= tr[idx].bv;
                                }
                                else {
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
                        if (lres.cardinality() == 0) {lres = tr[idx].bv;}
                        else {
                            if (fun) {
                                lres |= tr[idx].bv;
                            }
                            else {
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
            if (fun) lres |= tmp;
            else lres &= tmp;
            /*if (fl == 1) tmp.resize(sz);*/
            // cout << "cal mid" << endl;
        }
        // 右边
        // if (R != snap_idx[Rtree + 1] - 1)
        {
            //            cout << "Right" << endl;
            auto &tr = tree[Rtree].tr;
            uint idx = R - snap_idx[Rtree];
            if (idx == (tr.size() + 1) / 2) {
                if (fun) {
                    lres |= tr[tr.back().id].bv;
                }
                else {
                    lres &= tr[tr.back().id].bv;
                }
                // cout << "Rcal: " << idx << endl;
            } else if (idx == 0) {
                if (fun) {
                    lres |= tr[idx].bv;
                }
                else {
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
                            lres |= tr[idx].bv;
                        }
                        else {
                            lres &= tr[idx].bv;
                        }
                        // cout << "Rcal: " << idx << endl;
                    }
                    if (tr[idx].fa < tr.back().id && tr[tr[idx].fa].lson == tr[idx].id && tr[idx].id != tr.back().id &&
                        tr[tr[idx].fa - 1].dep == tr[tr[idx].fa].dep) {
                        while (tr[idx].fa <= tr.back().id && tr[tr[idx].fa].lson == tr[idx].id && tr[idx].id != tr.back().id) {
                            if (brk) {
                                if (fun) {
                                    lres |= tr[idx].bv;
                                }
                                else {
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
                        brk = false;
                    }
                    if (brk) break;
                }
            }
        }
    RES:
        // dynamic_bitset<> rr;

        // auto one = rr.find_first();
        // while (one != rr.npos) {
        //     all_edges++;
        // }

        all_edges = 0;
        for (auto i : lres) {
            all_edges++;
        }

        // std::cout << "  ----------------------------------------------Res: " << all_edges << std::endl;
        //        CSV << "TR" << ',' << L << ',' << R << ',' << all_edges << endl;
        // cout << "peek memoty: " << mem::getValue() / 1024 << "Mb" << std::endl;

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

    CSV << intSz << ",Roaring" << std::endl;
    CSV.close();
    return 0;
}
