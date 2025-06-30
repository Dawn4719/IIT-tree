#ifndef _UTILITY_RESOURCE_0EFC332C_91AC_4126_9BAB_F32C12AAD376_H__
#define _UTILITY_RESOURCE_0EFC332C_91AC_4126_9BAB_F32C12AAD376_H__

#include <map>
#include <tuple>

class UtilityResource
{
public:
    static int cpu_count();
    // 当前cpu，区分子线程
    static void current_cpu(std::map<int, double>& cpus);

    static uint64_t total_mem();
    // 当前内存，RES
    static uint64_t current_mem();

private:
    static void _enmu_pid_sysfile();
    static std::string _read_stat_file(const char* file_name);
    static int _get_process_time_all();
    static int _get_process_time_tid(int tid);
    static uint64_t _get_res_mem_all();
    static uint64_t _get_res_mem_pid();

private:
    static int g_cpu_count;
    static int g_pid;
    static std::map<int, std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>> g_tid_cpus; // <tid: <last, curr, last_all, curr_all>>
};

#endif // _UTILITY_RESOURCE_0EFC332C_91AC_4126_9BAB_F32C12AAD376_H__