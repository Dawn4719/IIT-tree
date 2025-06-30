#include "Mem.h"

#include <sys/types.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <dirent.h>
#include <string>

int UtilityResource::g_cpu_count = sysconf(_SC_NPROCESSORS_CONF);
int UtilityResource::g_pid = getpid();
std::map<int, std::tuple<uint64_t, uint64_t, uint64_t, uint64_t>> UtilityResource::g_tid_cpus;

int UtilityResource::cpu_count()
{
    return g_cpu_count;
}

void UtilityResource::current_cpu(std::map<int, double>& cpus)
{
    _enmu_pid_sysfile();
    if (cpus.empty())
    {
        for (auto it = g_tid_cpus.begin(); it != g_tid_cpus.end(); ++it)
        {
            cpus[it->first] = 0.0f;
        }
    }
    for (auto it = cpus.begin(); it != cpus.end(); ++it)
    {
        auto itf = g_tid_cpus.find(it->first);
        if (itf != g_tid_cpus.end())
        {
            std::get<0>(itf->second) = std::get<1>(itf->second);
            std::get<1>(itf->second) = _get_process_time_tid(it->first);
            std::get<2>(itf->second) = std::get<3>(itf->second);
            std::get<3>(itf->second) = _get_process_time_all();
            it->second = double(std::get<1>(itf->second) - std::get<0>(itf->second)) / (std::get<3>(itf->second) - std::get<2>(itf->second)) * g_cpu_count * 100;
        }
        else
        {
            it->second = 0.0f;
        }
    }
}

uint64_t UtilityResource::total_mem()
{
    return _get_res_mem_all();
}

uint64_t UtilityResource::current_mem()
{
    return _get_res_mem_pid();
}

void UtilityResource::_enmu_pid_sysfile()
{
    auto last_tid_cpus = g_tid_cpus;
    g_tid_cpus.clear();
    DIR* dir = opendir("/proc/self/task");
    if (dir == NULL)
    {
        return;
    }
    struct dirent* entry;
    while ((entry = readdir(dir)))
    {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
        {
            continue;
        }
        int tid = std::atoi(entry->d_name);
        auto it = last_tid_cpus.find(tid);
        if (it != last_tid_cpus.end())
        {
            g_tid_cpus[tid] = it->second;
        }
        else
        {
            g_tid_cpus[tid] = std::forward_as_tuple(0, 0, 0, 0);
        }
    }
    closedir(dir);
}

std::string UtilityResource::_read_stat_file(const char* file_name)
{
    std::string stat_info;
    FILE* fd;
    if ((fd = fopen(file_name, "r")))
    {
        stat_info.resize(1024);
        fgets(const_cast<char*>(stat_info.c_str()), 1024, fd);
        stat_info.resize(strlen(stat_info.c_str()));
        fclose(fd);
    }
    return stat_info;
}

int UtilityResource::_get_process_time_all()
{
    std::string all_sysinfo = _read_stat_file("/proc/stat");

    char tcpu[7];
    unsigned int user, nice, sys, idle, iowait, irq, softirq, steal;
    sscanf(all_sysinfo.c_str(), "%s%d%d%d%d%d%d%d%d", tcpu, &user, &nice, &sys, &idle, &iowait, &irq, &softirq, &steal);
    return user + nice + sys + idle + iowait + irq + softirq + steal;
}

int UtilityResource::_get_process_time_tid(int tid)
{
    std::string tid_syspath = "/proc/self/task/";
    tid_syspath += std::to_string(tid) + "/stat";

    std::string tid_sysinfo = _read_stat_file(tid_syspath.c_str());

    const char *b = tid_sysinfo.c_str();
    const char *e = b + tid_sysinfo.size();
    for (int i = 0; i < 13 && b++ < e;)
    {
        while (*(b++) != ' ');
        ++i;
    }
    int utime, stime, cutime, cstime;
    sscanf(b, "%d%d%d%d", &utime, &stime, &cutime, &cstime);
    return utime + stime + cutime + cstime;
}

uint64_t UtilityResource::_get_res_mem_all()
{
    struct sysinfo s;
    if(0 == sysinfo(&s))
    {
        return s.totalram * s.mem_unit;
    }
    return 0;
}

uint64_t  UtilityResource::_get_res_mem_pid()
{
    std::string pid_meminfo = _read_stat_file("/proc/self/statm");
    uint64_t virt, res;
    sscanf(pid_meminfo.c_str(), "%lu%lu", &virt, &res);
    return res * (getpagesize() / 1024);
}