#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "ExternalTests/LiveGraphVsSuperNova.hpp"
#include "ExternalTests/SortledtonVsSuperNova.hpp"

namespace paper = APCDAGTests::PaperCSV;
using Case = APCDAGTests::BenchmarkCase;
using Fabric = APCDAGTests::RuntimeAPCFabricBackend;
using RowWrite = APCDAGTests::RowLockedVectorDAG<std::mutex, false>;
using RowRead = APCDAGTests::RowLockedVectorDAG<std::shared_mutex, true>;
using LiveAbsolute = LiveGraphVsSuperNova::LiveGraphAbsoluteNativeBackend;
using LiveNative = LiveGraphVsSuperNova::LiveGraphNativeDAGBackend;
using LiveBidir = LiveGraphVsSuperNova::LiveGraphBackend;
using SortAbsolute = SortledtonVsSuperNova::SortledtonAbsoluteNativeBackend;
using SortNative = SortledtonVsSuperNova::SortledtonNativeDAGBackend;
using SortBidir = SortledtonVsSuperNova::SortledtonBidirBackend;

namespace {
constexpr std::size_t runs = 5;
constexpr std::size_t threads = 18;
constexpr std::size_t small_n = 100;
constexpr std::size_t large_n = 1'000'000;
constexpr std::uint8_t small_k = 4;
constexpr std::uint8_t large_k = 32;
constexpr auto interval = std::chrono::milliseconds{1000};
constexpr std::array<const char*, 8> backend_names{
    "SuperNova", "RowLock", "LiveGraph absolute native",
    "LiveGraph native DAG", "LiveGraph bidirectional DAG",
    "Sortledton absolute native", "Sortledton native DAG",
    "Sortledton bidirectional DAG"};

constexpr const char* csv_header =
    "test,backend,contract,run,node_count,parent_capacity,writer_threads,reader_threads,mode,distribution,"
    "duration_seconds,successful_replacements,successful_reads,mutation_retries,read_retries,"
    "unfinished_mutations,invalid_mutations,invalid_reads,mutation_retries_per_success,read_retries_per_success,"
    "distinct_child_rows_written,minimum_child_row_line_bytes,quiescent_reads_per_second,write_only_per_second,"
    "mixed_reads_per_second,mixed_writes_per_second,read_retention,write_retention,"
    "aggregate_logical_ops_per_second,payload_words_per_node,construction_seconds,valid,sample_count\n";

struct Job {
    int test = 0;                 // 0: Test 1, 1: 2A, 2: 2B, 3: 3A, 4: 3B.
    int backend = 0;              // See the switch in Measure().
    Case c{};
    std::size_t repetition = 1;
    std::size_t worker_count = 0; // Writers in Test 2, readers in Test 3.
    bool skew = false;
};

template<class T>
bool MeasureBackend(std::ofstream& output, const Job& job,
                    const char* name, const char* contract) {
    if (job.test == 0)
        return paper::OneTest1<T>(output, name, contract, job.c, job.repetition);
    if (job.test == 1 || job.test == 2)
        return paper::OneMutation<T>(output, paper::TestNames[job.test], name,
            contract, job.c, job.repetition, job.worker_count,
            job.test == 2, job.skew, interval);
    return paper::OneMixed<T>(output, paper::TestNames[job.test], name,
        contract, job.c, job.repetition, job.worker_count,
        job.test == 4, job.skew, interval);
}

bool Measure(std::ofstream& output, const Job& job) {
    const Case absolute_case{job.c.NodeCount, 1};
    Job absolute_job = job;
    absolute_job.c = absolute_case;
    switch (job.backend) {
    case 0: return MeasureBackend<Fabric>(output, job,
        "SuperNova", "bounded_bidirectional_DAG");
    case 1:
        if (job.test == 1 || job.test == 2)
            return MeasureBackend<RowWrite>(output, job,
                "RowLock", "bounded_bidirectional_DAG");
        return MeasureBackend<RowRead>(output, job,
            "RowLock", "bounded_bidirectional_DAG");
    case 2: return MeasureBackend<LiveAbsolute>(output, absolute_job,
        "LiveGraph_absolute_native_lower_bound", "one_way_single_parent_no_reverse");
    case 3: return MeasureBackend<LiveNative>(output, job,
        "LiveGraph_native_DAG", "one_way_K_parent_no_reverse");
    case 4: return MeasureBackend<LiveBidir>(output, job,
        "LiveGraph_bidirectional_DAG", "bidirectional_K_parent_transaction");
    case 5: return MeasureBackend<SortAbsolute>(output, absolute_job,
        "Sortledton_absolute_native_lower_bound",
        job.test <= 2 ? "one_way_single_parent_no_reverse"
                      : "guarded_one_way_single_parent_no_reverse");
    case 6: return MeasureBackend<SortNative>(output, job,
        "Sortledton_native_DAG",
        job.test <= 2 ? "one_way_K_parent_no_reverse"
                      : "guarded_one_way_K_parent_no_reverse");
    case 7: return MeasureBackend<SortBidir>(output, job,
        "Sortledton_bidirectional_DAG",
        job.test <= 2 ? "bidirectional_K_parent_transaction"
                      : "guarded_bidirectional_K_parent_transaction");
    default: return false;
    }
}

// One process owns exactly one backend, geometry, thread count, distribution,
// and repetition. A library allocator may retain freed pages within a process;
// exiting the child releases them before the next graph is constructed.
bool RunIsolated(const Job& job, const std::string& path) {
    const pid_t pid = fork();
    if (pid < 0) {
        std::cerr << "fork failed: " << std::strerror(errno) << '\n';
        return false;
    }
    if (pid == 0) {
        std::ofstream output(path, std::ios::app);
        const bool ok = output && Measure(output, job);
        output.flush();
        const bool written = output.good();
        output.close();
        _exit(ok && written ? 0 : 1);
    }
    int status = 0;
    pid_t result;
    do { result = waitpid(pid, &status, 0); }
    while (result == -1 && errno == EINTR);
    if (result == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0)
        return true;
    std::cerr << "Failed: " << paper::TestNames[job.test]
              << " backend=" << backend_names[job.backend]
              << " N=" << job.c.NodeCount
              << " K=" << unsigned(job.c.ParentCapacity)
              << " threads=" << job.worker_count
              << " skew=" << job.skew << " run=" << job.repetition;
    if (result == pid && WIFSIGNALED(status))
        std::cerr << " signal=" << WTERMSIG(status)
                  << " (signal 9 may indicate the OOM killer)";
    else if (result == pid && WIFEXITED(status))
        std::cerr << " exit=" << WEXITSTATUS(status);
    else
        std::cerr << " waitpid: " << std::strerror(errno);
    std::cerr << '\n';
    return false;
}

bool RunTest(int test, const std::string& prefix) {
    const std::string path = prefix + "_" + paper::TestNames[test] + ".csv";
    std::ofstream output(path, std::ios::trunc);
    if (!output) { std::cerr << "Cannot create " << path << '\n'; return false; }
    output << csv_header;
    output.close();
    if (!output) return false;

    std::cout << "Starting " << paper::TestNames[test] << " -> " << path << '\n';
    for (int backend = 0; backend < 8; ++backend) {
        if (test == 0 && (backend == 2 || backend == 5)) continue;
        std::cout << "  Backend " << backend << "/7: "
                  << backend_names[backend] << std::endl;
        for (std::size_t n : {small_n, std::size_t{65'536}, std::size_t{262'144},
                              std::size_t{524'288}, large_n}) {
            if ((test == 0 || test == 1 || test == 3) && n != small_n) continue;
            for (std::uint8_t k : {small_k, large_k}) {
                // Only K=32 is used for the large working-set scale ladder.
                if (n != small_n && k != large_k) continue;
                // K=1 lower-bound mappings are independent of requested K.
                if ((backend == 2 || backend == 5) && k != large_k) continue;
                for (std::size_t count = 1; count <= threads - 2; ++count) {
                    if (test == 0 && count != 1) continue;
                    for (int skew = 0; skew < ((test == 2 || test == 4) ? 2 : 1); ++skew) {
                        std::cout << "    N=" << n << " K=" << unsigned(k)
                                  << " threads=" << count << " skew=" << skew
                                  << " runs=" << runs << std::endl;
                        for (std::size_t rep = 1; rep <= runs; ++rep) {
                            const Job job{test, backend, {n, k}, rep, count, bool(skew)};
                            std::cout << "      run " << rep << '/' << runs << std::endl;
                            if (!RunIsolated(job, path)) return false;
                        }
                    }
                }
            }
        }
    }
    if (!paper::AppendMedians(path, runs)) {
        std::cerr << "Cannot append valid medians to " << path << '\n';
        return false;
    }
    std::cout << "Completed " << path << " (raw samples and medians)\n";
    return true;
}
} // namespace

int main(int argc, char** argv) {
    int selected = -1;
    if (argc > 2) {
        std::cerr << "Usage: " << argv[0] << " [all|test1|test2a|test2b|test3a|test3b]\n";
        return 2;
    }
    if (argc == 2 && std::string(argv[1]) != "all") {
        bool found = false;
        const std::array<std::string, 5> choices{"test1", "test2a", "test2b", "test3a", "test3b"};
        for (int i = 0; i < 5; ++i)
            if (choices[i] == argv[1]) { selected = i; found = true; break; }
        if (!found) {
            std::cerr << "Usage: " << argv[0] << " [all|test1|test2a|test2b|test3a|test3b]\n";
            return 2;
        }
    }
    if (!APCDAGTests::ValidateRunArguments(small_n, large_n, small_k, large_k, threads))
        return 1;
    std::error_code error;
    std::filesystem::create_directories("results", error);
    if (error) { std::cerr << "Cannot create results/: " << error.message() << '\n'; return 1; }
    const std::string prefix = "results/SuperNova_LiveGraph_Sortledton_RowLock_N100_to_N1000000";
    for (int test = 0; test < 5; ++test)
        if ((selected < 0 || selected == test) && !RunTest(test, prefix)) return 1;
    return 0;
}
