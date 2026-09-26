#pragma once

// SortledtonVsSuperNova.hpp
//
// Expected layout:
//   ExternalTests/SortledtonVsSuperNova.hpp
//   sortledton/
//   SuperNova/
//
// C++20. This header uses Sortledton's public headers but Sortledton is NOT
// header-only: the executable must also link the compiled Sortledton library
// (or the exact upstream object files) plus SuperNova's atomiccim_core.
//
// int main() { return SortledtonVsSuperNova::Run(); }
//
// Default output:
//   SortledtonVsSuperNova_results.txt
//
// Benchmark representations:
//   S-native : one directed Sortledton edge per logical parent relation.
//              This is the native/lower-bound contract and intentionally has
//              no reverse-child adjacency.
//   S-bidir  : two directed Sortledton edges per logical relation so parent
//              lookup + reverse-child traversal match Fabric end to end.
//   Fabric   : SuperNova's native bounded bidirectional DAG representation.
//
// Sortledton has no relation-label channel matching Fabric's H/V axes in this
// adapter. H/V and direction are therefore encoded with disjoint vertex-ID
// ranges. The 1 KiB payload comparison is represented as 128 weighted edges
// per node in a separate Sortledton graph. This representation difference is
// disclosed in Test 1 and must not be described as a native vertex-property
// comparison.
//
// Test 1: quiescent construction, relation scans, payload scans, replacement,
//         integrity verification, and representation/footprint disclosure.
// Test 2: hotspot + distributed mutation; every backend receives the same
//         wall-clock measurement interval.
// Test 3: stable reads with two concurrent writers; every backend receives the
//         same wall-clock measurement interval.
//
// Retry-limit events are reported as progress/contention metrics and do not
// invalidate an otherwise healthy, integrity-correct sample.
#include "../sortledton/data-structure/TransactionManager.h"
#include "../sortledton/data-structure/VersioningBlockedSkipListAdjacencyList.h"
#include "../sortledton/data-structure/VersionedBlockedPropertyEdgeIterator.h"
#include "../SuperNova/core/headers/NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"
#include "../SuperNova/core/TestFiles/TestKit.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

namespace SortledtonVsSuperNova
{
using namespace APCDAGTests;
using namespace APCDAGTests::BenchmarkCore;

// The repository has no edge labels or vertex properties. H/V and direction are
// encoded with disjoint vertex ranges; each parent edge stores its ordinal as an
// eight-byte property. Payload uses a separate weighted graph with 128 edges/node.
// This encoding is disclosed in the output and is not a native vertex-property row.
template <bool Bidirectional>
class SortledtonBackend
{
public:
    bool Initialize(std::size_t nodes, std::size_t words, std::uint8_t k,
                    bool = false)
    {
        constexpr std::size_t SORTLEDTON_VERTEX_NAMESPACES = 4u;
        constexpr std::size_t SORTLEDTON_MAX_PAYLOAD_WORDS = 128u;

        if (!nodes ||
            nodes > UINT32_MAX / SORTLEDTON_VERTEX_NAMESPACES ||
            !words || words > SORTLEDTON_MAX_PAYLOAD_WORDS ||
            !k || k > ADS::COMPILED_MAX_DIRECT_PARENTS_PER_AXIS)
            return false;

        try
        {
            Fatal_.store(false, std::memory_order_release);
            Nodes_ = nodes;
            Words_ = words;
            K_ = k;

            // Thread id 0 is reserved for the benchmark/control thread.
            // Run() limits usable_thread_count so worker ids remain < 64.
            Manager_ = std::make_unique<TransactionManager>(64u);
            Manager_->register_thread(0u);
            Topology_ = std::make_unique<VersioningBlockedSkipListAdjacencyList>(
                128u, sizeof(std::uint64_t), *Manager_);
            Payload_ = std::make_unique<VersioningBlockedSkipListAdjacencyList>(
                128u, sizeof(std::uint64_t), *Manager_);
            if (!CreateVertices(*Topology_, (Bidirectional ? 4u : 2u) * nodes) ||
                !CreateVertices(*Payload_, nodes + words)) return false;
            Staging_.assign(words, 0u);
            StageNode_ = StageWord_ = 0u;
            return true;
        }
        catch (...) { Fatal_.store(true); return false; }
    }

    bool RegisterThread(std::size_t id) noexcept
    {
        if (!Manager_ || id >= 64u)
            return false;

        try
        {
            Manager_->register_thread(id);
            return true;
        }
        catch (...)
        {
            Fatal_.store(true, std::memory_order_release);
            return false;
        }
    }

    void DeregisterThread(std::size_t id) noexcept
    {
        if (!Manager_ || id >= 64u)
            return;

        try
        {
            Manager_->deregister_thread(id);
        }
        catch (...)
        {
            Fatal_.store(true, std::memory_order_release);
        }
    }
    bool Healthy() const noexcept { return !Fatal_.load(); }

    bool PrimePayloadStorage() noexcept
    {
        if (!Payload_ || !Manager_) return false;
        for (std::size_t node = 0; node < Nodes_; ++node)
        {
            if (!Write(*Payload_, [&](SnapshotTransaction& tx)
            {
                // Sortledton queues a pointer to the property until execute().
                static thread_local std::uint64_t zero = 0u;
                for (std::size_t word = 0; word < Words_; ++word)
                    tx.insert_edge(edge_t(node, Nodes_ + word),
                        reinterpret_cast<char*>(&zero), sizeof(zero));
                return true;
            })) return false;
        }
        return true;
    }

    bool StorePayload(std::size_t node, std::uint32_t word,
                      std::uint64_t value, bool) noexcept
    {
        if (node >= Nodes_ || node != StageNode_ ||
            word != StageWord_ || word >= Words_ ||
            Fatal_.load(std::memory_order_acquire))
            return false;
        Staging_[word] = value;
        if (++StageWord_ < Words_) return true;
        const bool ok = Write(*Payload_, [&](SnapshotTransaction& tx)
        {
            for (std::size_t i = 0; i < Words_; ++i)
                tx.insert_edge(edge_t(node, Nodes_ + i),
                    reinterpret_cast<char*>(&Staging_[i]), sizeof(std::uint64_t));
            return true;
        });
        if (ok) { ++StageNode_; StageWord_ = 0u; }
        return ok;
    }

    bool LoadPayload(std::size_t node, std::uint32_t word,
                     std::uint64_t& value, bool) noexcept
    {
        if (node >= Nodes_ || word >= Words_ ||
            Fatal_.load(std::memory_order_acquire))
            return false;
        return Read(*Payload_, [&](SnapshotTransaction& tx)
        { return tx.get_weight(edge_t(node, Nodes_ + word),
                    reinterpret_cast<char*>(&value)); });
    }

    bool AddParent(std::size_t parent, std::size_t child, Axis axis,
                   std::uint32_t = DEFAULT_MAX_TRIES) noexcept
    {
        if (!Legal(parent, child) ||
            Fatal_.load(std::memory_order_acquire))
            return false;
        return Write(*Topology_, [&](SnapshotTransaction& tx)
        {
            const edge_t back(BackSource(child, axis), parent);
            if (tx.has_edge(back)) return false;
            static thread_local std::uint64_t ordinal = 0u;
            ordinal = 0u;
            for (std::uint8_t i = 0u; i < K_; ++i)
            {
                bool used = false;
                if (!ForEdges(tx, BackSource(child, axis),
                    [&](std::size_t, std::uint64_t stored)
                    { if (stored == i) used = true; })) return false;
                if (!used) { ordinal = i; break; }
                if (i + 1u == K_) return false;
            }
            if constexpr (Bidirectional)
            {
                if (tx.has_edge(edge_t(FrontSource(parent, axis), child)))
                    return false;
            }
            tx.insert_edge(back, reinterpret_cast<char*>(&ordinal), sizeof(ordinal));
            if constexpr (Bidirectional)
                tx.insert_edge(edge_t(FrontSource(parent, axis), child),
                    reinterpret_cast<char*>(&ordinal), sizeof(ordinal));
            return true;
        });
    }

    bool ReplaceParent(std::size_t old_parent, std::size_t new_parent,
                       std::size_t child, Axis axis,
                       std::uint32_t = DEFAULT_MAX_TRIES) noexcept
    {
        if (!Legal(old_parent, child) || !Legal(new_parent, child) ||
            old_parent == new_parent ||
            Fatal_.load(std::memory_order_acquire))
            return false;
        return Write(*Topology_, [&](SnapshotTransaction& tx)
        {
            const edge_t old_back(BackSource(child, axis), old_parent);
            const edge_t new_back(BackSource(child, axis), new_parent);
            static thread_local std::uint64_t ordinal = 0u;
            ordinal = 0u;
            if (!tx.get_weight(old_back, reinterpret_cast<char*>(&ordinal)) ||
                ordinal >= K_ || tx.has_edge(new_back)) return false;
            if constexpr (Bidirectional)
            {
                const edge_t old_front(FrontSource(old_parent, axis), child);
                const edge_t new_front(FrontSource(new_parent, axis), child);
                std::uint64_t check = UINT64_MAX;
                if (!tx.get_weight(old_front, reinterpret_cast<char*>(&check)) ||
                    check != ordinal || tx.has_edge(new_front)) return false;
                tx.delete_edge(old_front);
                tx.insert_edge(new_front, reinterpret_cast<char*>(&ordinal),
                               sizeof(ordinal));
            }
            tx.delete_edge(old_back);
            tx.insert_edge(new_back, reinterpret_cast<char*>(&ordinal),
                           sizeof(ordinal));
            return true;
        });
    }

    ReadResult FindParent(std::size_t child, Axis axis, std::uint8_t ordinal,
                          std::uint32_t = 1u) noexcept
    {
        if (child >= Nodes_ || ordinal >= K_ ||
            Fatal_.load(std::memory_order_acquire))
            return {};
        ReadResult result{};
        const bool ok = Read(*Topology_, [&](SnapshotTransaction& tx)
        {
            return ForEdges(tx, BackSource(child, axis),
                [&](std::size_t parent, std::uint64_t stored)
                {
                    if (stored == ordinal)
                    {
                        if (!Legal(parent, child) || result.IsFound())
                            Fatal_.store(true);
                        else result = Found(parent, child, ordinal);
                    }
                });
        });
        return ok && !Fatal_.load() ? result : ReadResult{};
    }
    ReadResult StableFindParent(std::size_t child, Axis axis, std::uint8_t ordinal,
                                std::uint32_t tries = 1u) noexcept
    { return FindParent(child, axis, ordinal, tries); }

    ReadResult FindFirstChild(std::size_t parent, Axis axis,
                              std::uint32_t = 1u) noexcept
    { return ChildRead(parent, axis, 0u, false); }
    ReadResult FindLastChild(std::size_t parent, Axis axis,
                             std::uint32_t = 1u) noexcept
    { return ChildRead(parent, axis, UINT32_MAX, true); }
    ReadResult FindNextChild(std::size_t parent, Axis axis,
                             std::uint32_t cursor, std::uint32_t = 1u) noexcept
    { return ChildRead(parent, axis, cursor, false); }
    ReadResult FindPreviousChild(std::size_t parent, Axis axis,
                                 std::uint32_t cursor, std::uint32_t = 1u) noexcept
    { return ChildRead(parent, axis, cursor, true); }

    std::uint64_t BenchmarkParentScan(Axis axis, std::uint32_t rounds) noexcept
    {
        std::uint64_t sum = 0u;
        for (std::uint32_t r = 0; r < rounds; ++r)
            if (!Read(*Topology_, [&](SnapshotTransaction& tx)
            {
                for (std::size_t child = 0; child < Nodes_; ++child)
                {
                    std::array<ReadResult, 64u> found{};
                    if (!ForEdges(tx, BackSource(child, axis),
                        [&](std::size_t parent, std::uint64_t ord)
                        {
                            if (ord >= K_ || !Legal(parent, child) ||
                                found[ord].IsFound()) Fatal_.store(true);
                            else found[ord] = Found(parent, child, ord);
                        })) return false;
                    for (std::size_t i = 0; i < K_; ++i)
                        sum += found[i].Locator +
                            static_cast<std::uint64_t>(found[i].Outcome);
                }
                return true;
            })) break;
        return sum;
    }

    std::uint64_t BenchmarkReverseScan(Axis axis, bool payload,
                                       std::uint32_t rounds) noexcept
    {
        std::uint64_t sum = 0u;
        if constexpr (Bidirectional)
        {
            for (std::uint32_t r = 0; r < rounds; ++r)
            {
                std::vector<std::size_t> payload_nodes;
                if (!Read(*Topology_, [&](SnapshotTransaction& tx)
                {
                    for (std::size_t parent = 0; parent < Nodes_; ++parent)
                    {
                        if (!ForEdges(tx, FrontSource(parent, axis),
                            [&](std::size_t child, std::uint64_t ord)
                            {
                                if (ord >= K_ || !Legal(parent, child))
                                { Fatal_.store(true); return; }
                                sum += child * K_ + ord;
                                if (payload)
                                    payload_nodes.push_back(child);
                            })) return false;
                        sum += static_cast<std::uint64_t>(ReadOperation::NONE);
                    }
                    return true;
                })) break;
                // Sortledton's transaction manager permits one active snapshot
                // per thread, so close the topology snapshot before payload IO.
                for (std::size_t child : payload_nodes)
                {
                    std::uint64_t value = 0u;
                    if (!LoadPayload(child, child % Words_, value, false))
                    { Fatal_.store(true); break; }
                    sum ^= value;
                }
            }
        }
        return sum;
    }

    std::uint64_t BenchmarkPayloadScan(bool, std::uint32_t rounds) noexcept
    {
        std::uint64_t sum = 0u;
        for (std::uint32_t r = 0; r < rounds; ++r)
            if (!Read(*Payload_, [&](SnapshotTransaction& tx)
            {
                for (std::size_t node = 0; node < Nodes_; ++node)
                {
                    std::size_t count = 0;
                    if (!ForEdges(tx, node,
                        [&](std::size_t target, std::uint64_t value)
                        {
                            if (target < Nodes_ || target >= Nodes_ + Words_)
                                Fatal_.store(true);
                            ++count; sum += value;
                        })) return false;
                    if (count != Words_) Fatal_.store(true);
                }
                return true;
            })) break;
        return sum;
    }

    bool VerifyFullTest1Graph(const BenchmarkCase& config) noexcept
    {
        if (config.NodeCount != Nodes_ || config.ParentCapacity != K_) return false;
        for (Axis axis : {Axis::HORIZONTAL, Axis::VERTICAL})
            if (!Read(*Topology_, [&](SnapshotTransaction& tx)
            {
                std::vector<std::size_t> expected_children(Nodes_, 0u);
                for (std::size_t child = 0; child < Nodes_; ++child)
                {
                    const std::size_t count = std::min<std::size_t>(K_, child);
                    std::array<bool, 64u> seen{};
                    std::size_t observed = 0u;
                    if (!ForEdges(tx, BackSource(child, axis),
                        [&](std::size_t parent, std::uint64_t ord)
                        {
                            if (ord >= count || seen[ord] ||
                                parent != (axis == Axis::HORIZONTAL
                                    ? child - 1u - ord : ord))
                            { Fatal_.store(true); return; }
                            seen[ord] = true; ++observed;
                            ++expected_children[parent];
                        })) return false;
                    if (observed != count) return false;
                }
                if constexpr (Bidirectional)
                {
                    for (std::size_t parent = 0; parent < Nodes_; ++parent)
                    {
                        std::size_t observed = 0u;
                        if (!ForEdges(tx, FrontSource(parent, axis),
                            [&](std::size_t child, std::uint64_t ord)
                            {
                                const std::size_t count = std::min<std::size_t>(K_, child);
                                if (ord >= count || child <= parent ||
                                    parent != (axis == Axis::HORIZONTAL
                                        ? child - 1u - ord : ord))
                                { Fatal_.store(true); return; }
                                ++observed;
                            })) return false;
                        if (observed != expected_children[parent]) return false;
                    }
                }
                return true;
            })) return false;
        return Healthy();
    }

    bool VerifyPayloadPattern() noexcept
    {
        return Read(*Payload_, [&](SnapshotTransaction& tx)
        {
            for (std::size_t node = 0; node < Nodes_; ++node)
            {
                std::size_t count = 0u;
                std::vector<bool> seen(Words_, false);
                if (!ForEdges(tx, node,
                    [&](std::size_t target, std::uint64_t value)
                    {
                        if (target < Nodes_ || target >= Nodes_ + Words_)
                        { Fatal_.store(true); return; }
                        const std::size_t word = target - Nodes_;
                        if (seen[word] ||
                            value != ((static_cast<std::uint64_t>(node + 1u) << 32u)
                                ^ static_cast<std::uint64_t>(word + 1u)))
                        { Fatal_.store(true); return; }
                        seen[word] = true; ++count;
                    })) return false;
                if (count != Words_) return false;
            }
            return true;
        });
    }

private:
    bool Legal(std::size_t parent, std::size_t child) const noexcept
    { return parent < child && child < Nodes_; }
    std::size_t BackSource(std::size_t child, Axis axis) const noexcept
    { return child + (axis == Axis::HORIZONTAL ? 0u : Nodes_); }
    std::size_t FrontSource(std::size_t parent, Axis axis) const noexcept
    { return parent + (axis == Axis::HORIZONTAL ? 2u : 3u) * Nodes_; }
    ReadResult Found(std::size_t node, std::size_t child,
                     std::uint64_t ord) const noexcept
    { return {node, static_cast<std::uint32_t>(child * K_ + ord),
              ReadOperation::FOUND, true}; }

    template <class Fn>
    bool Write(VersioningBlockedSkipListAdjacencyList& ds, Fn&& fn) noexcept
    {
        if (Fatal_.load()) return false;
        try
        {
            auto tx = Manager_->getSnapshotTransaction(&ds, false);
            bool ok = false;
            try { ok = fn(tx) && tx.execute(); }
            catch (...)
            {
                // Upstream rolls back inserted vertices but does not generally
                // undo already applied edge edits after an execute exception.
                // Treat that as fatal rather than assuming an unchanged graph.
                Fatal_.store(true);
                Manager_->transactionCompleted(tx);
                return false;
            }
            Manager_->transactionCompleted(tx);
            return ok;
        }
        catch (...) { Fatal_.store(true); return false; }
    }
    template <class Fn>
    bool Read(VersioningBlockedSkipListAdjacencyList& ds, Fn&& fn) noexcept
    {
        if (Fatal_.load()) return false;
        try
        {
            auto tx = Manager_->getSnapshotTransaction(&ds, false);
            bool ok = false;
            try { ok = fn(tx); }
            catch (...) { Fatal_.store(true); }
            Manager_->transactionCompleted(tx);
            return ok && !Fatal_.load();
        }
        catch (...) { Fatal_.store(true); return false; }
    }
    bool CreateVertices(VersioningBlockedSkipListAdjacencyList& ds,
                        std::size_t count) noexcept
    {
        for (std::size_t base = 0; base < count; base += 256u)
            if (!Write(ds, [&](SnapshotTransaction& tx)
            {
                for (std::size_t id = base; id < std::min(count, base + 256u); ++id)
                    tx.insert_vertex(id);
                return true;
            })) return false;
        return true;
    }
    template <class Fn>
    bool ForEdges(SnapshotTransaction& tx, std::size_t source, Fn&& visit)
    {
        auto iter = tx.neighbourhood_with_properties_blocked_p(
            tx.physical_id(source));
        while (iter.has_next_block())
        {
            auto [versioned, begin, end, weights, unused] =
                iter.next_block_with_properties();
            (void)unused;
            if (versioned)
            {
                while (iter.has_next_edge())
                {
                    auto [dst, weight] = iter.next_with_properties();
                    std::uint64_t bits = 0u;
                    std::memcpy(&bits, &weight, sizeof(bits));
                    visit(static_cast<std::size_t>(tx.logical_id(dst)), bits);
                }
            }
            else
            {
                for (auto edge = begin; edge < end; ++edge, ++weights)
                {
                    std::uint64_t bits = 0u;
                    std::memcpy(&bits, weights, sizeof(bits));
                    visit(static_cast<std::size_t>(tx.logical_id(*edge)), bits);
                }
            }
        }
        return !Fatal_.load();
    }
    ReadResult ChildRead(std::size_t parent, Axis axis,
                         std::uint32_t cursor, bool reverse) noexcept
    {
        if constexpr (!Bidirectional) return {};
        if (parent >= Nodes_) return {};
        ReadResult best{};
        const bool ok = Read(*Topology_, [&](SnapshotTransaction& tx)
        {
            return ForEdges(tx, FrontSource(parent, axis),
                [&](std::size_t child, std::uint64_t ord)
                {
                    if (ord >= K_ || !Legal(parent, child))
                    { Fatal_.store(true); return; }
                    const auto candidate = Found(child, child, ord);
                    if ((!reverse && candidate.Locator > cursor &&
                         (!best.IsFound() || candidate.Locator < best.Locator)) ||
                        (reverse && candidate.Locator < cursor &&
                         (!best.IsFound() || candidate.Locator > best.Locator)))
                        best = candidate;
                });
        });
        return ok ? best : ReadResult{};
    }

    // Destroy graphs before their manager. The main thread stays registered
    // until manager destruction, after all worker threads have been joined.
    std::unique_ptr<TransactionManager> Manager_{};
    std::unique_ptr<VersioningBlockedSkipListAdjacencyList> Topology_{};
    std::unique_ptr<VersioningBlockedSkipListAdjacencyList> Payload_{};
    std::size_t Nodes_ = 0u, Words_ = 0u;
    std::uint8_t K_ = 0u;
    std::vector<std::uint64_t> Staging_{};
    std::size_t StageNode_ = 0u, StageWord_ = 0u;
    std::atomic<bool> Fatal_{false};
};

using SortledtonBidirBackend = SortledtonBackend<true>;
using SortledtonNativeBackend = SortledtonBackend<false>;

namespace ExternalFairness
{
template <class Backend>
class WorkerRegistration
{
public:
    WorkerRegistration(Backend& backend, std::size_t id) noexcept
        : Backend_(backend), Id_(id)
    {
        if constexpr (requires { backend.RegisterThread(id); })
            Ok_ = backend.RegisterThread(id);
    }
    ~WorkerRegistration() noexcept
    {
        if constexpr (requires { Backend_.DeregisterThread(Id_); })
            if (Ok_) Backend_.DeregisterThread(Id_);
    }
    bool Ok() const noexcept { return Ok_; }
private:
    Backend& Backend_;
    std::size_t Id_;
    bool Ok_ = true;
};

using Duration = std::chrono::milliseconds;
constexpr std::uint32_t RETRY_ATTEMPT_LIMIT = 4'096u;
constexpr std::uint32_t STARVATION_ROUND_LIMIT = 16u;
constexpr std::uint32_t DEFAULT_MEASUREMENT_MS = 250u;
// Every sample uses a freshly built backend. A warmup on that same backend
// can exhaust retries and change its graph before the measurement starts.
constexpr std::uint32_t DEADLINE_CHECK_EVERY = 64u;

inline bool HealthyBackend(auto& backend) noexcept
{
    if constexpr (requires { backend.Healthy(); }) return backend.Healthy();
    return true;
}

// Native one-way Sortledton intentionally has no reverse-child adjacency.
// A runtime `require_reverse == false` does not prevent template instantiation
// of ReverseContains(), so reverse verification also needs a compile-time guard.
template <class Backend>
inline constexpr bool HasReverseTraversal =
    requires(
        Backend& backend,
        std::size_t parent,
        Axis axis,
        std::uint32_t locator)
    {
        backend.FindFirstChild(parent, axis, DEFAULT_MAX_TRIES);
        backend.FindNextChild(parent, axis, locator, DEFAULT_MAX_TRIES);
    };

template <class Backend>
bool VerifyReverseIfRequested(
    Backend& backend,
    bool require_reverse,
    std::size_t parent,
    std::size_t child,
    Axis axis,
    std::size_t node_count)
{
    if (!require_reverse) return true;

    if constexpr (HasReverseTraversal<Backend>)
    {
        return ReverseContains(backend, parent, child, axis, node_count);
    }
    else
    {
        // Native one-way Sortledton is valid only when the caller intentionally
        // disables reverse-child verification.
        return false;
    }
}

enum class AttemptStatus : std::uint8_t
{
    SUCCESS,
    DEADLINE,
    RETRY_LIMIT,
    FATAL
};
using TimePoint = Clock::time_point;

// One logical replacement is one completed public backend operation. A retry
// budget hit is a measured progress/contention event, not corruption. Only an
// unhealthy backend is FATAL. This matters for fixed-duration throughput tests:
// one unlucky 4096-attempt operation must not truncate an otherwise valid sample.
template <class Backend>
AttemptStatus RetryReplaceTimed(Backend& backend, std::size_t old_parent,
    std::size_t new_parent, std::size_t child, Axis axis, TimePoint deadline,
    std::uint64_t& retries) noexcept
{
    for (std::uint32_t attempt = 0; attempt < RETRY_ATTEMPT_LIMIT; ++attempt)
    {
        if (attempt != 0u && (attempt & 63u) == 0u && Clock::now() >= deadline)
            return AttemptStatus::DEADLINE;

        if (backend.ReplaceParent(old_parent, new_parent, child, axis, 1u))
            return AttemptStatus::SUCCESS;

        ++retries;
        if (!HealthyBackend(backend))
            return AttemptStatus::FATAL;

        PerturbSchedule(attempt);
    }

    return AttemptStatus::RETRY_LIMIT;
}

struct TimedMutationResult
{
    bool Ok = false;
    bool FinalVerificationOk = false;
    double ElapsedNs = 0.0;
    std::uint64_t Success = 0u;
    std::uint64_t Retries = 0u;
    std::uint64_t Exhaustions = 0u; // retry-limit hits, not integrity failures

    // Aggregate inverse throughput, not individual call latency.
    double NsPerSuccess() const noexcept
    { return Ok && Success ? ElapsedNs / static_cast<double>(Success) : 0.0; }

    double Mops() const noexcept
    { return Ok && ElapsedNs > 0.0 ? static_cast<double>(Success) * 1000.0 / ElapsedNs : 0.0; }

    double RetryRate() const noexcept
    { return Success ? static_cast<double>(Retries) / static_cast<double>(Success) : 0.0; }

    double RetryLimitRate() const noexcept
    { return Success ? static_cast<double>(Exhaustions) / static_cast<double>(Success) : 0.0; }
};

template <class Backend>
bool VerifyTimedMutationState(Backend& backend, const MutationScenario& scenario,
    std::size_t writer_count, const std::vector<std::size_t>& final_h,
    const std::vector<std::size_t>& final_v, bool require_reverse)
{
    for (std::size_t writer = 0; writer < writer_count; ++writer)
    {
        const std::size_t child = scenario.Child(writer);
        for (const auto& [axis, expected] :
             std::array<std::pair<Axis, std::size_t>, 2u>{{
                 {Axis::HORIZONTAL, final_h[writer]},
                 {Axis::VERTICAL, final_v[writer]}}})
        {
            const ReadResult read = backend.FindParent(child, axis, 0u, DEFAULT_MAX_TRIES);
            if (!read.IsFound() || read.Node != expected) return false;
            if (!VerifyReverseIfRequested(backend, require_reverse, expected,
                    child, axis, scenario.Config.NodeCount)) return false;
        }
    }
    return HealthyBackend(backend);
}

template <class Backend>
TimedMutationResult RunTimedMutationWorkers(Backend& backend,
    const MutationScenario& scenario, const MutationSchedule& schedule,
    std::size_t writer_count, Duration measurement, bool require_reverse)
{
    struct Worker
    {
        std::uint64_t success = 0u;
        std::uint64_t retries = 0u;
        std::uint64_t exhausted = 0u;
        std::size_t h = 0u;
        std::size_t v = 0u;
    };

    std::vector<Worker> stats(writer_count);
    std::atomic<bool> fatal_failure{false};
    TimePoint begin{}, deadline{};

    std::barrier start_barrier(
        static_cast<std::ptrdiff_t>(writer_count + 1u),
        [&]() noexcept
        {
            begin = Clock::now();
            deadline = begin + measurement;
        });

    std::vector<std::thread> threads;
    threads.reserve(writer_count);

    for (std::size_t writer = 0u; writer < writer_count; ++writer)
    {
        threads.emplace_back([&, writer]() noexcept
        {
            Worker& out = stats[writer];
            WorkerRegistration registration(backend, writer + 1u);
            if (!registration.Ok()) fatal_failure.store(true);
            const std::size_t child = scenario.Child(writer);
            out.h = scenario.InitialH(writer);
            out.v = scenario.InitialV(writer);

            std::size_t index = 0u;
            std::uint32_t until_check = 0u;
            start_barrier.arrive_and_wait();

            while (true)
            {
                if (until_check-- == 0u)
                {
                    if (fatal_failure.load(std::memory_order_relaxed) ||
                        Clock::now() >= deadline)
                    {
                        break;
                    }
                    until_check = DEADLINE_CHECK_EVERY - 1u;
                }

                const MutationStep step =
                    schedule[writer][index++ % schedule[writer].size()];

                auto replace_axis = [&](std::size_t& current,
                                        std::size_t target,
                                        Axis axis) noexcept -> bool
                {
                    if (current == target)
                        return true;

                    const AttemptStatus status = RetryReplaceTimed(
                        backend, current, target, child, axis, deadline, out.retries);

                    switch (status)
                    {
                    case AttemptStatus::SUCCESS:
                        current = target;
                        ++out.success;
                        return true;

                    case AttemptStatus::DEADLINE:
                        return false;

                    case AttemptStatus::RETRY_LIMIT:
                        // The relation is unchanged because ReplaceParent failed
                        // atomically. Count the progress event and keep the timed
                        // sample running rather than throwing away the whole sample.
                        ++out.exhausted;
                        std::this_thread::yield();
                        return true;

                    case AttemptStatus::FATAL:
                    default:
                        fatal_failure.store(true, std::memory_order_release);
                        return false;
                    }
                };

                if (!replace_axis(out.h, step.HParent, Axis::HORIZONTAL))
                    break;

                if (!replace_axis(out.v, step.VParent, Axis::VERTICAL))
                    break;
            }
        });
    }

    start_barrier.arrive_and_wait();

    for (auto& thread : threads)
        thread.join();

    const auto end = Clock::now();

    std::vector<std::size_t> final_h(writer_count), final_v(writer_count);
    TimedMutationResult result{};

    for (std::size_t i = 0u; i < writer_count; ++i)
    {
        final_h[i] = stats[i].h;
        final_v[i] = stats[i].v;
        result.Success += stats[i].success;
        result.Retries += stats[i].retries;
        result.Exhaustions += stats[i].exhausted;
    }

    result.ElapsedNs =
        std::chrono::duration<double, std::nano>(end - begin).count();

    result.FinalVerificationOk = VerifyTimedMutationState(
        backend, scenario, writer_count, final_h, final_v, require_reverse);

    result.Ok =
        !fatal_failure.load(std::memory_order_acquire) &&
        result.Success != 0u &&
        result.FinalVerificationOk;

    return result;
}

enum class TimedStableReadStatus : std::uint8_t
{ SUCCESS, DEADLINE, RETRY_LIMIT, BAD_CONTRACT, INVALID_PARENT };

template <class Backend>
TimedStableReadStatus StableReadOneTimed(Backend& backend,
    const ReaderScenario& scenario, std::size_t writer, TimePoint deadline,
    std::uint64_t& retries) noexcept
{
    const WriterSpec& spec = scenario.Writers[writer];
    for (std::uint32_t attempt = 0; attempt < RETRY_ATTEMPT_LIMIT; ++attempt)
    {
        if (attempt != 0u && (attempt & 63u) == 0u && Clock::now() >= deadline)
            return TimedStableReadStatus::DEADLINE;
        const ReadResult read = backend.StableFindParent(
            spec.Child, spec.RelationAxis, 0u, 1u);
        if (!read.ContractValid()) return TimedStableReadStatus::BAD_CONTRACT;
        if (read.IsRetry())
        { ++retries; PerturbSchedule(attempt); continue; }
        if (!read.IsFound() || !scenario.ParentAllowed(writer, read.Node))
            return TimedStableReadStatus::INVALID_PARENT;
        return TimedStableReadStatus::SUCCESS;
    }
    return TimedStableReadStatus::RETRY_LIMIT;
}

struct TimedReaderResult
{
    bool Ok = false, CorrectnessOk = false, FinalVerificationOk = false;
    double ElapsedNs = 0.0;
    std::uint64_t StableReads = 0u, ReaderRetries = 0u, ReaderStarvations = 0u;
    std::uint64_t WriterSuccess = 0u, WriterRetries = 0u, WriterExhaustions = 0u;

    double NsPerStableRead() const noexcept
    { return Ok && StableReads ? ElapsedNs / static_cast<double>(StableReads) : 0.0; }

    double ReadMops() const noexcept
    { return Ok && ElapsedNs > 0.0 ? static_cast<double>(StableReads) * 1000.0 / ElapsedNs : 0.0; }

    double WriterMops() const noexcept
    { return Ok && ElapsedNs > 0.0 ? static_cast<double>(WriterSuccess) * 1000.0 / ElapsedNs : 0.0; }

    double RetryRate() const noexcept
    { return StableReads ? static_cast<double>(ReaderRetries) / static_cast<double>(StableReads) : 0.0; }

    double ReaderRetryLimitRate() const noexcept
    { return StableReads ? static_cast<double>(ReaderStarvations) / static_cast<double>(StableReads) : 0.0; }

    double WriterRetryLimitRate() const noexcept
    { return WriterSuccess ? static_cast<double>(WriterExhaustions) / static_cast<double>(WriterSuccess) : 0.0; }
};

template <class Backend>
bool VerifyTimedReaderState(Backend& backend, const ReaderScenario& scenario,
    bool require_reverse)
{
    for (std::size_t writer = 0; writer < scenario.Writers.size(); ++writer)
    {
        const WriterSpec& spec = scenario.Writers[writer];
        const ReadResult read = backend.FindParent(
            spec.Child, spec.RelationAxis, 0u, DEFAULT_MAX_TRIES);
        if (!read.IsFound() || !scenario.ParentAllowed(writer, read.Node)) return false;
        if (!VerifyReverseIfRequested(backend, require_reverse, read.Node,
                spec.Child, spec.RelationAxis, scenario.Config.NodeCount)) return false;
    }
    return HealthyBackend(backend);
}

template <class Backend>
TimedReaderResult RunTimedReadersWithWriters(Backend& backend,
    const ReaderScenario& scenario, const ParentSchedule& schedule,
    std::size_t reader_count, Duration measurement, bool require_reverse)
{
    constexpr std::size_t WRITERS = ConcurrencyConfig::READER_WRITER_COUNT;

    struct Writer
    {
        std::uint64_t success = 0u;
        std::uint64_t retries = 0u;
        std::uint64_t exhausted = 0u;
    };

    struct Reader
    {
        std::uint64_t success = 0u;
        std::uint64_t retries = 0u;
        std::uint64_t starved = 0u;
    };

    std::array<Writer, WRITERS> writers{};
    std::vector<Reader> readers(reader_count);
    std::atomic<bool> fatal_failure{false};
    std::atomic<bool> correctness_failed{false};
    TimePoint begin{}, deadline{};

    std::barrier start_barrier(
        static_cast<std::ptrdiff_t>(WRITERS + reader_count + 1u),
        [&]() noexcept
        {
            begin = Clock::now();
            deadline = begin + measurement;
        });

    std::vector<std::thread> threads;
    threads.reserve(WRITERS + reader_count);

    for (std::size_t writer = 0u; writer < WRITERS; ++writer)
    {
        threads.emplace_back([&, writer]() noexcept
        {
            Writer& out = writers[writer];
            WorkerRegistration registration(backend, writer + 1u);
            if (!registration.Ok()) fatal_failure.store(true);
            const WriterSpec spec = scenario.Writers[writer];
            std::size_t current = spec.InitialParent;
            std::size_t index = 0u;
            std::uint32_t until_check = 0u;

            start_barrier.arrive_and_wait();

            while (true)
            {
                if (until_check-- == 0u)
                {
                    if (fatal_failure.load(std::memory_order_relaxed) ||
                        correctness_failed.load(std::memory_order_relaxed) ||
                        Clock::now() >= deadline)
                    {
                        break;
                    }
                    until_check = DEADLINE_CHECK_EVERY - 1u;
                }

                const std::size_t target =
                    schedule[writer][index++ % schedule[writer].size()];

                if (target == current)
                    continue;

                const AttemptStatus status = RetryReplaceTimed(
                    backend, current, target, spec.Child,
                    spec.RelationAxis, deadline, out.retries);

                if (status == AttemptStatus::SUCCESS)
                {
                    current = target;
                    ++out.success;
                    continue;
                }

                if (status == AttemptStatus::DEADLINE)
                    break;

                if (status == AttemptStatus::RETRY_LIMIT)
                {
                    ++out.exhausted;
                    std::this_thread::yield();
                    continue;
                }

                fatal_failure.store(true, std::memory_order_release);
                break;
            }
        });
    }

    for (std::size_t reader = 0u; reader < reader_count; ++reader)
    {
        threads.emplace_back([&, reader]() noexcept
        {
            Reader& out = readers[reader];
            WorkerRegistration registration(backend, WRITERS + reader + 1u);
            if (!registration.Ok()) fatal_failure.store(true);
            const std::size_t observed = reader % WRITERS;
            std::uint32_t until_check = 0u;

            start_barrier.arrive_and_wait();

            while (true)
            {
                if (until_check-- == 0u)
                {
                    if (fatal_failure.load(std::memory_order_relaxed) ||
                        correctness_failed.load(std::memory_order_relaxed) ||
                        Clock::now() >= deadline)
                    {
                        break;
                    }
                    until_check = DEADLINE_CHECK_EVERY - 1u;
                }

                const TimedStableReadStatus status =
                    StableReadOneTimed(backend, scenario, observed, deadline, out.retries);

                if (status == TimedStableReadStatus::SUCCESS)
                {
                    ++out.success;
                    continue;
                }

                if (status == TimedStableReadStatus::DEADLINE)
                    break;

                if (status == TimedStableReadStatus::RETRY_LIMIT)
                {
                    // Starvation is reported, but the equal-time sample continues.
                    ++out.starved;
                    std::this_thread::yield();
                    continue;
                }

                correctness_failed.store(true, std::memory_order_release);
                break;
            }
        });
    }

    start_barrier.arrive_and_wait();

    for (auto& thread : threads)
        thread.join();

    const auto end = Clock::now();
    TimedReaderResult result{};

    result.ElapsedNs =
        std::chrono::duration<double, std::nano>(end - begin).count();

    for (const auto& writer : writers)
    {
        result.WriterSuccess += writer.success;
        result.WriterRetries += writer.retries;
        result.WriterExhaustions += writer.exhausted;
    }

    for (const auto& reader : readers)
    {
        result.StableReads += reader.success;
        result.ReaderRetries += reader.retries;
        result.ReaderStarvations += reader.starved;
    }

    result.FinalVerificationOk =
        VerifyTimedReaderState(backend, scenario, require_reverse);

    result.CorrectnessOk =
        !correctness_failed.load(std::memory_order_acquire) &&
        result.FinalVerificationOk;

    result.Ok =
        result.CorrectnessOk &&
        !fatal_failure.load(std::memory_order_acquire) &&
        result.StableReads != 0u &&
        result.WriterSuccess != 0u;

    return result;
}


inline std::string FormatThroughput(double mops)
{
    const double ops_per_second = mops * 1'000'000.0;
    std::ostringstream out;

    if (ops_per_second >= 1'000'000.0)
    {
        out << std::fixed << std::setprecision(2)
            << (ops_per_second / 1'000'000.0) << " Mops/s";
    }
    else if (ops_per_second >= 1'000.0)
    {
        out << std::fixed << std::setprecision(1)
            << (ops_per_second / 1'000.0) << " Kops/s";
    }
    else
    {
        out << std::fixed << std::setprecision(0)
            << ops_per_second << " ops/s";
    }

    return out.str();
}

inline std::string FormatRate(double value)
{
    if (value == 0.0)
        return "0";

    std::ostringstream out;
    if (value < 0.0001)
        out << std::scientific << std::setprecision(2) << value;
    else
        out << std::fixed << std::setprecision(4) << value;

    return out.str();
}

inline std::string FormatRatioValue(double numerator, double denominator)
{
    if (numerator <= 0.0 || denominator <= 0.0)
        return "N/A";

    std::ostringstream out;
    out << std::fixed << std::setprecision(2)
        << (numerator / denominator) << 'x';
    return out.str();
}

} // namespace ExternalFairness

inline void StageBegin(std::size_t stage, std::size_t total, const char* label)
{
    std::cout << "    [stage " << stage << '/' << total << "] "
              << label << "..." << std::flush;
}

inline void StageEnd(bool ok = true)
{
    std::cout << (ok ? " done\n" : " FAIL\n");
}

inline void SweepSampleProgress(
    const char* kind,
    std::size_t current,
    std::size_t total,
    std::size_t sample,
    std::size_t sample_total)
{
    (void)sample_total;
    if (sample == 0u)
    {
        std::cout << "    [" << std::setw(2) << current << '/' << total
                  << ' ' << kind << "] samples:";
    }
    std::cout << ' ' << (sample + 1u) << std::flush;
}

inline void SweepSampleDone()
{
    std::cout << "  done\n";
}

template <typename Backend>
bool InitializeStructuralBackend(Backend& backend, const BenchmarkCase& config)
{
    // Tests 2-3 measure only topology/concurrency. Keep one minimal 64-bit
    // logical payload word rather than letting unused payload regions distort
    // cache footprint differently between the two storage engines.
    constexpr std::size_t STRUCTURAL_PAYLOAD_WORDS = 1u;
    if (!InitializeBackend(
        backend, config, STRUCTURAL_PAYLOAD_WORDS, true)) return false;

    if constexpr (requires { backend.PrimePayloadStorage(); })
        return backend.PrimePayloadStorage();
    return true;
}

template <typename Backend>
bool BuildMutationBackendFair(
    Backend& backend,
    const MutationScenario& scenario,
    std::size_t writer_count)
{
    if (writer_count == 0u || writer_count > scenario.MaxWriters ||
        !InitializeStructuralBackend(backend, scenario.Config)) return false;

    for (std::size_t writer = 0u; writer < writer_count; ++writer)
    {
        const std::size_t child = scenario.Child(writer);
        if (!backend.AddParent(
                scenario.InitialH(writer), child, Axis::HORIZONTAL) ||
            !backend.AddParent(
                scenario.InitialV(writer), child, Axis::VERTICAL))
            return false;
    }
    return true;
}

template <typename Backend>
bool BuildReaderBackendFair(Backend& backend, const ReaderScenario& scenario)
{
    if (!InitializeStructuralBackend(backend, scenario.Config)) return false;
    for (const WriterSpec& writer : scenario.Writers)
    {
        if (!backend.AddParent(
            writer.InitialParent, writer.Child, writer.RelationAxis)) return false;
    }
    return true;
}

struct ComparisonTiming { double Sortledton = 0.0, Fabric = 0.0; };

template <class LiveFn, class FabricFn>
ComparisonTiming MeasureComparison(LiveFn&& live, FabricFn&& fabric)
{
    std::array<double, ConcurrencyConfig::MEASURED_RUNS> l{}, f{};
    for (std::size_t run = 0; run < l.size(); ++run)
        if ((run & 1u) == 0u) { l[run] = live(); f[run] = fabric(); }
        else { f[run] = fabric(); l[run] = live(); }
    return {Median(l), Median(f)};
}

inline void PrintComparisonRow(const char* name, const ComparisonTiming& t)
{
    std::cout << "  " << std::left << std::setw(27) << name
        << " S=" << std::right << std::setw(10) << std::fixed
        << std::setprecision(2) << t.Sortledton << " ns/op"
        << "  Fabric=" << std::setw(10) << t.Fabric << " ns/op"
        << "  cost(S/F)=" << std::setw(8)
        << Ratio(t.Sortledton, t.Fabric) << "x\n";
}


namespace Test01
{
using namespace BenchmarkCore;

// TestKit's 100,000 replacement pairs were appropriate for the very cheap
// in-process baseline, but with a transactional graph they expand to
// 800,000 committed write transactions per axis across four measured samples.
// Keep the same operation count for both backends while bounding each measured
// sample to a long-enough, publication-useful interval. This is intentionally
// local to the external comparison and does not alter TestKit.
constexpr std::uint32_t TEST1_REPLACE_PAIRS_PER_SAMPLE = 5'000u;
constexpr std::uint64_t TEST1_REPLACE_OPS_PER_SAMPLE =
    static_cast<std::uint64_t>(TEST1_REPLACE_PAIRS_PER_SAMPLE) * 2u;


inline bool BuildGenericTest1MutationContext(
    SortledtonNativeBackend& backend,
    const BenchmarkCase& config)
{
    if (!backend.Initialize(
            config.NodeCount, TEST1_PAYLOAD_WORDS,
            config.ParentCapacity, true) ||
        !backend.PrimePayloadStorage())
        return false;

    for (std::size_t child = 0u; child < config.NodeCount; ++child)
    {
        const std::size_t count = std::min<std::size_t>(
            config.ParentCapacity, child);
        for (std::size_t ordinal = 0u; ordinal < count; ++ordinal)
        {
            const std::size_t h_parent = child - 1u - ordinal;
            const std::size_t v_parent = ordinal;
            if (!backend.AddParent(h_parent, child, Axis::HORIZONTAL, 1u) ||
                !backend.AddParent(v_parent, child, Axis::VERTICAL, 1u))
                return false;
        }
    }
    return backend.Healthy();
}

struct ThreeWayReplacementTiming
{
    double Native = 0.0;
    double Bidir = 0.0;
    double Fabric = 0.0;
};

template <class NativeFn, class BidirFn, class FabricFn>
ThreeWayReplacementTiming MeasureThreeWayReplacement(
    NativeFn&& native,
    BidirFn&& bidir,
    FabricFn&& fabric)
{
    std::array<double, ConcurrencyConfig::MEASURED_RUNS> n{}, b{}, f{};
    for (std::size_t run = 0u; run < ConcurrencyConfig::MEASURED_RUNS; ++run)
    {
        switch (run % 4u)
        {
        case 0u: n[run] = native(); b[run] = bidir();  f[run] = fabric(); break;
        case 1u: b[run] = bidir();  f[run] = fabric(); n[run] = native(); break;
        case 2u: f[run] = fabric(); n[run] = native(); b[run] = bidir();  break;
        default:n[run] = native(); f[run] = fabric(); b[run] = bidir();  break;
        }
    }
    return {Median(n), Median(b), Median(f)};
}

inline void PrintThreeWayReplacementRow(
    const char* name,
    const ThreeWayReplacementTiming& timing)
{
    std::cout << "  " << std::left << std::setw(27) << name << '\n'
        << "      S-native=" << std::right << std::setw(10) << std::fixed
        << std::setprecision(2) << timing.Native << " ns/op"
        << "  S-bidir=" << std::setw(10) << timing.Bidir << " ns/op"
        << "  Fabric=" << std::setw(10) << timing.Fabric << " ns/op\n"
        << "      cost ratios: S-native/Fabric="
        << Ratio(timing.Native, timing.Fabric) << "x  S-bidir/Fabric="
        << Ratio(timing.Bidir, timing.Fabric) << "x\n";
}

template <class Backend>
bool VerifyPayload(Backend& backend, const BenchmarkCase& config)
{
    for (std::size_t node = 0; node < config.NodeCount; ++node)
        for (std::uint32_t word = 0; word < TEST1_PAYLOAD_WORDS; ++word)
        {
            std::uint64_t value = 0;
            const auto expected =
                (static_cast<std::uint64_t>(node + 1u) << 32u) ^
                static_cast<std::uint64_t>(word + 1u);
            if (!backend.LoadPayload(node, word, value, false) ||
                value != expected) return false;
        }
    return true;
}

inline bool RunScenario(const BenchmarkCase& config, std::size_t case_index)
{
    constexpr std::size_t STAGE_COUNT = 13u;
    std::cout
        << "\n  CASE " << case_index << "/4"
        << "  N=" << config.NodeCount
        << "  K=" << static_cast<unsigned>(config.ParentCapacity) << '\n';

    const std::uint64_t edge_count = EdgeCountPerAxis(
        config.NodeCount, config.ParentCapacity);
    const std::uint64_t parent_calls =
        config.NodeCount * static_cast<std::uint64_t>(config.ParentCapacity);
    const std::uint64_t reverse_calls = edge_count + config.NodeCount;
    const std::uint64_t payload_calls =
        config.NodeCount * TEST1_PAYLOAD_WORDS;

    bool construction_ok = true;
    StageBegin(1u, STAGE_COUNT, "measured incremental construction");
    const auto construction = MeasureComparison(
        [&]()
        {
            return MeasureNsPerOperation(1u, [&]() -> std::uint64_t
            {
                SortledtonBidirBackend backend{};
                const bool ok = BuildFullTest1Graph(backend, config);
                construction_ok = construction_ok && ok;
                return ok ? static_cast<std::uint64_t>(config.NodeCount) : 0u;
            });
        },
        [&]()
        {
            return MeasureNsPerOperation(1u, [&]() -> std::uint64_t
            {
                RuntimeAPCFabricBackend backend{};
                const bool ok = BuildFullTest1Graph(backend, config);
                construction_ok = construction_ok && ok;
                return ok ? backend.ApproxStorageBytes() : 0u;
            });
        });
    StageEnd(construction_ok);
    if (!construction_ok) return false;
    PrintComparisonRow("construction / graph", construction);

    StageBegin(2u, STAGE_COUNT, "independent benchmark graph build");
    SortledtonNativeBackend native_backend{};
    SortledtonBidirBackend bidir_backend{};
    RuntimeAPCFabricBackend fabric_backend{};
    const bool persistent_build =
        BuildGenericTest1MutationContext(native_backend, config) &&
        BuildFullTest1Graph(bidir_backend, config) &&
        BuildFullTest1Graph(fabric_backend, config);
    StageEnd(persistent_build);
    if (!persistent_build) return false;

    const std::uint64_t native_topology_edges = edge_count * 2u;
    const std::uint64_t bidir_topology_edges = edge_count * 4u;
    const std::uint64_t payload_edges =
        static_cast<std::uint64_t>(config.NodeCount) * TEST1_PAYLOAD_WORDS;

    std::cout
        << "    logical edges/axis=" << edge_count
        << "  payload/node=" << TEST1_PAYLOAD_WORDS * sizeof(std::uint64_t)
        << " B\n"
        << "    [representation footprint]\n"
        << "      S-native topology edges : " << native_topology_edges
        << "  (H+V, one directed edge/relation)\n"
        << "      S-bidir topology edges  : " << bidir_topology_edges
        << "  (H+V, two directed edges/relation)\n"
        << "      Sortledton payload edges: " << payload_edges
        << "  (8-byte property/edge; separate graph)\n"
        << "      Sortledton allocated B  : N/A"
        << "  (no comparable public byte counter used by this adapter)\n"
        << "      Fabric slab bytes       : "
        << fabric_backend.ApproxStorageBytes() << "\n";

    StageBegin(3u, STAGE_COUNT, "pre-benchmark topology + payload verification");
    const bool live_proof = bidir_backend.VerifyFullTest1Graph(config);
    const GraphProof fabric_proof = ProveRuntimeCombinedDAG(fabric_backend, config);
    const bool payload_ok = bidir_backend.VerifyPayloadPattern() &&
        VerifyPayload(fabric_backend, config);
    bool ok = live_proof && fabric_proof.Passed() && payload_ok &&
        bidir_backend.Healthy();
    StageEnd(ok);
    if (!ok) return false;

    auto parent_scan = [&](auto& backend, Axis axis)
    {
        const std::uint32_t rounds = RoundsForTarget(
            TARGET_TRAVERSAL_CALLS, parent_calls);
        return MeasureNsPerOperation(parent_calls * rounds, [&]()
        {
            if constexpr (requires { backend.BenchmarkParentScan(axis, rounds); })
            {
                return backend.BenchmarkParentScan(axis, rounds);
            }
            else
            {
                std::uint64_t checksum = 0u;
                for (std::uint32_t r = 0u; r < rounds; ++r)
                    for (std::size_t child = 0u; child < config.NodeCount; ++child)
                        for (std::uint8_t ordinal = 0u;
                             ordinal < config.ParentCapacity; ++ordinal)
                        {
                            const BenchmarkReadResult read = BenchmarkFindParentCall(
                                backend, child, axis, ordinal, 1u);
                            checksum += read.Locator +
                                static_cast<std::uint64_t>(read.Outcome);
                        }
                return checksum;
            }
        });
    };

    auto reverse_scan = [&](auto& backend, Axis axis, bool payload)
    {
        const std::uint32_t rounds = RoundsForTarget(
            payload ? TARGET_GRAPH_PAYLOAD_CALLS : TARGET_TRAVERSAL_CALLS,
            reverse_calls);
        return MeasureNsPerOperation(reverse_calls * rounds, [&]()
        {
            if constexpr (requires {
                backend.BenchmarkReverseScan(axis, payload, rounds);
            })
            {
                return backend.BenchmarkReverseScan(axis, payload, rounds);
            }
            else
            {
                std::uint64_t checksum = 0u;
                for (std::uint32_t r = 0u; r < rounds; ++r)
                {
                    for (std::size_t parent = 0u; parent < config.NodeCount; ++parent)
                    {
                        BenchmarkReadResult read = BenchmarkFindFirstChildCall(
                            backend, parent, axis, 1u);
                        while (read.IsFound())
                        {
                            checksum += read.Locator;
                            if (payload && read.HasNodeHint())
                            {
                                std::uint64_t value = 0u;
                                const std::uint32_t word = static_cast<std::uint32_t>(
                                    read.NodeHint % TEST1_PAYLOAD_WORDS);
                                if (!backend.LoadPayload(
                                    read.NodeHint, word, value, false))
                                    checksum ^= UINT64_MAX;
                                checksum ^= value;
                            }
                            read = BenchmarkFindNextChildCall(
                                backend, parent, axis, read.Locator, 1u);
                        }
                        checksum += static_cast<std::uint64_t>(read.Outcome);
                    }
                }
                return checksum;
            }
        });
    };

    auto payload_scan = [&](auto& backend, bool atomic)
    {
        const std::uint32_t rounds = RoundsForTarget(
            TARGET_PAYLOAD_CALLS, payload_calls);
        return MeasureNsPerOperation(payload_calls * rounds, [&]()
        {
            if constexpr (requires { backend.BenchmarkPayloadScan(atomic, rounds); })
            {
                return backend.BenchmarkPayloadScan(atomic, rounds);
            }
            else
            {
                std::uint64_t checksum = 0u;
                for (std::uint32_t r = 0u; r < rounds; ++r)
                    for (std::size_t node = 0u; node < config.NodeCount; ++node)
                        for (std::uint32_t word = 0u;
                             word < TEST1_PAYLOAD_WORDS; ++word)
                        {
                            std::uint64_t value = 0u;
                            if (!backend.LoadPayload(node, word, value, atomic))
                                checksum ^= UINT64_MAX;
                            checksum += value;
                        }
                return checksum;
            }
        });
    };

    bool all_replacements_succeeded = true;
    auto replace_scan = [&](auto& backend, Axis axis)
    {
        const std::size_t child =
            static_cast<std::size_t>(config.ParentCapacity) + 1u;
        const std::size_t old_parent =
            axis == Axis::HORIZONTAL ? child - 1u : 0u;
        const std::size_t alternate = axis == Axis::HORIZONTAL
            ? child - static_cast<std::size_t>(config.ParentCapacity) - 1u
            : static_cast<std::size_t>(config.ParentCapacity);

        return MeasureNsPerOperation(
            TEST1_REPLACE_OPS_PER_SAMPLE,
            [&]()
            {
                std::uint64_t checksum = 0u;
                for (std::uint32_t r = 0u;
                     r < TEST1_REPLACE_PAIRS_PER_SAMPLE; ++r)
                {
                    // Test 1 is single-threaded: one attempt per logical
                    // replacement gives both backends identical semantics and
                    // avoids charging SuperNova's internal retry budget against
                    // Sortledton, whose adapter deliberately performs one txn.
                    const bool a = backend.ReplaceParent(
                        old_parent, alternate, child, axis, 1u);
                    const bool b = backend.ReplaceParent(
                        alternate, old_parent, child, axis, 1u);
                    all_replacements_succeeded =
                        all_replacements_succeeded && a && b;
                    checksum += static_cast<std::uint64_t>(a) +
                        static_cast<std::uint64_t>(b);
                }
                return checksum;
            });
    };

    StageBegin(4u, STAGE_COUNT, "H parent lookup bulk scan");
    const ComparisonTiming h_parent = MeasureComparison(
        [&] { return parent_scan(bidir_backend, Axis::HORIZONTAL); },
        [&] { return parent_scan(fabric_backend, Axis::HORIZONTAL); });
    StageEnd(bidir_backend.Healthy());
    PrintComparisonRow("H parent scan", h_parent);

    StageBegin(5u, STAGE_COUNT, "V parent lookup bulk scan");
    const ComparisonTiming v_parent = MeasureComparison(
        [&] { return parent_scan(bidir_backend, Axis::VERTICAL); },
        [&] { return parent_scan(fabric_backend, Axis::VERTICAL); });
    StageEnd(bidir_backend.Healthy());
    PrintComparisonRow("V parent scan", v_parent);

    StageBegin(6u, STAGE_COUNT, "H bidirectional reverse-child traversal");
    const ComparisonTiming h_reverse = MeasureComparison(
        [&] { return reverse_scan(bidir_backend, Axis::HORIZONTAL, false); },
        [&] { return reverse_scan(fabric_backend, Axis::HORIZONTAL, false); });
    StageEnd(bidir_backend.Healthy());
    PrintComparisonRow("H reverse-child scan", h_reverse);

    StageBegin(7u, STAGE_COUNT, "V bidirectional reverse-child traversal");
    const ComparisonTiming v_reverse = MeasureComparison(
        [&] { return reverse_scan(bidir_backend, Axis::VERTICAL, false); },
        [&] { return reverse_scan(fabric_backend, Axis::VERTICAL, false); });
    StageEnd(bidir_backend.Healthy());
    PrintComparisonRow("V reverse-child scan", v_reverse);

    StageBegin(8u, STAGE_COUNT, "sequential payload scan: Sortledton weighted-edge payload vs Fabric direct");
    const ComparisonTiming direct = MeasureComparison(
        [&] { return payload_scan(bidir_backend, false); },
        [&] { return payload_scan(fabric_backend, false); });
    StageEnd(bidir_backend.Healthy());
    PrintComparisonRow("payload edges/direct", direct);

    StageBegin(9u, STAGE_COUNT, "sequential payload scan: Sortledton weighted-edge payload vs Fabric atomic");
    const ComparisonTiming atomic = MeasureComparison(
        [&] { return payload_scan(bidir_backend, true); },
        [&] { return payload_scan(fabric_backend, true); });
    StageEnd(bidir_backend.Healthy());
    PrintComparisonRow("payload edges/atomic", atomic);

    StageBegin(10u, STAGE_COUNT, "H child traversal + payload");
    const ComparisonTiming graph_payload = MeasureComparison(
        [&] { return reverse_scan(bidir_backend, Axis::HORIZONTAL, true); },
        [&] { return reverse_scan(fabric_backend, Axis::HORIZONTAL, true); });
    StageEnd(bidir_backend.Healthy());
    PrintComparisonRow("H child + payload", graph_payload);

    std::cout << "    replacement workload="
              << TEST1_REPLACE_OPS_PER_SAMPLE
              << " logical replacements/sample x "
              << ConcurrencyConfig::MEASURED_RUNS << " samples/backend\n";

    StageBegin(11u, STAGE_COUNT, "H parent replacement: native / bidir / Fabric");
    const ThreeWayReplacementTiming h_replace = MeasureThreeWayReplacement(
        [&] { return replace_scan(native_backend, Axis::HORIZONTAL); },
        [&] { return replace_scan(bidir_backend, Axis::HORIZONTAL); },
        [&] { return replace_scan(fabric_backend, Axis::HORIZONTAL); });
    StageEnd(all_replacements_succeeded && native_backend.Healthy() && bidir_backend.Healthy());
    if (!all_replacements_succeeded || !native_backend.Healthy() ||
        !bidir_backend.Healthy()) return false;
    PrintThreeWayReplacementRow("H parent replace", h_replace);

    StageBegin(12u, STAGE_COUNT, "V parent replacement: native / bidir / Fabric");
    const ThreeWayReplacementTiming v_replace = MeasureThreeWayReplacement(
        [&] { return replace_scan(native_backend, Axis::VERTICAL); },
        [&] { return replace_scan(bidir_backend, Axis::VERTICAL); },
        [&] { return replace_scan(fabric_backend, Axis::VERTICAL); });
    StageEnd(all_replacements_succeeded && native_backend.Healthy() && bidir_backend.Healthy());
    if (!all_replacements_succeeded || !native_backend.Healthy() ||
        !bidir_backend.Healthy()) return false;
    PrintThreeWayReplacementRow("V parent replace", v_replace);

    StageBegin(13u, STAGE_COUNT, "post-mutation topology verification");
    const bool live_after = bidir_backend.VerifyFullTest1Graph(config);
    const GraphProof fabric_after = ProveRuntimeCombinedDAG(fabric_backend, config);
    const bool native_after = native_backend.VerifyFullTest1Graph(config);
    ok = ok && all_replacements_succeeded && live_after && native_after &&
        fabric_after.Passed() && native_backend.Healthy() && bidir_backend.Healthy();
    StageEnd(ok);
    std::cout << "    integrity=" << (ok ? "PASS" : "FAIL") << '\n';
    return ok;
}

inline Result Run(const std::array<BenchmarkCase, 4u>& cases)
{
    Banner("TEST 1 - SCALED FAIR BIDIRECTIONAL DAG / 1 KiB PAYLOAD COMPARISON");
    std::cout
        << "Official Sortledton transactional structures vs single-region SuperNova Fabric.\n"
        << "Each case uses the same logical N, K and fully populated H/V bounded DAG.\n"
        << "S-bidir uses two directed Sortledton edges/logical relation. Because this\n"
        << "Sortledton API has no Fabric-equivalent vertex payload row, 1 KiB/node is\n"
        << "encoded as 128 weighted payload edges in a separate Sortledton graph.\n"
        << "Four measured samples are taken per row with backend order alternated.\n"
        << "Bulk Sortledton read rows retain one snapshot/iterator per scan round.\n"
        << "Read/traversal rows use the bidirectional contract; replacement rows also\n"
        << "show S-native one-edge transactions as a non-equivalent lower bound.\n";

    bool ok = true;
    for (std::size_t i = 0u; i < cases.size(); ++i)
        ok = RunScenario(cases[i], i + 1u) && ok;

    std::cout << "\nTEST 1 OVERALL: " << (ok ? "PASS" : "FAIL") << '\n';
    return ok ? Result::PASS : Result::FAIL;
}
} // namespace Test01

namespace Test02
{
using namespace BenchmarkCore;
using namespace ExternalFairness;

struct SampleTriplet
{
    TimedMutationResult Native{};
    TimedMutationResult Bidir{};
    TimedMutationResult Fabric{};
};

inline bool BuildThreeMutationBackends(
    SortledtonNativeBackend& native,
    SortledtonBidirBackend& bidir,
    RuntimeAPCFabricBackend& fabric,
    const MutationScenario& scenario,
    std::size_t writer_count)
{
    return
        BuildMutationBackendFair(native, scenario, writer_count) &&
        BuildMutationBackendFair(bidir, scenario, writer_count) &&
        BuildMutationBackendFair(fabric, scenario, writer_count);
}

inline bool RunCase(
    const BenchmarkCase& config,
    MutationLocality locality,
    std::size_t max_writers,
    std::size_t case_index,
    Duration measurement)
{
    const auto maybe_scenario =
        MakeMutationScenario(config, locality, max_writers);
    if (!maybe_scenario.has_value())
        return false;

    const MutationScenario scenario = maybe_scenario.value();
    const MutationSchedule schedule = BuildMutationSchedule(scenario);

    std::cout
        << "\n  CASE " << case_index << "/4"
        << " | N=" << config.NodeCount
        << " | K=" << static_cast<unsigned>(config.ParentCapacity)
        << " | parent-pool=" << scenario.ParentCount
        << " | " << measurement.count() << " ms/backend/sample\n";

    bool all_ok = true;

    for (std::size_t writer_count = 1u;
         writer_count <= max_writers;
         ++writer_count)
    {
        std::array<double, ConcurrencyConfig::MEASURED_RUNS>
            native_mops{}, bidir_mops{}, fabric_mops{};
        std::array<double, ConcurrencyConfig::MEASURED_RUNS>
            native_retry{}, bidir_retry{}, fabric_retry{};
        std::array<double, ConcurrencyConfig::MEASURED_RUNS>
            native_limit{}, bidir_limit{}, fabric_limit{};

        std::size_t native_bad = 0u;
        std::size_t bidir_bad = 0u;
        std::size_t fabric_bad = 0u;

        for (std::size_t run = 0u;
             run < ConcurrencyConfig::MEASURED_RUNS;
             ++run)
        {
            SweepSampleProgress(
                "writers", writer_count, max_writers, run,
                ConcurrencyConfig::MEASURED_RUNS);

            SortledtonNativeBackend native{};
            SortledtonBidirBackend bidir{};
            RuntimeAPCFabricBackend fabric{};

            if (!BuildThreeMutationBackends(
                    native, bidir, fabric, scenario, writer_count))
            {
                std::cout << "  SETUP FAIL\n";
                return false;
            }

            SampleTriplet result{};

            auto run_native = [&]()
            {
                result.Native = RunTimedMutationWorkers(
                    native, scenario, schedule,
                    writer_count, measurement, false);
            };

            auto run_bidir = [&]()
            {
                result.Bidir = RunTimedMutationWorkers(
                    bidir, scenario, schedule,
                    writer_count, measurement, true);
            };

            auto run_fabric = [&]()
            {
                result.Fabric = RunTimedMutationWorkers(
                    fabric, scenario, schedule,
                    writer_count, measurement, true);
            };

            // Rotate order to reduce systematic thermal/cache ordering bias.
            switch (run % 4u)
            {
            case 0u: run_native(); run_bidir();  run_fabric(); break;
            case 1u: run_bidir();  run_fabric(); run_native(); break;
            case 2u: run_fabric(); run_native(); run_bidir();  break;
            default: run_native(); run_fabric(); run_bidir();  break;
            }

            native_bad += static_cast<std::size_t>(!result.Native.Ok);
            bidir_bad += static_cast<std::size_t>(!result.Bidir.Ok);
            fabric_bad += static_cast<std::size_t>(!result.Fabric.Ok);

            native_mops[run] = result.Native.Mops();
            bidir_mops[run] = result.Bidir.Mops();
            fabric_mops[run] = result.Fabric.Mops();

            native_retry[run] = result.Native.RetryRate();
            bidir_retry[run] = result.Bidir.RetryRate();
            fabric_retry[run] = result.Fabric.RetryRate();

            native_limit[run] = result.Native.RetryLimitRate();
            bidir_limit[run] = result.Bidir.RetryLimitRate();
            fabric_limit[run] = result.Fabric.RetryLimitRate();
        }

        SweepSampleDone();

        const bool native_ok = native_bad == 0u;
        const bool bidir_ok = bidir_bad == 0u;
        const bool fabric_ok = fabric_bad == 0u;
        const bool point_ok = native_ok && bidir_ok && fabric_ok;

        const double n_mops = native_ok ? Median(native_mops) : 0.0;
        const double b_mops = bidir_ok ? Median(bidir_mops) : 0.0;
        const double f_mops = fabric_ok ? Median(fabric_mops) : 0.0;

        std::cout
            << "      throughput | N=" << std::setw(12)
            << (native_ok ? FormatThroughput(n_mops) : "N/A")
            << " | B=" << std::setw(12)
            << (bidir_ok ? FormatThroughput(b_mops) : "N/A")
            << " | F=" << std::setw(12)
            << (fabric_ok ? FormatThroughput(f_mops) : "N/A")
            << '\n';

        std::cout
            << "      contention | retry/op N/B/F="
            << FormatRate(native_ok ? Median(native_retry) : 0.0) << '/'
            << FormatRate(bidir_ok ? Median(bidir_retry) : 0.0) << '/'
            << FormatRate(fabric_ok ? Median(fabric_retry) : 0.0)
            << " | retry-limit/op="
            << FormatRate(native_ok ? Median(native_limit) : 0.0) << '/'
            << FormatRate(bidir_ok ? Median(bidir_limit) : 0.0) << '/'
            << FormatRate(fabric_ok ? Median(fabric_limit) : 0.0)
            << '\n';

        if (point_ok)
        {
            std::cout
                << "      ratio      | throughput F/N="
                << FormatRatioValue(f_mops, n_mops)
                << " | F/B=" << FormatRatioValue(f_mops, b_mops)
                << " | PASS\n";
        }
        else
        {
            std::cout
                << "      status     | FAIL | invalid samples N/B/F="
                << native_bad << '/' << bidir_bad << '/' << fabric_bad
                << " of " << ConcurrencyConfig::MEASURED_RUNS << '\n';
        }

        all_ok = all_ok && point_ok;
    }

    return all_ok;
}

inline Result Run(
    const std::array<BenchmarkCase, 4u>& cases,
    std::size_t max_writers,
    Duration measurement)
{
    Banner("TEST 2A - HOTSPOT STRUCTURAL MUTATION / FIXED-DURATION");
    std::cout
        << "Equal wall-clock interval for all backends. N=S-native (one-way directed graph),\n"
        << "B=S-bidir (end-to-end bidirectional), F=SuperNova Fabric. Throughput is\n"
        << "successful replacements/s. retry-limit/op reports 4096-attempt budget hits;\n"
        << "these are contention/progress events, not integrity failures.\n";

    bool hotspot_ok = true;
    for (std::size_t i = 0u; i < cases.size(); ++i)
        hotspot_ok = RunCase(
            cases[i], MutationLocality::HOTSPOT, max_writers, i + 1u,
            measurement) && hotspot_ok;
    std::cout << "\nTEST 2A OVERALL: " << (hotspot_ok ? "PASS" : "FAIL") << '\n';

    Banner("TEST 2B - DISTRIBUTED STRUCTURAL MUTATION / FIXED-DURATION");
    std::cout
        << "Distributed-parent version of the same equal-time test. N/B/F use the same\n"
        << "deterministic schedule; throughput and contention rates are reported separately.\n";

    bool distributed_ok = true;
    for (std::size_t i = 0u; i < cases.size(); ++i)
        distributed_ok = RunCase(
            cases[i], MutationLocality::DISTRIBUTED, max_writers, i + 1u,
            measurement) && distributed_ok;
    std::cout << "\nTEST 2B OVERALL: " << (distributed_ok ? "PASS" : "FAIL") << '\n';

    const bool ok = hotspot_ok && distributed_ok;
    std::cout << "\nTEST 2 OVERALL: " << (ok ? "PASS" : "FAIL") << '\n';
    return ok ? Result::PASS : Result::FAIL;
}
} // namespace Test02

namespace Test03
{
using namespace BenchmarkCore;
using namespace ExternalFairness;

struct ReaderTriplet
{
    TimedReaderResult Native{};
    TimedReaderResult Bidir{};
    TimedReaderResult Fabric{};
};

inline bool BuildThreeReaderBackends(
    SortledtonNativeBackend& native,
    SortledtonBidirBackend& bidir,
    RuntimeAPCFabricBackend& fabric,
    const ReaderScenario& scenario)
{
    return
        BuildReaderBackendFair(native, scenario) &&
        BuildReaderBackendFair(bidir, scenario) &&
        BuildReaderBackendFair(fabric, scenario);
}

inline bool RunCase(
    const BenchmarkCase& config,
    bool distributed,
    std::size_t max_readers,
    std::size_t case_index,
    Duration measurement)
{
    const auto maybe_scenario = MakeReaderScenario(config, distributed);
    if (!maybe_scenario.has_value())
        return false;

    const ReaderScenario scenario = maybe_scenario.value();
    const ParentSchedule schedule = BuildParentSchedule(scenario);

    std::cout
        << "\n  CASE " << case_index << "/4"
        << " | N=" << config.NodeCount
        << " | K=" << static_cast<unsigned>(config.ParentCapacity)
        << " | parent-pool=" << scenario.ParentCount
        << " | " << measurement.count() << " ms/backend/sample\n";

    bool all_ok = true;

    for (std::size_t reader_count = 1u;
         reader_count <= max_readers;
         ++reader_count)
    {
        std::array<double, ConcurrencyConfig::MEASURED_RUNS>
            n_read{}, b_read{}, f_read{};
        std::array<double, ConcurrencyConfig::MEASURED_RUNS>
            n_write{}, b_write{}, f_write{};
        std::array<double, ConcurrencyConfig::MEASURED_RUNS>
            n_retry{}, b_retry{}, f_retry{};
        std::array<double, ConcurrencyConfig::MEASURED_RUNS>
            n_read_limit{}, b_read_limit{}, f_read_limit{};
        std::array<double, ConcurrencyConfig::MEASURED_RUNS>
            n_write_limit{}, b_write_limit{}, f_write_limit{};

        std::size_t native_bad = 0u;
        std::size_t bidir_bad = 0u;
        std::size_t fabric_bad = 0u;

        for (std::size_t run = 0u;
             run < ConcurrencyConfig::MEASURED_RUNS;
             ++run)
        {
            SweepSampleProgress(
                "readers", reader_count, max_readers, run,
                ConcurrencyConfig::MEASURED_RUNS);

            SortledtonNativeBackend native{};
            SortledtonBidirBackend bidir{};
            RuntimeAPCFabricBackend fabric{};

            if (!BuildThreeReaderBackends(native, bidir, fabric, scenario))
            {
                std::cout << "  SETUP FAIL\n";
                return false;
            }

            ReaderTriplet result{};

            auto run_native = [&]()
            {
                result.Native = RunTimedReadersWithWriters(
                    native, scenario, schedule,
                    reader_count, measurement, false);
            };

            auto run_bidir = [&]()
            {
                result.Bidir = RunTimedReadersWithWriters(
                    bidir, scenario, schedule,
                    reader_count, measurement, true);
            };

            auto run_fabric = [&]()
            {
                result.Fabric = RunTimedReadersWithWriters(
                    fabric, scenario, schedule,
                    reader_count, measurement, true);
            };

            switch (run % 4u)
            {
            case 0u: run_native(); run_bidir();  run_fabric(); break;
            case 1u: run_bidir();  run_fabric(); run_native(); break;
            case 2u: run_fabric(); run_native(); run_bidir();  break;
            default: run_native(); run_fabric(); run_bidir();  break;
            }

            native_bad += static_cast<std::size_t>(!result.Native.Ok);
            bidir_bad += static_cast<std::size_t>(!result.Bidir.Ok);
            fabric_bad += static_cast<std::size_t>(!result.Fabric.Ok);

            n_read[run] = result.Native.ReadMops();
            b_read[run] = result.Bidir.ReadMops();
            f_read[run] = result.Fabric.ReadMops();

            n_write[run] = result.Native.WriterMops();
            b_write[run] = result.Bidir.WriterMops();
            f_write[run] = result.Fabric.WriterMops();

            n_retry[run] = result.Native.RetryRate();
            b_retry[run] = result.Bidir.RetryRate();
            f_retry[run] = result.Fabric.RetryRate();

            n_read_limit[run] = result.Native.ReaderRetryLimitRate();
            b_read_limit[run] = result.Bidir.ReaderRetryLimitRate();
            f_read_limit[run] = result.Fabric.ReaderRetryLimitRate();

            n_write_limit[run] = result.Native.WriterRetryLimitRate();
            b_write_limit[run] = result.Bidir.WriterRetryLimitRate();
            f_write_limit[run] = result.Fabric.WriterRetryLimitRate();
        }

        SweepSampleDone();

        const bool native_ok = native_bad == 0u;
        const bool bidir_ok = bidir_bad == 0u;
        const bool fabric_ok = fabric_bad == 0u;
        const bool point_ok = native_ok && bidir_ok && fabric_ok;

        const double nr = native_ok ? Median(n_read) : 0.0;
        const double br = bidir_ok ? Median(b_read) : 0.0;
        const double fr = fabric_ok ? Median(f_read) : 0.0;

        const double nw = native_ok ? Median(n_write) : 0.0;
        const double bw = bidir_ok ? Median(b_write) : 0.0;
        const double fw = fabric_ok ? Median(f_write) : 0.0;

        std::cout
            << "      reads      | N=" << std::setw(12)
            << (native_ok ? FormatThroughput(nr) : "N/A")
            << " | B=" << std::setw(12)
            << (bidir_ok ? FormatThroughput(br) : "N/A")
            << " | F=" << std::setw(12)
            << (fabric_ok ? FormatThroughput(fr) : "N/A")
            << '\n';

        std::cout
            << "      writers    | N=" << std::setw(12)
            << (native_ok ? FormatThroughput(nw) : "N/A")
            << " | B=" << std::setw(12)
            << (bidir_ok ? FormatThroughput(bw) : "N/A")
            << " | F=" << std::setw(12)
            << (fabric_ok ? FormatThroughput(fw) : "N/A")
            << '\n';

        std::cout
            << "      contention | read-retry/op N/B/F="
            << FormatRate(native_ok ? Median(n_retry) : 0.0) << '/'
            << FormatRate(bidir_ok ? Median(b_retry) : 0.0) << '/'
            << FormatRate(fabric_ok ? Median(f_retry) : 0.0)
            << " | read-limit/op="
            << FormatRate(native_ok ? Median(n_read_limit) : 0.0) << '/'
            << FormatRate(bidir_ok ? Median(b_read_limit) : 0.0) << '/'
            << FormatRate(fabric_ok ? Median(f_read_limit) : 0.0)
            << " | write-limit/op="
            << FormatRate(native_ok ? Median(n_write_limit) : 0.0) << '/'
            << FormatRate(bidir_ok ? Median(b_write_limit) : 0.0) << '/'
            << FormatRate(fabric_ok ? Median(f_write_limit) : 0.0)
            << '\n';

        if (point_ok)
        {
            std::cout
                << "      ratio      | read throughput F/N="
                << FormatRatioValue(fr, nr)
                << " | F/B=" << FormatRatioValue(fr, br)
                << " | PASS\n";
        }
        else
        {
            std::cout
                << "      status     | FAIL | invalid samples N/B/F="
                << native_bad << '/' << bidir_bad << '/' << fabric_bad
                << " of " << ConcurrencyConfig::MEASURED_RUNS << '\n';
        }

        all_ok = all_ok && point_ok;
    }

    return all_ok;
}

inline Result Run(
    const std::array<BenchmarkCase, 4u>& cases,
    std::size_t max_readers,
    Duration measurement)
{
    Banner("TEST 3A - HOTSPOT STABLE READS + TWO WRITERS / FIXED-DURATION");
    std::cout
        << "Equal read/write interval for all backends. N=S-native, B=S-bidir, F=Fabric.\n"
        << "Both Sortledton modes use a fresh Sortledton snapshot/read; Fabric uses its public\n"
        << "sequence-validated stable read. Retry-limit events are reported, not hidden.\n";

    bool hotspot_ok = true;
    for (std::size_t i = 0u; i < cases.size(); ++i)
        hotspot_ok = RunCase(
            cases[i], false, max_readers, i + 1u, measurement) && hotspot_ok;
    std::cout << "\nTEST 3A OVERALL: " << (hotspot_ok ? "PASS" : "FAIL") << '\n';

    Banner("TEST 3B - DISTRIBUTED STABLE READS + TWO WRITERS / FIXED-DURATION");
    std::cout
        << "Two writers mutate separate child relations across up to 100 legal parents.\n"
        << "Reader sweep uses the same interval for N/B/F; read throughput, writer\n"
        << "throughput, retries, and retry-limit events are reported separately.\n";

    bool distributed_ok = true;
    for (std::size_t i = 0u; i < cases.size(); ++i)
        distributed_ok = RunCase(
            cases[i], true, max_readers, i + 1u, measurement) && distributed_ok;
    std::cout << "\nTEST 3B OVERALL: " << (distributed_ok ? "PASS" : "FAIL") << '\n';

    const bool ok = hotspot_ok && distributed_ok;
    std::cout << "\nTEST 3 OVERALL: " << (ok ? "PASS" : "FAIL") << '\n';
    return ok ? Result::PASS : Result::FAIL;
}
} // namespace Test03

class ScopedBenchmarkOutput final
{
public:
    explicit ScopedBenchmarkOutput(const std::filesystem::path& output_path)
        : File_(output_path, std::ios::out | std::ios::trunc)
    {
        if (!File_.is_open())
            return;

        PreviousBuffer_ = std::cout.rdbuf(File_.rdbuf());
        Active_ = true;
    }

    ScopedBenchmarkOutput(const ScopedBenchmarkOutput&) = delete;
    ScopedBenchmarkOutput& operator=(const ScopedBenchmarkOutput&) = delete;

    ~ScopedBenchmarkOutput() noexcept
    {
        if (!Active_)
            return;

        std::cout.flush();
        std::cout.rdbuf(PreviousBuffer_);
        File_.flush();
    }

    bool IsOpen() const noexcept
    {
        return File_.is_open();
    }

private:
    std::ofstream File_{};
    std::streambuf* PreviousBuffer_ = nullptr;
    bool Active_ = false;
};

inline int Run(std::size_t lower_node_count = 100u,
               std::size_t higher_node_count = 10'000u,
               std::size_t lower_parent_capacity = 4u,
               std::size_t higher_parent_capacity = 32u,
               std::size_t usable_thread_count = 18u,
               std::uint32_t measurement_ms = ExternalFairness::DEFAULT_MEASUREMENT_MS,
               const std::filesystem::path& result_path =
                   "SortledtonVsSuperNova_results.txt")
{
    ScopedBenchmarkOutput output(result_path);
    if (!output.IsOpen())
    {
        std::cerr
            << "SortledtonVsSuperNova: failed to open result file: "
            << result_path << '\n';
        return 1;
    }

    // Keep the terminal quiet. All std::cout output from this point, including
    // TestKit banners/environment text, is captured by result_path.
    PrintBenchmarkEnvironment();
    std::cout
        << "  result_file            : " << result_path.string() << "\n"
        << "  output_mode            : text file (terminal suppressed)\n";
    if (!ValidateRunArguments(lower_node_count, higher_node_count,
                              lower_parent_capacity, higher_parent_capacity,
                              usable_thread_count)) return 1;
    // TransactionManager registers the main thread at id 0 and permits 63
    // additional threads (two writers plus the reader sweep in Test 3).
    if (usable_thread_count > 63u)
    {
        std::cout << "  configuration error: usable_thread_count must be <= 63\n";
        return 1;
    }
    const auto cases = MakeBenchmarkCases(lower_node_count, higher_node_count,
        static_cast<std::uint8_t>(lower_parent_capacity),
        static_cast<std::uint8_t>(higher_parent_capacity));
    const auto workers = usable_thread_count - 2u;
    if (measurement_ms == 0u) return 1;
    const ExternalFairness::Duration measurement(measurement_ms);
    std::cout
        << "\nSortledton-native vs Sortledton-bidirectional vs SuperNova Fabric; "
        << "four measured samples/point; writer/reader sweep 1.." << workers
        << "; fixed interval=" << measurement_ms << " ms/system/sample\n"
        << "  N=S-native one-way transaction (lower bound / different contract)\n"
        << "  B=S-bidir end-to-end bidirectional transaction\n"
        << "  F=SuperNova Fabric native bidirectional relation transaction\n";
    const auto first = Test01::Run(cases);
    const auto second = Test02::Run(cases, workers, measurement);
    const auto third = Test03::Run(cases, workers, measurement);
    Banner("SORTLEDTON VS SUPERNOVA TESTS 1-3 SUMMARY");
    std::cout << "  Test 1: " << ResultName(first)
              << "\n  Test 2: " << ResultName(second)
              << "\n  Test 3: " << ResultName(third) << '\n';
    return first == Result::PASS && second == Result::PASS &&
           third == Result::PASS ? 0 : 1;
}

inline int RunAll(std::size_t lower_node_count = 100u,
                  std::size_t higher_node_count = 10'000u,
                  std::size_t lower_parent_capacity = 4u,
                  std::size_t higher_parent_capacity = 32u,
                  std::size_t usable_thread_count = 18u,
                  std::uint32_t measurement_ms = ExternalFairness::DEFAULT_MEASUREMENT_MS,
                  const std::filesystem::path& result_path =
                      "SortledtonVsSuperNova_results.txt")
{
    return Run(lower_node_count, higher_node_count, lower_parent_capacity,
               higher_parent_capacity, usable_thread_count, measurement_ms,
               result_path);
}
} // namespace SortledtonVsSuperNova
