#pragma once

// Place in a benchmark directory beside the LiveGraph and SuperNova repository
// directories (the same layout as the original include stub). Compile as C++20
// and link the official LiveGraph and SuperNova implementations.
// int main() { return LiveGraphVsSuperNova::Run(); }
// By default all benchmark output is written to:
//     LiveGraphVsSuperNova_results.txt
// in the process working directory. A custom output path may be passed as the
// final Run()/RunAll() argument.
//
// Three representations are distinguished deliberately:
//   (1) LiveGraph-native: one directed physical edge per logical parent relation.
//       This is a lower-bound/native transactional comparison and does NOT provide
//       Fabric's reverse-child adjacency contract.
//   (2) LiveGraph-bidirectional: two physical edges per logical relation, matching
//       Fabric's parent lookup + reverse-child traversal contract end to end.
//   (3) SuperNova Fabric: bounded bidirectional DAG relation rows in one relocatable slab.
// Test 1 reports full-contract bulk scans and separate storage footprints.
// Tests 2-3 give all three systems the same nominal measurement interval.
// Concurrent results are reported primarily as aggregate throughput. A retry-limit
// event means one logical operation consumed the external 4096-attempt budget;
// it is reported as a progress/contention metric and does NOT invalidate an
// otherwise healthy, integrity-correct sample. Contract/correctness/backend-fatal
// failures still invalidate the sample. Native one-way LiveGraph is a different
// contract and is labeled accordingly.

#include <stdexcept>
#include <string>
#include <string_view>
#include "../LiveGraph/bind/livegraph.hpp"
#include "../SuperNova/core/headers/NeuromorphicTimeSpace/VagueTemoraryPremativeFabric.hpp"
#include "../SuperNova/core/TestFiles/TestKit.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include <barrier>
#include <chrono>
#include <filesystem>
#include <thread>
#include <system_error>
#include <sstream>
#include <fstream>

#if defined(__linux__) || defined(__unix__)
#include <sys/stat.h>
#endif

namespace LiveGraphVsSuperNova
{
using namespace APCDAGTests;
using namespace APCDAGTests::BenchmarkCore;

class LiveGraphBackend
{
public:
    bool Initialize(std::size_t nodes, std::size_t words, std::uint8_t k,
                    bool single_payload_region = false)
    {
        (void)single_payload_region;
        if (nodes == 0 || nodes > UINT32_MAX || k == 0 ||
            k > ADS::COMPILED_MAX_DIRECT_PARENTS_PER_AXIS ||
            nodes > UINT32_MAX / k || words == 0 || words > UINT32_MAX)
            return false;
        try
        {
            Fatal_.store(false, std::memory_order_release);
            Graph_ = std::make_unique<lg::Graph>("", "", 1ull << 40,
                                                   static_cast<lg::vertex_t>(nodes + 1));
            auto tx = Graph_->begin_transaction();
            for (std::size_t node = 0; node < nodes; ++node)
                if (tx.new_vertex() != node) return false;
            tx.commit();
            NodeCount_ = nodes;
            Words_ = words;
            K_ = k;
            Staging_.assign(words, 0u);
            StageNode_ = StageWord_ = 0u;
            return true;
        }
        catch (const std::exception&) { Fatal_.store(true); return false; }
    }

    // Tests 2-3 are structural tests. SuperNova still owns a minimal payload
    // region because its runtime schema requires a region; give LiveGraph the
    // same logical one-word vertex property so neither side is an empty-shell
    // graph during the structural measurements. This setup is outside timing.
    bool PrimePayloadStorage() noexcept
    {
        if (!Graph_ || Words_ == 0u || Fatal_.load()) return false;
        try
        {
            std::vector<std::uint64_t> zeros(Words_, 0u);
            const std::string_view payload(
                reinterpret_cast<const char*>(zeros.data()),
                zeros.size() * sizeof(std::uint64_t));
            auto tx = Graph_->begin_transaction();
            for (std::size_t node = 0u; node < NodeCount_; ++node)
                tx.put_vertex(static_cast<lg::vertex_t>(node), payload);
            tx.commit();
            return true;
        }
        catch (const lg::Transaction::RollbackExcept&) { return false; }
        catch (const std::exception&) { Fatal_.store(true); return false; }
    }

    bool AddParent(std::size_t parent, std::size_t child, Axis axis,
                   std::uint32_t tries = DEFAULT_MAX_TRIES) noexcept
    {
        (void)tries;
        if (!Legal(parent, child) || Fatal_.load()) return false;
        try
        {
            auto tx = Graph_->begin_transaction();

            // One logical parent relation per (axis,parent,child).
            if (!tx.get_edge(parent, Front(axis), child).empty()) return false;

            std::optional<unsigned> free_ordinal{};
            for (unsigned ordinal = 0u; ordinal < K_; ++ordinal)
            {
                auto edges = tx.get_edges(child, Back(axis, ordinal));
                if (!edges.valid())
                {
                    free_ordinal = ordinal;
                    break;
                }

                const auto data = edges.edge_data();
                const auto existing_parent = static_cast<std::size_t>(edges.dst_id());
                if (!EdgeDataMatchesOrdinal(data, ordinal) ||
                    !Legal(existing_parent, child))
                {
                    Fatal_.store(true);
                    return false;
                }
                edges.next();
                if (edges.valid())
                {
                    // A reverse ordinal label is intentionally one-to-one.
                    Fatal_.store(true);
                    return false;
                }
            }
            if (!free_ordinal.has_value()) return false;

            const char encoded = static_cast<char>(free_ordinal.value() + 1u);
            const std::string_view data(&encoded, 1u);
            tx.put_edge(parent, Front(axis), child, data);
            tx.put_edge(child, Back(axis, free_ordinal.value()), parent, data);
            tx.commit();
            return true;
        }
        catch (const lg::Transaction::RollbackExcept&) { return false; }
        catch (const std::exception&) { Fatal_.store(true); return false; }
    }

    bool ReplaceParent(std::size_t old_parent, std::size_t new_parent,
                       std::size_t child, Axis axis,
                       std::uint32_t tries = DEFAULT_MAX_TRIES) noexcept
    {
        (void)tries; // TestKit::RetryReplace owns the common retry budget.
        if (!Legal(old_parent, child) || !Legal(new_parent, child) ||
            old_parent == new_parent || Fatal_.load()) return false;
        try
        {
            auto tx = Graph_->begin_transaction();
            const auto forward = tx.get_edge(old_parent, Front(axis), child);
            unsigned ordinal = 0u;
            if (!DecodeOrdinal(forward, ordinal))
            {
                Fatal_.store(true);
                return false;
            }

            const auto reverse = tx.get_edge(
                child, Back(axis, ordinal), old_parent);
            if (reverse != forward ||
                !tx.get_edge(new_parent, Front(axis), child).empty())
            {
                Fatal_.store(true);
                return false;
            }

            if (!tx.del_edge(old_parent, Front(axis), child) ||
                !tx.del_edge(child, Back(axis, ordinal), old_parent))
            {
                Fatal_.store(true);
                return false;
            }

            tx.put_edge(new_parent, Front(axis), child, forward);
            tx.put_edge(child, Back(axis, ordinal), new_parent, forward);
            tx.commit();
            return true;
        }
        catch (const lg::Transaction::RollbackExcept&) { return false; }
        catch (const std::exception&) { Fatal_.store(true); return false; }
    }

    ReadResult FindParent(std::size_t child, Axis axis, std::uint8_t ordinal,
                          std::uint32_t tries = 1u) noexcept
    {
        (void)tries;
        if (child >= NodeCount_ || ordinal >= K_ || Fatal_.load()) return {};
        try
        {
            auto tx = Graph_->begin_read_only_transaction();
            return ParentRead(tx, child, axis, ordinal);
        }
        catch (const std::exception&) { Fatal_.store(true); return {}; }
    }

    ReadResult StableFindParent(std::size_t child, Axis axis,
                                std::uint8_t ordinal,
                                std::uint32_t tries = 1u) noexcept
    {
        // Intentionally a fresh MVCC snapshot for every Test-3 logical read.
        return FindParent(child, axis, ordinal, tries);
    }

    ReadResult FindFirstChild(std::size_t parent, Axis axis,
                              std::uint32_t tries = 1u) noexcept
    { return ChildRead(parent, axis, ChildOp::First, 0u, tries); }
    ReadResult FindLastChild(std::size_t parent, Axis axis,
                             std::uint32_t tries = 1u) noexcept
    { return ChildRead(parent, axis, ChildOp::Last, 0u, tries); }
    ReadResult FindNextChild(std::size_t parent, Axis axis, std::uint32_t locator,
                             std::uint32_t tries = 1u) noexcept
    { return ChildRead(parent, axis, ChildOp::Next, locator, tries); }
    ReadResult FindPreviousChild(std::size_t parent, Axis axis, std::uint32_t locator,
                                 std::uint32_t tries = 1u) noexcept
    { return ChildRead(parent, axis, ChildOp::Previous, locator, tries); }

    bool StorePayload(std::size_t node, std::uint32_t word,
                      std::uint64_t value, bool atomic) noexcept
    {
        (void)atomic;
        // InitializePayload writes one node's words in order. Commit its
        // complete property once, avoiding 128 partial vertex transactions.
        if (node != StageNode_ || word != StageWord_ || node >= NodeCount_ ||
            word >= Words_ || Fatal_.load()) return false;
        Staging_[word] = value;
        if (++StageWord_ != Words_) return true;
        try
        {
            auto tx = Graph_->begin_transaction();
            tx.put_vertex(node, std::string_view(
                reinterpret_cast<const char*>(Staging_.data()),
                Staging_.size() * sizeof(std::uint64_t)));
            tx.commit();
            ++StageNode_;
            StageWord_ = 0u;
            return true;
        }
        catch (const lg::Transaction::RollbackExcept&) { return false; }
        catch (const std::exception&) { Fatal_.store(true); return false; }
    }

    bool LoadPayload(std::size_t node, std::uint32_t word,
                     std::uint64_t& value, bool atomic) noexcept
    {
        (void)atomic;
        if (node >= NodeCount_ || word >= Words_ || Fatal_.load()) return false;
        try
        {
            auto tx = Graph_->begin_read_only_transaction();
            return LoadPayloadFromTransaction(tx, node, word, value);
        }
        catch (const std::exception&) { Fatal_.store(true); return false; }
    }

    // Test 1 is explicitly a bulk scan. These methods use one LiveGraph snapshot
    // per complete scan round and retain the native edge iterator, rather than
    // charging transaction creation + rescan-from-beginning to every child/word.
    // Tests 2-3 do NOT use these methods.
    std::uint64_t BenchmarkParentScan(Axis axis, std::uint32_t rounds) noexcept
    {
        std::uint64_t checksum = 0u;
        if (Fatal_.load()) return checksum;
        try
        {
            for (std::uint32_t r = 0u; r < rounds; ++r)
            {
                auto tx = Graph_->begin_read_only_transaction();
                for (std::size_t child = 0u; child < NodeCount_; ++child)
                {
                    for (std::uint8_t ordinal = 0u; ordinal < K_; ++ordinal)
                    {
                        const ReadResult read = ParentRead(tx, child, axis, ordinal);
                        checksum += static_cast<std::uint64_t>(read.Locator) +
                            static_cast<std::uint64_t>(read.Outcome);
                    }
                }
            }
        }
        catch (const std::exception&) { Fatal_.store(true); }
        return checksum;
    }

    std::uint64_t BenchmarkReverseScan(
        Axis axis, bool payload, std::uint32_t rounds) noexcept
    {
        std::uint64_t checksum = 0u;
        if (Fatal_.load()) return checksum;
        try
        {
            for (std::uint32_t r = 0u; r < rounds; ++r)
            {
                auto tx = Graph_->begin_read_only_transaction();
                for (std::size_t parent = 0u; parent < NodeCount_; ++parent)
                {
                    auto edges = tx.get_edges(parent, Front(axis));
                    for (; edges.valid(); edges.next())
                    {
                        const auto data = edges.edge_data();
                        const std::size_t child = static_cast<std::size_t>(edges.dst_id());
                        unsigned ordinal = 0u;
                        if (!DecodeOrdinal(data, ordinal) || !Legal(parent, child))
                        {
                            Fatal_.store(true);
                            return checksum ^ UINT64_MAX;
                        }

                        checksum += Locator(child, ordinal);
                        if (payload)
                        {
                            const std::uint32_t word = static_cast<std::uint32_t>(
                                child % Words_);
                            std::uint64_t value = 0u;
                            if (!LoadPayloadFromTransaction(tx, child, word, value))
                            {
                                Fatal_.store(true);
                                return checksum ^ UINT64_MAX;
                            }
                            checksum ^= value;
                        }
                    }
                    // Match the generic First/Next traversal's terminal NONE.
                    checksum += static_cast<std::uint64_t>(ReadOperation::NONE);
                }
            }
        }
        catch (const std::exception&) { Fatal_.store(true); }
        return checksum;
    }

    std::uint64_t BenchmarkPayloadScan(bool atomic, std::uint32_t rounds) noexcept
    {
        (void)atomic; // LiveGraph property reads are snapshot reads in both rows.
        std::uint64_t checksum = 0u;
        if (Fatal_.load()) return checksum;
        try
        {
            for (std::uint32_t r = 0u; r < rounds; ++r)
            {
                auto tx = Graph_->begin_read_only_transaction();
                for (std::size_t node = 0u; node < NodeCount_; ++node)
                {
                    const auto data = tx.get_vertex(node);
                    if (data.size() != Words_ * sizeof(std::uint64_t))
                    {
                        Fatal_.store(true);
                        return checksum ^ UINT64_MAX;
                    }
                    for (std::uint32_t word = 0u; word < Words_; ++word)
                    {
                        std::uint64_t value = 0u;
                        std::memcpy(
                            &value,
                            data.data() + static_cast<std::size_t>(word) * sizeof(value),
                            sizeof(value));
                        checksum += value;
                    }
                }
            }
        }
        catch (const std::exception&) { Fatal_.store(true); }
        return checksum;
    }

    bool VerifyPayloadPattern() noexcept
    {
        if (Fatal_.load()) return false;
        try
        {
            auto tx = Graph_->begin_read_only_transaction();
            for (std::size_t node = 0u; node < NodeCount_; ++node)
            {
                const auto data = tx.get_vertex(node);
                if (data.size() != Words_ * sizeof(std::uint64_t)) return false;
                for (std::uint32_t word = 0u; word < Words_; ++word)
                {
                    std::uint64_t value = 0u;
                    std::memcpy(
                        &value,
                        data.data() + static_cast<std::size_t>(word) * sizeof(value),
                        sizeof(value));
                    const std::uint64_t expected =
                        (static_cast<std::uint64_t>(node + 1u) << 32u) ^
                        static_cast<std::uint64_t>(word + 1u);
                    if (value != expected) return false;
                }
            }
            return true;
        }
        catch (const std::exception&) { Fatal_.store(true); return false; }
    }

    // Fast quiescent Test-1 proof using LiveGraph's native snapshot/iterators.
    // It verifies the same exact expected H/V topology produced by
    // BuildFullTest1Graph, including both physical directions and ordinals.
    bool VerifyFullTest1Graph(const BenchmarkCase& config) noexcept
    {
        if (config.NodeCount != NodeCount_ ||
            config.ParentCapacity != K_ || Fatal_.load()) return false;
        try
        {
            auto tx = Graph_->begin_read_only_transaction();
            for (const Axis axis : {Axis::HORIZONTAL, Axis::VERTICAL})
            {
                std::uint64_t forward_count = 0u;
                for (std::size_t child = 0u; child < NodeCount_; ++child)
                {
                    const std::size_t count = std::min<std::size_t>(K_, child);
                    for (std::uint8_t ordinal = 0u; ordinal < K_; ++ordinal)
                    {
                        auto reverse = tx.get_edges(child, Back(axis, ordinal));
                        if (ordinal >= count)
                        {
                            if (reverse.valid()) return false;
                            continue;
                        }

                        const std::size_t expected_parent = axis == Axis::HORIZONTAL
                            ? child - 1u - ordinal
                            : static_cast<std::size_t>(ordinal);
                        if (!reverse.valid() ||
                            static_cast<std::size_t>(reverse.dst_id()) != expected_parent ||
                            !EdgeDataMatchesOrdinal(reverse.edge_data(), ordinal))
                            return false;
                        reverse.next();
                        if (reverse.valid()) return false;

                        const auto forward = tx.get_edge(
                            expected_parent, Front(axis), child);
                        if (!EdgeDataMatchesOrdinal(forward, ordinal)) return false;
                    }
                }

                for (std::size_t parent = 0u; parent < NodeCount_; ++parent)
                {
                    auto edges = tx.get_edges(parent, Front(axis));
                    for (; edges.valid(); edges.next())
                    {
                        const std::size_t child = static_cast<std::size_t>(edges.dst_id());
                        unsigned ordinal = 0u;
                        if (!DecodeOrdinal(edges.edge_data(), ordinal) ||
                            !Legal(parent, child)) return false;
                        const std::size_t expected_parent = axis == Axis::HORIZONTAL
                            ? child - 1u - ordinal
                            : ordinal;
                        if (expected_parent != parent) return false;
                        ++forward_count;
                    }
                }
                if (forward_count != EdgeCountPerAxis(NodeCount_, K_)) return false;
            }
            return true;
        }
        catch (const std::exception&) { Fatal_.store(true); return false; }
    }

    bool Healthy() const noexcept { return !Fatal_.load(); }

private:
    enum class ChildOp { First, Last, Next, Previous };

    static constexpr lg::label_t H_FRONT = 1u;
    static constexpr lg::label_t V_FRONT = 2u;
    static constexpr lg::label_t H_BACK_BASE = 0x0100u;
    static constexpr lg::label_t V_BACK_BASE = 0x0200u;

    static constexpr lg::label_t Front(Axis axis) noexcept
    { return axis == Axis::HORIZONTAL ? H_FRONT : V_FRONT; }

    static constexpr lg::label_t Back(Axis axis, unsigned ordinal) noexcept
    {
        return static_cast<lg::label_t>(
            (axis == Axis::HORIZONTAL ? H_BACK_BASE : V_BACK_BASE) + ordinal);
    }

    bool Legal(std::size_t parent, std::size_t child) const noexcept
    { return parent < child && child < NodeCount_; }

    std::uint32_t Locator(std::size_t child, unsigned ordinal) const noexcept
    { return static_cast<std::uint32_t>(child * K_ + ordinal); }

    static ReadResult Found(std::size_t node, std::uint32_t locator) noexcept
    { return {node, locator, ReadOperation::FOUND, true}; }

    bool DecodeOrdinal(std::string_view data, unsigned& ordinal) const noexcept
    {
        if (data.size() != 1u) return false;
        const unsigned encoded = static_cast<unsigned char>(data[0]);
        if (encoded == 0u || encoded > K_) return false;
        ordinal = encoded - 1u;
        return true;
    }

    bool EdgeDataMatchesOrdinal(std::string_view data, unsigned ordinal) const noexcept
    {
        unsigned decoded = 0u;
        return DecodeOrdinal(data, decoded) && decoded == ordinal;
    }

    ReadResult ParentRead(
        lg::Transaction& tx,
        std::size_t child,
        Axis axis,
        std::uint8_t ordinal)
    {
        auto edges = tx.get_edges(child, Back(axis, ordinal));
        if (!edges.valid()) return {};

        const std::size_t parent = static_cast<std::size_t>(edges.dst_id());
        if (!EdgeDataMatchesOrdinal(edges.edge_data(), ordinal) ||
            !Legal(parent, child))
        {
            Fatal_.store(true);
            return {};
        }
        edges.next();
        if (edges.valid())
        {
            Fatal_.store(true);
            return {};
        }
        return Found(parent, Locator(child, ordinal));
    }

    bool LoadPayloadFromTransaction(
        lg::Transaction& tx,
        std::size_t node,
        std::uint32_t word,
        std::uint64_t& value)
    {
        const auto data = tx.get_vertex(node);
        if (data.size() != Words_ * sizeof(std::uint64_t)) return false;
        std::memcpy(
            &value,
            data.data() + static_cast<std::size_t>(word) * sizeof(value),
            sizeof(value));
        return true;
    }

    ReadResult ChildRead(std::size_t parent, Axis axis, ChildOp op,
                         std::uint32_t cursor, std::uint32_t tries) noexcept
    {
        (void)tries;
        if (parent >= NodeCount_ || Fatal_.load()) return {};
        try
        {
            auto tx = Graph_->begin_read_only_transaction();
            // reverse=true reverses iterator order; it does not reverse the edge.
            const bool reverse = op == ChildOp::Last || op == ChildOp::Previous;
            auto edges = tx.get_edges(parent, Front(axis), reverse);
            bool after = op == ChildOp::First || op == ChildOp::Last;
            for (; edges.valid(); edges.next())
            {
                const auto data = edges.edge_data();
                const std::size_t child = static_cast<std::size_t>(edges.dst_id());
                unsigned ordinal = 0u;
                if (!DecodeOrdinal(data, ordinal) || !Legal(parent, child))
                {
                    Fatal_.store(true);
                    return {};
                }
                const auto loc = Locator(child, ordinal);
                if (after) return Found(child, loc);
                if (loc == cursor) after = true;
            }
            return {};
        }
        catch (const std::exception&) { Fatal_.store(true); return {}; }
    }

    std::unique_ptr<lg::Graph> Graph_{};
    std::size_t NodeCount_ = 0u, Words_ = 0u;
    std::uint8_t K_ = 0u;
    std::vector<std::uint64_t> Staging_{};
    std::size_t StageNode_ = 0u, StageWord_ = 0u;
    std::atomic<bool> Fatal_{false};
};

// -----------------------------------------------------------------------------
// Native one-way LiveGraph adapter.
//
// One logical parent relation is one physical child->parent edge.  This adapter is
// intentionally NOT Fabric-equivalent because it does not maintain parent->child
// reverse adjacency.  It exists to separate LiveGraph's native transaction cost
// from the extra work required to emulate Fabric's bidirectional relation contract.
// Tests 2-3 only require one live parent per (child,axis), which is exactly what this
// adapter represents.
// -----------------------------------------------------------------------------
class GenericLiveGraphBackend
{
public:
    bool Initialize(std::size_t nodes, std::size_t words, std::uint8_t k,
                    bool single_payload_region = false)
    {
        (void)single_payload_region;
        if (nodes == 0u || nodes > UINT32_MAX || words == 0u || words > UINT32_MAX ||
            k == 0u || k > ADS::COMPILED_MAX_DIRECT_PARENTS_PER_AXIS)
            return false;
        try
        {
            Fatal_.store(false, std::memory_order_release);
            Graph_ = std::make_unique<lg::Graph>("", "", 1ull << 40,
                static_cast<lg::vertex_t>(nodes + 1u));
            auto tx = Graph_->begin_transaction();
            for (std::size_t node = 0u; node < nodes; ++node)
                if (tx.new_vertex() != node) return false;
            tx.commit();
            NodeCount_ = nodes;
            Words_ = words;
            K_ = k;
            return true;
        }
        catch (const std::exception&) { Fatal_.store(true); return false; }
    }

    bool PrimePayloadStorage() noexcept
    {
        if (!Graph_ || Words_ == 0u || Fatal_.load()) return false;
        try
        {
            std::vector<std::uint64_t> zeros(Words_, 0u);
            const std::string_view payload(
                reinterpret_cast<const char*>(zeros.data()),
                zeros.size() * sizeof(std::uint64_t));
            auto tx = Graph_->begin_transaction();
            for (std::size_t node = 0u; node < NodeCount_; ++node)
                tx.put_vertex(static_cast<lg::vertex_t>(node), payload);
            tx.commit();
            return true;
        }
        catch (const lg::Transaction::RollbackExcept&) { return false; }
        catch (const std::exception&) { Fatal_.store(true); return false; }
    }

    bool AddParent(std::size_t parent, std::size_t child, Axis axis,
                   std::uint32_t tries = DEFAULT_MAX_TRIES) noexcept
    {
        (void)tries;
        if (!Legal(parent, child) || Fatal_.load()) return false;
        try
        {
            auto tx = Graph_->begin_transaction();
            if (!tx.get_edge(child, Label(axis), parent).empty()) return false;
            const char present = 1;
            tx.put_edge(child, Label(axis), parent, std::string_view(&present, 1u));
            tx.commit();
            return true;
        }
        catch (const lg::Transaction::RollbackExcept&) { return false; }
        catch (const std::exception&) { Fatal_.store(true); return false; }
    }

    bool ReplaceParent(std::size_t old_parent, std::size_t new_parent,
                       std::size_t child, Axis axis,
                       std::uint32_t tries = DEFAULT_MAX_TRIES) noexcept
    {
        (void)tries;
        if (!Legal(old_parent, child) || !Legal(new_parent, child) ||
            old_parent == new_parent || Fatal_.load()) return false;
        try
        {
            auto tx = Graph_->begin_transaction();
            const auto old = tx.get_edge(child, Label(axis), old_parent);
            if (old.size() != 1u ||
                !tx.get_edge(child, Label(axis), new_parent).empty())
                return false;
            if (!tx.del_edge(child, Label(axis), old_parent)) return false;
            tx.put_edge(child, Label(axis), new_parent, old);
            tx.commit();
            return true;
        }
        catch (const lg::Transaction::RollbackExcept&) { return false; }
        catch (const std::exception&) { Fatal_.store(true); return false; }
    }

    ReadResult FindParent(std::size_t child, Axis axis, std::uint8_t ordinal,
                          std::uint32_t tries = 1u) noexcept
    {
        (void)tries;
        if (child >= NodeCount_ || ordinal != 0u || Fatal_.load()) return {};
        try
        {
            auto tx = Graph_->begin_read_only_transaction();
            auto edges = tx.get_edges(child, Label(axis));
            if (!edges.valid()) return {};
            const auto parent = static_cast<std::size_t>(edges.dst_id());
            if (!Legal(parent, child) || edges.edge_data().size() != 1u)
            { Fatal_.store(true); return {}; }
            edges.next();
            if (edges.valid())
            {
                // Tests 2-3 intentionally model one current parent per axis.
                Fatal_.store(true);
                return {};
            }
            return {parent, Locator(child, axis), ReadOperation::FOUND, true};
        }
        catch (const std::exception&) { Fatal_.store(true); return {}; }
    }

    ReadResult StableFindParent(std::size_t child, Axis axis,
                                std::uint8_t ordinal,
                                std::uint32_t tries = 1u) noexcept
    {
        // Fresh MVCC snapshot per logical read, matching the bidirectional adapter.
        return FindParent(child, axis, ordinal, tries);
    }

    bool Healthy() const noexcept { return !Fatal_.load(); }

private:
    static constexpr lg::label_t H_PARENT = 0x0400u;
    static constexpr lg::label_t V_PARENT = 0x0401u;

    static constexpr lg::label_t Label(Axis axis) noexcept
    { return axis == Axis::HORIZONTAL ? H_PARENT : V_PARENT; }

    bool Legal(std::size_t parent, std::size_t child) const noexcept
    { return parent < child && child < NodeCount_; }

    static std::uint32_t Locator(std::size_t child, Axis axis) noexcept
    {
        return static_cast<std::uint32_t>(
            child * 2u + (axis == Axis::VERTICAL ? 1u : 0u));
    }

    std::unique_ptr<lg::Graph> Graph_{};
    std::size_t NodeCount_ = 0u;
    std::size_t Words_ = 0u;
    std::uint8_t K_ = 0u;
    std::atomic<bool> Fatal_{false};
};

namespace ExternalFairness
{
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

// Generic/native LiveGraph intentionally has no reverse-child adjacency.
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
        // Native one-way LiveGraph is valid only when the caller intentionally
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

struct DiskFootprint
{
    bool Ok = false;
    std::uint64_t LogicalBytes = 0u;
    std::uint64_t AllocatedBytes = 0u;
};

inline std::uint64_t LogicalFileBytes(const std::filesystem::path& path) noexcept
{
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    return ec ? 0u : static_cast<std::uint64_t>(size);
}

inline std::uint64_t AllocatedFileBytes(const std::filesystem::path& path) noexcept
{
#if defined(__linux__) || defined(__unix__)
    struct stat st{};
    if (::stat(path.c_str(), &st) == 0)
        return static_cast<std::uint64_t>(st.st_blocks) * 512u;
#endif
    return LogicalFileBytes(path);
}

inline DiskFootprint DirectoryFootprint(const std::filesystem::path& root) noexcept
{
    DiskFootprint result{true, 0u, 0u};
    std::error_code ec;
    if (!std::filesystem::exists(root, ec) || ec) return {false, 0u, 0u};
    for (std::filesystem::recursive_directory_iterator it(root, ec), end;
         !ec && it != end; it.increment(ec))
    {
        if (!it->is_regular_file(ec) || ec) continue;
        result.LogicalBytes += LogicalFileBytes(it->path());
        result.AllocatedBytes += AllocatedFileBytes(it->path());
    }
    result.Ok = !ec;
    return result;
}

inline std::filesystem::path UniqueStorageRoot(const char* tag)
{
    static std::atomic<std::uint64_t> counter{0u};
    const auto id = counter.fetch_add(1u, std::memory_order_relaxed);
    return std::filesystem::temp_directory_path() /
        (std::string("livegraph_vs_supernova_") + tag + "_" + std::to_string(id));
}

inline DiskFootprint ProbeLiveGraphDiskFootprint(
    const BenchmarkCase& config,
    bool bidirectional)
{
    const std::filesystem::path root = UniqueStorageRoot(
        bidirectional ? "bidir" : "native");
    const std::filesystem::path block = root / "block";
    const std::filesystem::path wal = root / "wal";
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    ec.clear();
    std::filesystem::create_directories(root, ec);
    if (ec) return {};

    bool built = false;
    try
    {
        {
            lg::Graph graph(block.string(), wal.string(), 1ull << 40,
                static_cast<lg::vertex_t>(config.NodeCount + 1u));
            auto tx = graph.begin_batch_loader();
            for (std::size_t node = 0u; node < config.NodeCount; ++node)
            {
                if (tx.new_vertex() != node) throw std::runtime_error("vertex id mismatch");
                std::array<std::uint64_t, TEST1_PAYLOAD_WORDS> payload{};
                for (std::uint32_t word = 0u; word < TEST1_PAYLOAD_WORDS; ++word)
                    payload[word] =
                        (static_cast<std::uint64_t>(node + 1u) << 32u) ^
                        static_cast<std::uint64_t>(word + 1u);
                tx.put_vertex(node, std::string_view(
                    reinterpret_cast<const char*>(payload.data()), sizeof(payload)));
            }

            constexpr lg::label_t H_NATIVE = 0x0400u;
            constexpr lg::label_t V_NATIVE = 0x0401u;
            constexpr lg::label_t H_FRONT = 1u;
            constexpr lg::label_t V_FRONT = 2u;
            constexpr lg::label_t H_BACK_BASE = 0x0100u;
            constexpr lg::label_t V_BACK_BASE = 0x0200u;

            for (std::size_t child = 0u; child < config.NodeCount; ++child)
            {
                const std::size_t count = std::min<std::size_t>(
                    config.ParentCapacity, child);
                for (std::size_t ordinal = 0u; ordinal < count; ++ordinal)
                {
                    const std::size_t h_parent = child - 1u - ordinal;
                    const std::size_t v_parent = ordinal;
                    const char encoded = static_cast<char>(ordinal + 1u);
                    const std::string_view data(&encoded, 1u);
                    if (!bidirectional)
                    {
                        tx.put_edge(child, H_NATIVE, h_parent, data);
                        tx.put_edge(child, V_NATIVE, v_parent, data);
                    }
                    else
                    {
                        tx.put_edge(h_parent, H_FRONT, child, data);
                        tx.put_edge(child,
                            static_cast<lg::label_t>(H_BACK_BASE + ordinal),
                            h_parent, data);
                        tx.put_edge(v_parent, V_FRONT, child, data);
                        tx.put_edge(child,
                            static_cast<lg::label_t>(V_BACK_BASE + ordinal),
                            v_parent, data);
                    }
                }
            }
            tx.commit();
            built = true;
        }
    }
    catch (const std::exception&)
    {
        built = false;
    }

    const DiskFootprint footprint = built ? DirectoryFootprint(root) : DiskFootprint{};
    std::filesystem::remove_all(root, ec);
    return footprint;
}

inline void PrintStorageFootprint(
    const BenchmarkCase& config,
    std::size_t fabric_serialized_bytes)
{
    std::cout << "    [storage] persistent footprint probe..." << std::flush;
    const DiskFootprint native = ProbeLiveGraphDiskFootprint(config, false);
    const DiskFootprint bidir = ProbeLiveGraphDiskFootprint(config, true);
    std::cout << ((native.Ok && bidir.Ok) ? " done\n" : " partial\n");
    auto mib = [](std::uint64_t bytes) noexcept
    { return static_cast<double>(bytes) / (1024.0 * 1024.0); };

    if (native.Ok)
        std::cout << "      LiveGraph-native disk: allocated=" << std::fixed
                  << std::setprecision(3) << mib(native.AllocatedBytes)
                  << " MiB  logical-files=" << mib(native.LogicalBytes) << " MiB\n";
    else
        std::cout << "      LiveGraph-native disk: probe unavailable\n";

    if (bidir.Ok)
        std::cout << "      LiveGraph-bidir  disk: allocated=" << std::fixed
                  << std::setprecision(3) << mib(bidir.AllocatedBytes)
                  << " MiB  logical-files=" << mib(bidir.LogicalBytes) << " MiB\n";
    else
        std::cout << "      LiveGraph-bidir  disk: probe unavailable\n";

    std::cout << "      Fabric in-memory slab: " << std::fixed << std::setprecision(3)
              << mib(static_cast<std::uint64_t>(fabric_serialized_bytes))
              << " MiB (SlabBytesForTest; not a disk allocation)\n"
              << "      Storage figures use different persistence formats and are descriptive.\n";
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

struct ComparisonTiming { double LiveGraph = 0.0, Fabric = 0.0; };

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
        << " LG=" << std::right << std::setw(10) << std::fixed
        << std::setprecision(2) << t.LiveGraph << " ns/op"
        << "  Fabric=" << std::setw(10) << t.Fabric << " ns/op"
        << "  cost(LG/F)=" << std::setw(8)
        << Ratio(t.LiveGraph, t.Fabric) << "x\n";
}


namespace Test01
{
using namespace BenchmarkCore;

// TestKit's 100,000 replacement pairs were appropriate for the very cheap
// in-process baseline, but with a transactional MVCC graph they expand to
// 800,000 committed write transactions per axis across four measured samples.
// Keep the same operation count for both backends while bounding each measured
// sample to a long-enough, publication-useful interval. This is intentionally
// local to the external comparison and does not alter TestKit.
constexpr std::uint32_t TEST1_REPLACE_PAIRS_PER_SAMPLE = 5'000u;
constexpr std::uint64_t TEST1_REPLACE_OPS_PER_SAMPLE =
    static_cast<std::uint64_t>(TEST1_REPLACE_PAIRS_PER_SAMPLE) * 2u;


inline bool BuildGenericTest1MutationContext(
    GenericLiveGraphBackend& backend,
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
        << "      LG-native=" << std::right << std::setw(10) << std::fixed
        << std::setprecision(2) << timing.Native << " ns/op"
        << "  LG-bidir=" << std::setw(10) << timing.Bidir << " ns/op"
        << "  Fabric=" << std::setw(10) << timing.Fabric << " ns/op\n"
        << "      cost ratios: LG-native/Fabric="
        << Ratio(timing.Native, timing.Fabric) << "x  LG-bidir/Fabric="
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
                LiveGraphBackend backend{};
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

    StageBegin(2u, STAGE_COUNT, "persistent graph build");
    GenericLiveGraphBackend native_backend{};
    LiveGraphBackend live_backend{};
    RuntimeAPCFabricBackend fabric_backend{};
    const bool persistent_build =
        BuildGenericTest1MutationContext(native_backend, config) &&
        BuildFullTest1Graph(live_backend, config) &&
        BuildFullTest1Graph(fabric_backend, config);
    StageEnd(persistent_build);
    if (!persistent_build) return false;

    std::cout << "    edges/axis=" << edge_count
              << "  payload/node=" << TEST1_PAYLOAD_WORDS * sizeof(std::uint64_t)
              << " B  Fabric slab=" << fabric_backend.ApproxStorageBytes()
              << " B (persistent LiveGraph footprint is probed separately below)\n";

    ExternalFairness::PrintStorageFootprint(
        config, fabric_backend.ApproxStorageBytes());

    StageBegin(3u, STAGE_COUNT, "pre-benchmark topology + payload verification");
    const bool live_proof = live_backend.VerifyFullTest1Graph(config);
    const GraphProof fabric_proof = ProveRuntimeCombinedDAG(fabric_backend, config);
    const bool payload_ok = live_backend.VerifyPayloadPattern() &&
        VerifyPayload(fabric_backend, config);
    bool ok = live_proof && fabric_proof.Passed() && payload_ok &&
        live_backend.Healthy();
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
                    // LiveGraph, whose adapter deliberately performs one txn.
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
        [&] { return parent_scan(live_backend, Axis::HORIZONTAL); },
        [&] { return parent_scan(fabric_backend, Axis::HORIZONTAL); });
    StageEnd(live_backend.Healthy());
    PrintComparisonRow("H parent scan", h_parent);

    StageBegin(5u, STAGE_COUNT, "V parent lookup bulk scan");
    const ComparisonTiming v_parent = MeasureComparison(
        [&] { return parent_scan(live_backend, Axis::VERTICAL); },
        [&] { return parent_scan(fabric_backend, Axis::VERTICAL); });
    StageEnd(live_backend.Healthy());
    PrintComparisonRow("V parent scan", v_parent);

    StageBegin(6u, STAGE_COUNT, "H native reverse-child traversal");
    const ComparisonTiming h_reverse = MeasureComparison(
        [&] { return reverse_scan(live_backend, Axis::HORIZONTAL, false); },
        [&] { return reverse_scan(fabric_backend, Axis::HORIZONTAL, false); });
    StageEnd(live_backend.Healthy());
    PrintComparisonRow("H reverse-child scan", h_reverse);

    StageBegin(7u, STAGE_COUNT, "V native reverse-child traversal");
    const ComparisonTiming v_reverse = MeasureComparison(
        [&] { return reverse_scan(live_backend, Axis::VERTICAL, false); },
        [&] { return reverse_scan(fabric_backend, Axis::VERTICAL, false); });
    StageEnd(live_backend.Healthy());
    PrintComparisonRow("V reverse-child scan", v_reverse);

    StageBegin(8u, STAGE_COUNT, "sequential payload scan: LiveGraph snapshot vs Fabric direct");
    const ComparisonTiming direct = MeasureComparison(
        [&] { return payload_scan(live_backend, false); },
        [&] { return payload_scan(fabric_backend, false); });
    StageEnd(live_backend.Healthy());
    PrintComparisonRow("payload seq snapshot/direct", direct);

    StageBegin(9u, STAGE_COUNT, "sequential payload scan: LiveGraph snapshot vs Fabric atomic");
    const ComparisonTiming atomic = MeasureComparison(
        [&] { return payload_scan(live_backend, true); },
        [&] { return payload_scan(fabric_backend, true); });
    StageEnd(live_backend.Healthy());
    PrintComparisonRow("payload seq snapshot/atomic", atomic);

    StageBegin(10u, STAGE_COUNT, "H child traversal + payload");
    const ComparisonTiming graph_payload = MeasureComparison(
        [&] { return reverse_scan(live_backend, Axis::HORIZONTAL, true); },
        [&] { return reverse_scan(fabric_backend, Axis::HORIZONTAL, true); });
    StageEnd(live_backend.Healthy());
    PrintComparisonRow("H child + payload", graph_payload);

    std::cout << "    replacement workload="
              << TEST1_REPLACE_OPS_PER_SAMPLE
              << " logical replacements/sample x "
              << ConcurrencyConfig::MEASURED_RUNS << " samples/backend\n";

    StageBegin(11u, STAGE_COUNT, "H parent replacement: native / bidir / Fabric");
    const ThreeWayReplacementTiming h_replace = MeasureThreeWayReplacement(
        [&] { return replace_scan(native_backend, Axis::HORIZONTAL); },
        [&] { return replace_scan(live_backend, Axis::HORIZONTAL); },
        [&] { return replace_scan(fabric_backend, Axis::HORIZONTAL); });
    StageEnd(all_replacements_succeeded && native_backend.Healthy() && live_backend.Healthy());
    if (!all_replacements_succeeded || !native_backend.Healthy() ||
        !live_backend.Healthy()) return false;
    PrintThreeWayReplacementRow("H parent replace", h_replace);

    StageBegin(12u, STAGE_COUNT, "V parent replacement: native / bidir / Fabric");
    const ThreeWayReplacementTiming v_replace = MeasureThreeWayReplacement(
        [&] { return replace_scan(native_backend, Axis::VERTICAL); },
        [&] { return replace_scan(live_backend, Axis::VERTICAL); },
        [&] { return replace_scan(fabric_backend, Axis::VERTICAL); });
    StageEnd(all_replacements_succeeded && native_backend.Healthy() && live_backend.Healthy());
    if (!all_replacements_succeeded || !native_backend.Healthy() ||
        !live_backend.Healthy()) return false;
    PrintThreeWayReplacementRow("V parent replace", v_replace);

    StageBegin(13u, STAGE_COUNT, "post-mutation topology verification");
    const bool live_after = live_backend.VerifyFullTest1Graph(config);
    const GraphProof fabric_after = ProveRuntimeCombinedDAG(fabric_backend, config);
    ok = ok && all_replacements_succeeded && live_after &&
        fabric_after.Passed() && native_backend.Healthy() && live_backend.Healthy();
    StageEnd(ok);
    std::cout << "    integrity=" << (ok ? "PASS" : "FAIL") << '\n';
    return ok;
}

inline Result Run(const std::array<BenchmarkCase, 4u>& cases)
{
    Banner("TEST 1 - SCALED FAIR BIDIRECTIONAL DAG / 1 KiB PAYLOAD COMPARISON");
    std::cout
        << "Official LiveGraph vs single-region SuperNova Fabric.\n"
        << "Each case uses the same N, K, fully populated H/V bounded topology, and\n"
        << "exactly 128 x uint64_t (1024 B) payload per node. Four measured samples\n"
        << "are taken per row with backend order alternated. LiveGraph bulk read rows\n"
        << "use a retained read-only snapshot and native iterator for each scan round.\n"
        << "Read/traversal rows require the bidirectional contract; replacement rows additionally\n"
        << "show LG-native one-edge transactions as a non-equivalent lower bound.\n";

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
    GenericLiveGraphBackend& native,
    LiveGraphBackend& bidir,
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

            GenericLiveGraphBackend native{};
            LiveGraphBackend bidir{};
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
        << "Equal wall-clock interval for all backends. N=LG-native (one-way lower bound),\n"
        << "B=LG-bidir (end-to-end bidirectional), F=SuperNova Fabric. Throughput is\n"
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
    GenericLiveGraphBackend& native,
    LiveGraphBackend& bidir,
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

            GenericLiveGraphBackend native{};
            LiveGraphBackend bidir{};
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
        << "Equal read/write interval for all backends. N=LG-native, B=LG-bidir, F=Fabric.\n"
        << "Both LiveGraph modes use a fresh MVCC snapshot/read; Fabric uses its public\n"
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
                   "LiveGraphVsSuperNova_results.txt")
{
    ScopedBenchmarkOutput output(result_path);
    if (!output.IsOpen())
    {
        std::cerr
            << "LiveGraphVsSuperNova: failed to open result file: "
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
    const auto cases = MakeBenchmarkCases(lower_node_count, higher_node_count,
        static_cast<std::uint8_t>(lower_parent_capacity),
        static_cast<std::uint8_t>(higher_parent_capacity));
    const auto workers = usable_thread_count - 2u;
    if (measurement_ms == 0u) return 1;
    const ExternalFairness::Duration measurement(measurement_ms);
    std::cout << "\nLiveGraph-native vs LiveGraph-bidirectional vs SuperNova; "
              << "four measured samples/point; writer/reader sweep 1.." << workers
              << "; fixed interval=" << measurement_ms << " ms/system/sample\n";
    const auto first = Test01::Run(cases);
    const auto second = Test02::Run(cases, workers, measurement);
    const auto third = Test03::Run(cases, workers, measurement);
    Banner("LIVEGRAPH VS SUPERNOVA TESTS 1-3 SUMMARY");
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
                      "LiveGraphVsSuperNova_results.txt")
{
    return Run(lower_node_count, higher_node_count, lower_parent_capacity,
               higher_parent_capacity, usable_thread_count, measurement_ms,
               result_path);
}
} // namespace LiveGraphVsSuperNova
