#pragma once

// Place in a benchmark directory beside the LiveGraph and SuperNova repository
// directories (the same layout as the original include stub). Compile as C++20
// and link the official LiveGraph and SuperNova implementations.
// int main() { return LiveGraphVsSuperNova::Run(); }
//
// Each H/V logical edge uses two physical LiveGraph edges. Forward labels group
// child traversal; reverse labels encode the parent ordinal so FindParent is a
// direct labelled lookup rather than an adapter-induced O(K) scan. Both physical
// directions change in one transaction. TestKit schedules/barriers/counts remain
// unchanged. Test 1 bulk scans retain one native LiveGraph snapshot/iterator per
// scan round; Tests 2-3 retain one transaction per logical mutation/stable read.
// LiveGraph rollback retries and Fabric validation retries remain separate metrics.

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
    if (sample == 0u)
    {
        std::cout << "    [" << kind << ' ' << current << '/' << total
                  << "] samples" << std::flush;
    }
    std::cout << ' ' << (sample + 1u) << '/' << sample_total << std::flush;
}

inline void SweepSampleDone()
{
    std::cout << " done\n";
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
        << " LiveGraph=" << std::right << std::setw(10) << std::fixed
        << std::setprecision(2) << t.LiveGraph << " ns/op"
        << "  Fabric=" << std::setw(10) << t.Fabric << " ns/op"
        << "  Fabric/LiveGraph=" << std::setw(7)
        << Ratio(t.Fabric, t.LiveGraph) << "x\n";
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
    PrintComparisonRow("construction", construction);

    StageBegin(2u, STAGE_COUNT, "persistent graph build");
    LiveGraphBackend live_backend{};
    RuntimeAPCFabricBackend fabric_backend{};
    const bool persistent_build =
        BuildFullTest1Graph(live_backend, config) &&
        BuildFullTest1Graph(fabric_backend, config);
    StageEnd(persistent_build);
    if (!persistent_build) return false;

    std::cout << "    edges/axis=" << edge_count
              << "  payload/node=" << TEST1_PAYLOAD_WORDS * sizeof(std::uint64_t)
              << " B  Fabric slab=" << fabric_backend.ApproxStorageBytes()
              << " B (LiveGraph allocation not comparable to slab bytes)\n";

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

    StageBegin(11u, STAGE_COUNT, "H atomic parent replacement");
    const ComparisonTiming h_replace = MeasureComparison(
        [&] { return replace_scan(live_backend, Axis::HORIZONTAL); },
        [&] { return replace_scan(fabric_backend, Axis::HORIZONTAL); });
    StageEnd(all_replacements_succeeded && live_backend.Healthy());
    PrintComparisonRow("H parent replace", h_replace);

    StageBegin(12u, STAGE_COUNT, "V atomic parent replacement");
    const ComparisonTiming v_replace = MeasureComparison(
        [&] { return replace_scan(live_backend, Axis::VERTICAL); },
        [&] { return replace_scan(fabric_backend, Axis::VERTICAL); });
    StageEnd(all_replacements_succeeded && live_backend.Healthy());
    PrintComparisonRow("V parent replace", v_replace);

    StageBegin(13u, STAGE_COUNT, "post-mutation topology verification");
    const bool live_after = live_backend.VerifyFullTest1Graph(config);
    const GraphProof fabric_after = ProveRuntimeCombinedDAG(fabric_backend, config);
    ok = ok && all_replacements_succeeded && live_after &&
        fabric_after.Passed() && live_backend.Healthy();
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
        << "use a retained read-only snapshot and native iterator for each scan round.\n";

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

inline bool RunCase(
    const BenchmarkCase& config,
    MutationLocality locality,
    std::size_t max_writers,
    std::size_t case_index)
{
    const auto maybe_scenario = MakeMutationScenario(config, locality, max_writers);
    if (!maybe_scenario.has_value()) return false;
    const MutationScenario scenario = maybe_scenario.value();
    const MutationSchedule schedule = BuildMutationSchedule(scenario);

    std::cout
        << "\n  CASE " << case_index << "/4"
        << "  N=" << config.NodeCount
        << "  K=" << static_cast<unsigned>(config.ParentCapacity)
        << "  parent-pool=" << scenario.ParentCount << '\n';

    bool all_ok = true;
    for (std::size_t writer_count = 1u; writer_count <= max_writers; ++writer_count)
    {
        std::array<double, ConcurrencyConfig::MEASURED_RUNS> live_ns{};
        std::array<double, ConcurrencyConfig::MEASURED_RUNS> fabric_ns{};
        std::array<double, ConcurrencyConfig::MEASURED_RUNS> retry_rate{};
        std::array<double, ConcurrencyConfig::MEASURED_RUNS> live_retry_rate{};
        bool live_ok = true;

        for (std::size_t run = 0u; run < ConcurrencyConfig::MEASURED_RUNS; ++run)
        {
            SweepSampleProgress(
                "writers", writer_count, max_writers, run,
                ConcurrencyConfig::MEASURED_RUNS);
            LiveGraphBackend live_backend{};
            RuntimeAPCFabricBackend fabric_backend{};
            if (
                !BuildMutationBackendFair(live_backend, scenario, writer_count) ||
                !BuildMutationBackendFair(fabric_backend, scenario, writer_count)
            )
            {
                std::cout << " SETUP FAIL\n";
                return false;
            }

            MutationSweepResult live_result{};
            MutationSweepResult fabric_result{};
            if ((run & 1u) == 0u)
            {
                live_result = RunMutationWorkers(
                    live_backend, scenario, schedule, writer_count);
                fabric_result = RunMutationWorkers(
                    fabric_backend, scenario, schedule, writer_count);
            }
            else
            {
                fabric_result = RunMutationWorkers(
                    fabric_backend, scenario, schedule, writer_count);
                live_result = RunMutationWorkers(
                    live_backend, scenario, schedule, writer_count);
            }

            const bool integrity =
                live_result.Ok && fabric_result.Ok &&
                VerifyMutationScenario(
                    live_backend, scenario, schedule, writer_count) &&
                VerifyMutationScenario(
                    fabric_backend, scenario, schedule, writer_count);
            live_ok = live_ok && integrity && live_backend.Healthy();
            live_ns[run] = live_result.NsPerSuccess;
            fabric_ns[run] = fabric_result.NsPerSuccess;
            live_retry_rate[run] = live_result.Success == 0u ? 0.0 :
                static_cast<double>(live_result.Retries) /
                static_cast<double>(live_result.Success);
            retry_rate[run] = fabric_result.Success == 0u ? 0.0 :
                static_cast<double>(fabric_result.Retries) /
                static_cast<double>(fabric_result.Success);
        }
        SweepSampleDone();

        const double live_median = Median(live_ns);
        const double fabric_median = Median(fabric_ns);
        const double retries = Median(retry_rate);
        const double live_retries = Median(live_retry_rate);
        const double live_mops = live_median > 0.0
            ? ConcurrencyConfig::MILLION_OPERATIONS_PER_SECOND_FROM_NS / live_median
            : 0.0;
        const double fabric_mops = fabric_median > 0.0
            ? ConcurrencyConfig::MILLION_OPERATIONS_PER_SECOND_FROM_NS / fabric_median
            : 0.0;

        all_ok = all_ok && live_ok;
        std::cout
            << "    threads=" << std::setw(2) << writer_count
            << "  LiveGraph=" << std::setw(9) << std::fixed << std::setprecision(2)
            << live_median << " ns/op (" << std::setw(7) << live_mops << " M/s)"
            << "  Fabric=" << std::setw(9) << fabric_median
            << " ns/op (" << std::setw(7) << fabric_mops << " M/s)"
            << "  Fabric/LiveGraph=" << std::setw(6)
            << Ratio(fabric_median, live_median) << "x"
            << "  retry/success LG/Fabric=" << std::setprecision(4)
            << live_retries << "/" << retries
            << "  " << (live_ok ? "PASS" : "FAIL") << '\n';
    }
    return all_ok;
}

inline Result Run(
    const std::array<BenchmarkCase, 4u>& cases,
    std::size_t max_writers)
{
    Banner("TEST 2A - HOTSPOT STRUCTURAL MUTATION");
    std::cout
        << "Each writer owns one child; all writers contend on the same two H and two V\n"
        << "parents. LiveGraph commits each replacement as one transaction. Structural\n"
        << "tests use one minimal payload word on both backends so unused payload regions\n"
        << "do not dominate cache footprint. Every writer count 1..usable_threads-2 is measured.\n";

    bool hotspot_ok = true;
    for (std::size_t i = 0u; i < cases.size(); ++i)
        hotspot_ok = RunCase(
            cases[i], MutationLocality::HOTSPOT, max_writers, i + 1u) && hotspot_ok;
    std::cout << "\nTEST 2A OVERALL: " << (hotspot_ok ? "PASS" : "FAIL") << '\n';

    Banner("TEST 2B - DISTRIBUTED STRUCTURAL MUTATION");
    std::cout
        << "Each writer owns one child and follows the same deterministic random schedule\n"
        << "for both backends. The parent pool is min(100, legal predecessor nodes).\n"
        << "The same minimal structural payload and 1..(usable_threads-2) sweep are used.\n";

    bool distributed_ok = true;
    for (std::size_t i = 0u; i < cases.size(); ++i)
        distributed_ok = RunCase(
            cases[i], MutationLocality::DISTRIBUTED, max_writers, i + 1u) && distributed_ok;
    std::cout << "\nTEST 2B OVERALL: " << (distributed_ok ? "PASS" : "FAIL") << '\n';

    const bool ok = hotspot_ok && distributed_ok;
    std::cout << "\nTEST 2 OVERALL: " << (ok ? "PASS" : "FAIL") << '\n';
    return ok ? Result::PASS : Result::FAIL;
}
} // namespace Test02

namespace Test03
{
using namespace BenchmarkCore;

inline bool RunCase(
    const BenchmarkCase& config,
    bool distributed,
    std::size_t max_readers,
    std::size_t case_index)
{
    const auto maybe_scenario = MakeReaderScenario(config, distributed);
    if (!maybe_scenario.has_value()) return false;
    const ReaderScenario scenario = maybe_scenario.value();
    const ParentSchedule schedule = BuildParentSchedule(scenario);

    std::cout
        << "\n  CASE " << case_index << "/4"
        << "  N=" << config.NodeCount
        << "  K=" << static_cast<unsigned>(config.ParentCapacity)
        << "  parent-pool=" << scenario.ParentCount << '\n';

    bool all_ok = true;
    for (std::size_t reader_count = 1u; reader_count <= max_readers; ++reader_count)
    {
        std::array<double, ConcurrencyConfig::MEASURED_RUNS> live_ns{};
        std::array<double, ConcurrencyConfig::MEASURED_RUNS> fabric_ns{};
        std::array<double, ConcurrencyConfig::MEASURED_RUNS> retry_rate{};
        std::array<double, ConcurrencyConfig::MEASURED_RUNS> live_retry_rate{};
        std::array<double, ConcurrencyConfig::MEASURED_RUNS> live_starvations{};
        std::array<double, ConcurrencyConfig::MEASURED_RUNS> fabric_starvations{};
        std::array<double, ConcurrencyConfig::MEASURED_RUNS> live_writer_mops{};
        std::array<double, ConcurrencyConfig::MEASURED_RUNS> fabric_writer_mops{};
        bool live_ok = true;
        bool live_correctness_ok = true;
        bool fabric_correctness_ok = true;
        bool live_progress_ok = true;
        bool fabric_progress_ok = true;

        for (std::size_t run = 0u; run < ConcurrencyConfig::MEASURED_RUNS; ++run)
        {
            SweepSampleProgress(
                "readers", reader_count, max_readers, run,
                ConcurrencyConfig::MEASURED_RUNS);
            LiveGraphBackend live_backend{};
            RuntimeAPCFabricBackend fabric_backend{};
            if (
                !BuildReaderBackendFair(live_backend, scenario) ||
                !BuildReaderBackendFair(fabric_backend, scenario)
            )
            {
                std::cout << " SETUP FAIL\n";
                return false;
            }

            ReaderSweepResult live_result{};
            ReaderSweepResult fabric_result{};
            if ((run & 1u) == 0u)
            {
                live_result = RunReadersWithWriters(
                    live_backend, scenario, schedule, reader_count);
                fabric_result = RunReadersWithWriters(
                    fabric_backend, scenario, schedule, reader_count);
            }
            else
            {
                fabric_result = RunReadersWithWriters(
                    fabric_backend, scenario, schedule, reader_count);
                live_result = RunReadersWithWriters(
                    live_backend, scenario, schedule, reader_count);
            }

            live_ok = live_ok && live_result.Ok && fabric_result.Ok && live_backend.Healthy();
            live_correctness_ok =
                live_correctness_ok && live_result.CorrectnessOk;
            fabric_correctness_ok =
                fabric_correctness_ok && fabric_result.CorrectnessOk;
            live_progress_ok =
                live_progress_ok && live_result.ProgressOk;
            fabric_progress_ok =
                fabric_progress_ok && fabric_result.ProgressOk;

            live_ns[run] = live_result.NsPerStableRead;
            fabric_ns[run] = fabric_result.NsPerStableRead;
            live_retry_rate[run] = live_result.StableReads == 0u ? 0.0 :
                static_cast<double>(live_result.ReaderRetries) /
                static_cast<double>(live_result.StableReads);
            retry_rate[run] = fabric_result.StableReads == 0u ? 0.0 :
                static_cast<double>(fabric_result.ReaderRetries) /
                static_cast<double>(fabric_result.StableReads);
            live_starvations[run] =
                static_cast<double>(live_result.ReaderStarvations);
            fabric_starvations[run] =
                static_cast<double>(fabric_result.ReaderStarvations);

            live_writer_mops[run] = live_result.ElapsedNs > 0.0
                ? static_cast<double>(live_result.WriterSuccess) *
                    ConcurrencyConfig::MILLION_OPERATIONS_PER_SECOND_FROM_NS /
                    live_result.ElapsedNs
                : 0.0;
            fabric_writer_mops[run] = fabric_result.ElapsedNs > 0.0
                ? static_cast<double>(fabric_result.WriterSuccess) *
                    ConcurrencyConfig::MILLION_OPERATIONS_PER_SECOND_FROM_NS /
                    fabric_result.ElapsedNs
                : 0.0;
        }
        SweepSampleDone();

        const double live_median = Median(live_ns);
        const double fabric_median = Median(fabric_ns);
        const double retries = Median(retry_rate);
        const double live_retries = Median(live_retry_rate);
        const double live_read_mops = live_median > 0.0
            ? ConcurrencyConfig::MILLION_OPERATIONS_PER_SECOND_FROM_NS / live_median
            : 0.0;
        const double fabric_read_mops = fabric_median > 0.0
            ? ConcurrencyConfig::MILLION_OPERATIONS_PER_SECOND_FROM_NS / fabric_median
            : 0.0;

        all_ok = all_ok && live_ok;
        std::cout
            << "    readers=" << std::setw(2) << reader_count
            << "  LiveGraph=" << std::setw(9) << std::fixed << std::setprecision(2)
            << live_median << " ns (" << std::setw(7) << live_read_mops << " M/s)"
            << "  Fabric=" << std::setw(9) << fabric_median
            << " ns (" << std::setw(7) << fabric_read_mops << " M/s)"
            << "  Fabric/LiveGraph=" << std::setw(6)
            << Ratio(fabric_median, live_median) << "x"
            << "  retry/read LG/Fabric=" << std::setprecision(4)
            << live_retries << "/" << retries
            << "  starve LG/Fabric=" << std::setprecision(0)
            << Median(live_starvations) << "/" << Median(fabric_starvations)
            << "  writers M/s LG/Fabric=" << std::setprecision(2)
            << Median(live_writer_mops) << "/" << Median(fabric_writer_mops)
            << "  correctness="
            << (live_correctness_ok && fabric_correctness_ok ? "PASS" : "FAIL")
            << "  progress="
            << (live_progress_ok && fabric_progress_ok ? "PASS" : "FAIL")
            << "  " << (live_ok ? "PASS" : "FAIL") << '\n';
    }
    return all_ok;
}

inline Result Run(
    const std::array<BenchmarkCase, 4u>& cases,
    std::size_t max_readers)
{
    Banner("TEST 3A - HOTSPOT STABLE READS WITH TWO ACTIVE WRITERS");
    std::cout
        << "Exactly two writers remain active: H toggles 0<->1 and V toggles 2<->3\n"
        << "on the same child. Readers sweep 1..(usable_threads-2). Each\n"
        << "LiveGraph stable read starts a fresh read-only snapshot transaction; SuperNova\n"
        << "uses its public stable FindParent path. Structural payload is kept minimal.\n";

    bool hotspot_ok = true;
    for (std::size_t i = 0u; i < cases.size(); ++i)
        hotspot_ok = RunCase(cases[i], false, max_readers, i + 1u) && hotspot_ok;
    std::cout << "\nTEST 3A OVERALL: " << (hotspot_ok ? "PASS" : "FAIL") << '\n';

    Banner("TEST 3B - DISTRIBUTED STABLE READS WITH TWO ACTIVE WRITERS");
    std::cout
        << "Two writers own separate child relations and mutate across up to 100 legal\n"
        << "parents. Readers sweep every count from 1 through usable_threads-2; APC RETRY\n"
        << "outcomes are retried and never counted as successful stable reads. Structural\n"
        << "payload remains one logical uint64_t on both backends.\n";

    bool distributed_ok = true;
    for (std::size_t i = 0u; i < cases.size(); ++i)
        distributed_ok = RunCase(cases[i], true, max_readers, i + 1u) && distributed_ok;
    std::cout << "\nTEST 3B OVERALL: " << (distributed_ok ? "PASS" : "FAIL") << '\n';

    const bool ok = hotspot_ok && distributed_ok;
    std::cout << "\nTEST 3 OVERALL: " << (ok ? "PASS" : "FAIL") << '\n';
    return ok ? Result::PASS : Result::FAIL;
}
} // namespace Test03

inline int Run(std::size_t lower_node_count = 100u,
               std::size_t higher_node_count = 10'000u,
               std::size_t lower_parent_capacity = 4u,
               std::size_t higher_parent_capacity = 32u,
               std::size_t usable_thread_count = 18u)
{
    PrintBenchmarkEnvironment();
    if (!ValidateRunArguments(lower_node_count, higher_node_count,
                              lower_parent_capacity, higher_parent_capacity,
                              usable_thread_count)) return 1;
    const auto cases = MakeBenchmarkCases(lower_node_count, higher_node_count,
        static_cast<std::uint8_t>(lower_parent_capacity),
        static_cast<std::uint8_t>(higher_parent_capacity));
    const auto workers = usable_thread_count - 2u;
    std::cout << "\nLiveGraph vs SuperNova; four measured samples/point; "
              << "writer/reader sweep 1.." << workers << "\n";
    const auto first = Test01::Run(cases);
    const auto second = Test02::Run(cases, workers);
    const auto third = Test03::Run(cases, workers);
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
                  std::size_t usable_thread_count = 18u)
{
    return Run(lower_node_count, higher_node_count, lower_parent_capacity,
               higher_parent_capacity, usable_thread_count);
}
} // namespace LiveGraphVsSuperNova
