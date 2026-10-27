// Plaintext oracle test for the Top-1 garbled-circuit logic.
//
// ClearSession evaluates the same `top1_group` template that the two-party
// pipeline runs under EMP, so this pins the circuit semantics (saturating
// popcount, lexicographic argmin on (distance, rid), threshold accept) without
// needing two processes or a network.

#include "secure/top1_argmin.h"
#include "secure/name_pipeline_types.h"

#include <emp-tool/ir/session/clear_session.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <random>
#include <vector>

using namespace approx_psi::secure;
using namespace approx_psi::secure::namepipeline;

namespace {

int failures = 0;

#define CHECK(condition, message)                                              \
    do {                                                                       \
        if (!(condition)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, message);      \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

using Bits = std::array<std::uint8_t, kEllBytes>;

Bits make_bits(std::uint64_t seed)
{
    std::mt19937_64 rng(seed);
    Bits bits{};
    for (auto& byte : bits)
    {
        byte = static_cast<std::uint8_t>(rng() & 0xFFu);
    }
    return bits;
}

// Flip the first `count` bit positions (same layout as bit_at: big-endian).
Bits flipped(const Bits& in, int count)
{
    Bits out = in;
    for (int j = 0; j < count; ++j)
    {
        out[j >> 3] ^= static_cast<std::uint8_t>(1u << (7 - (j & 7)));
    }
    return out;
}

std::optional<std::uint16_t> run_top1(
    const Bits& query,
    const std::vector<Top1Candidate>& candidates,
    std::uint32_t threshold)
{
    emp::ClearSession session;
    return top1_group(
        session, /*am_alice=*/true, query, candidates.size(), &candidates, threshold);
}

Top1Candidate candidate(const Bits& bits, std::uint16_t identity, std::uint16_t rid)
{
    return Top1Candidate{bits, identity, rid};
}

void test_argmin_picks_smallest_distance()
{
    const Bits query = make_bits(1);
    const std::vector<Top1Candidate> candidates = {
        candidate(flipped(query, 7), 11, 4),
        candidate(flipped(query, 3), 22, 8),
        candidate(flipped(query, 3), 33, 2),
    };
    const auto result = run_top1(query, candidates, 32);
    CHECK(result.has_value() && *result == 33,
          "distance-3 tie must break on the lower row id (identity 33)");
}

void test_threshold_boundary()
{
    const Bits query = make_bits(2);
    // 32 differing bits is exactly the threshold: accepted.
    const std::vector<Top1Candidate> accepted = {candidate(flipped(query, 32), 7, 0)};
    const auto ok = run_top1(query, accepted, 32);
    CHECK(ok.has_value() && *ok == 7, "distance == threshold must be accepted");

    // 33 differing bits: rejected, sentinel identity.
    const std::vector<Top1Candidate> rejected = {candidate(flipped(query, 33), 7, 0)};
    const auto no = run_top1(query, rejected, 32);
    CHECK(no.has_value() && *no == kNoIdentity,
          "distance > threshold must return the sentinel identity");
}

void test_saturating_counter()
{
    const Bits query = make_bits(3);
    // 200 differences saturate the 6-bit counter at 63, still > 32.
    const std::vector<Top1Candidate> candidates = {candidate(flipped(query, 200), 5, 0)};
    const auto result = run_top1(query, candidates, 32);
    CHECK(result.has_value() && *result == kNoIdentity,
          "large distance must saturate and be rejected");

    // A near candidate must still win against a saturated one.
    const std::vector<Top1Candidate> mixed = {
        candidate(flipped(query, 200), 5, 0),
        candidate(flipped(query, 4), 9, 1),
    };
    const auto better = run_top1(query, mixed, 32);
    CHECK(better.has_value() && *better == 9,
          "saturation must not disturb the argmin against a close candidate");
}

void test_empty_group()
{
    const Bits query = make_bits(4);
    const auto result = run_top1(query, {}, 32);
    CHECK(result.has_value() && *result == kNoIdentity,
          "an empty candidate set must return the sentinel identity");
}

void test_threshold_zero()
{
    const Bits query = make_bits(5);
    const std::vector<Top1Candidate> candidates = {
        candidate(flipped(query, 0), 77, 3),
        candidate(flipped(query, 1), 88, 1),
    };
    const auto exact = run_top1(query, candidates, 0);
    CHECK(exact.has_value() && *exact == 77, "threshold 0 accepts only an exact match");

    const auto none = run_top1(query, {candidate(flipped(query, 1), 88, 1)}, 0);
    CHECK(none.has_value() && *none == kNoIdentity, "threshold 0 rejects a 1-bit difference");
}

void test_bit_helpers()
{
    const Bits query = make_bits(6);
    emp::ClearSession session;
    using Ctx = emp::ClearSession::ctx_t;
    using B = emp::Bit_T<Ctx>;
    Ctx& ctx = session.ctx();

    auto feed = [&](const Bits& bits) {
        std::vector<B> out;
        for (std::size_t j = 0; j < kEllBits; ++j)
        {
            out.push_back(session.input<B>(emp::ALICE, bit_at(bits, j)));
        }
        return out;
    };
    const auto a = feed(query);
    const auto b = feed(flipped(query, 10));
    const auto distance = saturating_distance(a, b, ctx);
    int value = 0;
    for (int k = 0; k < 6; ++k)
    {
        if (*session.reveal(distance[static_cast<std::size_t>(k)], emp::PUBLIC))
        {
            value |= 1 << k;
        }
    }
    CHECK(value == 10, "saturating popcount must equal the Hamming distance");
}

} // namespace

int main()
{
    test_argmin_picks_smallest_distance();
    test_threshold_boundary();
    test_saturating_counter();
    test_empty_group();
    test_threshold_zero();
    test_bit_helpers();
    if (failures != 0)
    {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("test_name_top1_clear: all checks passed\n");
    return 0;
}
