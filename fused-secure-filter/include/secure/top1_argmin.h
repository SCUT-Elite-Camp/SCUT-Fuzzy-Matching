#pragma once

// Garbled-circuit Top-1 selection over a candidate group.
//
// Inputs
//   * client (EMP BOB)   : the query's 256-bit vector
//   * server (EMP ALICE) : per candidate, 256 bits + 16-bit identity + 16-bit
//                          database row index (tie-break only, never revealed)
//
// Circuit
//   1. 6-bit saturating Hamming counter.  Any distance above 63 saturates to
//      63, which is exact for both the <= threshold test (threshold <= 32) and
//      the argmin among acceptable candidates (their distances are < 63).
//   2. Lexicographic argmin over (distance, row index).
//   3. Output the winning identity, or 0xFFFF when the best distance exceeds
//      the threshold.  The identity is revealed to the client only.
//
// ``topk_group`` is the same circuit extended to the k smallest candidates:
// it keeps k sorted (distance, row index, identity) slots with an oblivious
// insertion pass per candidate and reveals the k identities, each masked to
// 0xFFFF when its own distance exceeds the threshold.

#include "secure/name_pipeline_types.h"

#include <emp-tool/circuits/bit.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

namespace approx_psi::secure::namepipeline {

template <class B>
B less_than_bits(const std::vector<B>& x, const std::vector<B>& y)
{
    typename B::context_type& ctx = *x[0].context();
    B lt = B::constant(ctx, false);
    B eq = B::constant(ctx, true);
    for (std::size_t k = x.size(); k-- > 0;)
    {
        const B x_zero_y_one = (!x[k]) & y[k];
        lt = lt | (eq & x_zero_y_one);
        eq = eq & (!(x[k] ^ y[k]));
    }
    return lt;
}

template <class B>
B equal_bits(const std::vector<B>& x, const std::vector<B>& y)
{
    typename B::context_type& ctx = *x[0].context();
    B eq = B::constant(ctx, true);
    for (std::size_t k = 0; k < x.size(); ++k)
    {
        eq = eq & (!(x[k] ^ y[k]));
    }
    return eq;
}

// 6-bit saturating popcount of (a XOR b).
template <class Ctx>
std::vector<emp::Bit_T<Ctx>> saturating_distance(
    const std::vector<emp::Bit_T<Ctx>>& a,
    const std::vector<emp::Bit_T<Ctx>>& b,
    Ctx& ctx)
{
    using B = emp::Bit_T<Ctx>;
    std::vector<B> distance(6);
    for (auto& bit : distance) bit = B::constant(ctx, false);

    for (std::size_t j = 0; j < a.size(); ++j)
    {
        const B diff = a[j] ^ b[j];
        const B full = distance[0] & distance[1] & distance[2] & distance[3] &
                       distance[4] & distance[5];
        B carry = diff & (!full);
        for (std::size_t k = 0; k < distance.size(); ++k)
        {
            const B old = distance[k];
            distance[k] = old ^ carry;
            carry = old & carry;
        }
    }
    return distance;
}

// Runs one group on both parties in lock-step.  Returns the identity on the
// client (EMP BOB) and std::nullopt on the server.
template <class Session>
std::optional<std::uint16_t> top1_group(
    Session& sess,
    bool am_alice,
    const std::array<std::uint8_t, kEllBytes>& query_bits,
    std::size_t candidate_count,
    const std::vector<Top1Candidate>* candidates,
    std::uint32_t threshold)
{
    using Ctx = typename Session::ctx_t;
    using B = emp::Bit_T<Ctx>;
    Ctx& ctx = sess.ctx();

    std::vector<B> query;
    query.reserve(kEllBits);
    for (std::size_t j = 0; j < kEllBits; ++j)
    {
        // The value is provided unconditionally so a plaintext ClearSession sees
        // it; in the 2PC session only BOB's value is used (ALICE's is ignored by
        // the input feed), and the server passes an all-zero placeholder.
        query.push_back(sess.template input<B>(emp::BOB, bit_at(query_bits, j)));
    }

    // Sentinel: distance 63, rid 0xFFFF, identity 0xFFFF.  Every real candidate
    // compares smaller, so no special-casing of the first row is needed.
    std::vector<B> best_distance(6), best_rid(16), best_identity(16);
    for (auto& bit : best_distance) bit = B::constant(ctx, true);
    for (auto& bit : best_rid) bit = B::constant(ctx, true);
    for (auto& bit : best_identity) bit = B::constant(ctx, true);

    for (std::size_t i = 0; i < candidate_count; ++i)
    {
        const Top1Candidate* candidate = nullptr;
        if (am_alice)
        {
            if (candidates == nullptr || i >= candidates->size())
            {
                throw std::runtime_error("top1_group: candidate data missing");
            }
            candidate = &(*candidates)[i];
        }

        std::vector<B> other(kEllBits);
        for (std::size_t j = 0; j < kEllBits; ++j)
        {
            other[j] = sess.template input<B>(
                emp::ALICE, candidate != nullptr && bit_at(candidate->bits, j));
        }
        const std::uint32_t identity_value = candidate != nullptr ? candidate->identity : 0;
        const std::uint32_t rid_value = candidate != nullptr ? candidate->rid : 0;

        std::vector<B> identity_bits(16), rid_bits(16);
        for (std::size_t k = 0; k < 16; ++k)
        {
            identity_bits[k] = sess.template input<B>(emp::ALICE, ((identity_value >> k) & 1u) != 0);
            rid_bits[k] = sess.template input<B>(emp::ALICE, ((rid_value >> k) & 1u) != 0);
        }

        const auto distance = saturating_distance(query, other, ctx);
        const B closer = less_than_bits(distance, best_distance);
        const B equal = equal_bits(distance, best_distance);
        const B lower_rid = less_than_bits(rid_bits, best_rid);
        const B take = closer | (equal & lower_rid);

        for (std::size_t k = 0; k < 6; ++k)
        {
            best_distance[k] = best_distance[k].select(take, distance[k]);
        }
        for (std::size_t k = 0; k < 16; ++k)
        {
            best_rid[k] = best_rid[k].select(take, rid_bits[k]);
            best_identity[k] = best_identity[k].select(take, identity_bits[k]);
        }
    }

    std::vector<B> limit(6);
    for (std::size_t k = 0; k < 6; ++k)
    {
        limit[k] = B::constant(ctx, ((threshold >> k) & 1u) != 0);
    }
    const B accepted =
        less_than_bits(best_distance, limit) | equal_bits(best_distance, limit);

    std::vector<B> output(16);
    for (std::size_t k = 0; k < 16; ++k)
    {
        // select(sel, t) = sel ? t : this, so start from the sentinel and take
        // the winning identity only when the candidate was accepted.
        output[k] = B::constant(ctx, ((kNoIdentity >> k) & 1u) != 0)
                        .select(accepted, best_identity[k]);
    }

    // Debug aid: NPSI_DEBUG=1 opens the internal state to both parties.
    if (std::getenv("NPSI_DEBUG") != nullptr)
    {
        std::uint32_t debug_distance = 0;
        for (std::size_t k = 0; k < 6; ++k)
        {
            const auto bit = sess.reveal(best_distance[k], emp::PUBLIC);
            if (bit.has_value() && *bit) debug_distance |= (1u << k);
        }
        std::uint32_t debug_identity = 0;
        for (std::size_t k = 0; k < 16; ++k)
        {
            const auto bit = sess.reveal(best_identity[k], emp::PUBLIC);
            if (bit.has_value() && *bit) debug_identity |= (1u << k);
        }
        const auto debug_accepted = sess.reveal(accepted, emp::PUBLIC);
        std::cout << "[top1-debug] candidates=" << candidate_count
                  << " best_distance=" << debug_distance
                  << " best_identity=" << debug_identity
                  << " accepted=" << (debug_accepted.has_value() && *debug_accepted ? 1 : 0)
                  << std::endl;
    }

    // Only the client learns the identity; the server still runs the reveal.
    std::uint16_t value = 0;
    bool complete = true;
    for (std::size_t k = 0; k < 16; ++k)
    {
        const auto bit = sess.reveal(output[k], emp::BOB);
        if (bit.has_value())
        {
            if (*bit) value |= static_cast<std::uint16_t>(1u << k);
        }
        else
        {
            complete = false;
        }
    }
    if (!complete) return std::nullopt;
    return std::optional<std::uint16_t>(value);
}

// Top-K variant of the circuit above.  Each candidate is inserted into a
// sorted list of k (distance, row index, identity) slots; the insertion is
// data-oblivious (a fixed k-step compare-and-swap chain), so the wire count
// depends only on ``candidate_count`` and ``k``.  Returns the k identities in
// ascending (distance, row index) order on the client (BOB); a slot whose
// distance exceeds ``threshold`` is returned as kNoIdentity.  On ALICE the
// returned values are meaningless (kNoIdentity) and are discarded.
template <class Session>
std::vector<std::uint16_t> topk_group(
    Session& sess,
    bool am_alice,
    const std::array<std::uint8_t, kEllBytes>& query_bits,
    std::size_t candidate_count,
    const std::vector<Top1Candidate>* candidates,
    std::uint32_t threshold,
    std::size_t k)
{
    using Ctx = typename Session::ctx_t;
    using B = emp::Bit_T<Ctx>;
    Ctx& ctx = sess.ctx();

    if (k == 0)
    {
        throw std::runtime_error("topk_group: k must be positive");
    }

    std::vector<B> query;
    query.reserve(kEllBits);
    for (std::size_t j = 0; j < kEllBits; ++j)
    {
        query.push_back(sess.template input<B>(emp::BOB, bit_at(query_bits, j)));
    }

    // Slot j holds the j-th smallest candidate seen so far.  All slots start
    // at the sentinel (distance 63, rid 0xFFFF, identity 0xFFFF).
    std::vector<std::vector<B>> slot_distance(k, std::vector<B>(6));
    std::vector<std::vector<B>> slot_rid(k, std::vector<B>(16));
    std::vector<std::vector<B>> slot_identity(k, std::vector<B>(16));
    for (std::size_t s = 0; s < k; ++s)
    {
        for (auto& bit : slot_distance[s]) bit = B::constant(ctx, true);
        for (auto& bit : slot_rid[s]) bit = B::constant(ctx, true);
        for (auto& bit : slot_identity[s]) bit = B::constant(ctx, true);
    }

    for (std::size_t i = 0; i < candidate_count; ++i)
    {
        const Top1Candidate* candidate = nullptr;
        if (am_alice)
        {
            if (candidates == nullptr || i >= candidates->size())
            {
                throw std::runtime_error("topk_group: candidate data missing");
            }
            candidate = &(*candidates)[i];
        }

        std::vector<B> other(kEllBits);
        for (std::size_t j = 0; j < kEllBits; ++j)
        {
            other[j] = sess.template input<B>(
                emp::ALICE, candidate != nullptr && bit_at(candidate->bits, j));
        }
        const std::uint32_t identity_value = candidate != nullptr ? candidate->identity : 0;
        const std::uint32_t rid_value = candidate != nullptr ? candidate->rid : 0;

        std::vector<B> carry_identity(16), carry_rid(16);
        for (std::size_t b = 0; b < 16; ++b)
        {
            carry_identity[b] = sess.template input<B>(
                emp::ALICE, ((identity_value >> b) & 1u) != 0);
            carry_rid[b] = sess.template input<B>(
                emp::ALICE, ((rid_value >> b) & 1u) != 0);
        }
        std::vector<B> carry_distance = saturating_distance(query, other, ctx);

        // Lexicographic (distance, row index) comparison of two slots.
        auto lex_less = [](const std::vector<B>& left_distance,
                           const std::vector<B>& left_rid,
                           const std::vector<B>& right_distance,
                           const std::vector<B>& right_rid) {
            const B closer = less_than_bits(left_distance, right_distance);
            const B equal = equal_bits(left_distance, right_distance);
            const B lower_rid = less_than_bits(left_rid, right_rid);
            return closer | (equal & lower_rid);
        };

        // Swap slot ``s`` with the carry when ``take`` is true (oblivious).
        auto swap_carry = [&](std::size_t s, const B& take) {
            for (std::size_t b = 0; b < 6; ++b)
            {
                const B old_slot = slot_distance[s][b];
                const B old_carry = carry_distance[b];
                slot_distance[s][b] = old_slot.select(take, old_carry);
                carry_distance[b] = old_carry.select(take, old_slot);
            }
            for (std::size_t b = 0; b < 16; ++b)
            {
                const B old_rid = slot_rid[s][b];
                const B old_carry_rid = carry_rid[b];
                slot_rid[s][b] = old_rid.select(take, old_carry_rid);
                carry_rid[b] = old_carry_rid.select(take, old_rid);
                const B old_identity = slot_identity[s][b];
                const B old_carry_identity = carry_identity[b];
                slot_identity[s][b] = old_identity.select(take, old_carry_identity);
                carry_identity[b] = old_carry_identity.select(take, old_identity);
            }
        };

        // The candidate enters at the worst slot only if it beats it.
        swap_carry(k - 1, lex_less(
            carry_distance, carry_rid, slot_distance[k - 1], slot_rid[k - 1]));

        // Bubble the new entry leftwards to restore ascending order.
        for (std::size_t s = k - 1; s-- > 0;)
        {
            const B take = lex_less(
                slot_distance[s + 1], slot_rid[s + 1],
                slot_distance[s], slot_rid[s]);
            for (std::size_t b = 0; b < 6; ++b)
            {
                const B left = slot_distance[s][b];
                const B right = slot_distance[s + 1][b];
                slot_distance[s][b] = left.select(take, right);
                slot_distance[s + 1][b] = right.select(take, left);
            }
            for (std::size_t b = 0; b < 16; ++b)
            {
                {
                    const B left = slot_rid[s][b];
                    const B right = slot_rid[s + 1][b];
                    slot_rid[s][b] = left.select(take, right);
                    slot_rid[s + 1][b] = right.select(take, left);
                }
                {
                    const B left = slot_identity[s][b];
                    const B right = slot_identity[s + 1][b];
                    slot_identity[s][b] = left.select(take, right);
                    slot_identity[s + 1][b] = right.select(take, left);
                }
            }
        }
    }

    std::vector<B> limit(6);
    for (std::size_t b = 0; b < 6; ++b)
    {
        limit[b] = B::constant(ctx, ((threshold >> b) & 1u) != 0);
    }

    std::vector<std::uint16_t> output;
    output.reserve(k);
    bool complete = true;
    for (std::size_t s = 0; s < k; ++s)
    {
        const B accepted = less_than_bits(slot_distance[s], limit) |
                           equal_bits(slot_distance[s], limit);

        std::vector<B> masked(16);
        for (std::size_t b = 0; b < 16; ++b)
        {
            masked[b] = B::constant(ctx, ((kNoIdentity >> b) & 1u) != 0)
                            .select(accepted, slot_identity[s][b]);
        }

        std::uint16_t value = 0;
        for (std::size_t b = 0; b < 16; ++b)
        {
            const auto bit = sess.reveal(masked[b], emp::BOB);
            if (bit.has_value())
            {
                if (*bit) value |= static_cast<std::uint16_t>(1u << b);
            }
            else
            {
                // ALICE runs the reveal but learns nothing; its return value is
                // discarded by the caller.  Keep the wire schedule unchanged.
                complete = false;
            }
        }
        output.push_back(value);
    }
    if (!complete) return {};
    return output;
}

} // namespace approx_psi::secure::namepipeline
