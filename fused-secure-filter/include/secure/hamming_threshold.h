#pragma once

// Threshold test for the Hamming weight of a secret-shared difference vector.
//
// The old path accumulated every XOR bit into a 16-bit UInt with a 16-bit mux
// per bit, which costs ~31 AND gates per input bit.  The circuit below is
// specific to the "small threshold" regime used by Approx-PSI (d = 4 in the
// paper): counts saturate at 7, so three counter bits are enough, and the
// saturation value (7) is still strictly greater than any accepted threshold
// d <= 6.  Both shapes below are data independent, so nothing about the
// distance or the inputs leaks through the gate pattern.

#include <emp-sh2pc/emp-sh2pc.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace approx_psi::secure {

// Saturating binary counter, generic in the threshold.  Cost is 2*W AND gates
// per input bit, where W = bits needed to represent threshold + 1.
template <class Ctx>
emp::Bit_T<Ctx> hamming_leq_saturating(
    const std::vector<emp::Bit_T<Ctx>>& diff,
    std::size_t threshold)
{
    using B = emp::Bit_T<Ctx>;
    if (diff.empty()) {
        throw std::invalid_argument("hamming_leq_saturating: empty diff");
    }
    Ctx& ctx = *diff.front().context();

    if (threshold >= diff.size()) {
        return B::constant(ctx, true);
    }

    const std::size_t cap = threshold + 1;
    std::size_t width = 1;
    while ((std::size_t{1} << width) <= cap) ++width;

    // Saturate at the all-ones value 2^width - 1, which is > threshold.
    std::vector<B> count;
    count.reserve(width);
    for (std::size_t k = 0; k < width; ++k) {
        count.push_back(B::constant(ctx, false));
    }

    for (const B& bit : diff) {
        B frozen = B::constant(ctx, true);
        for (std::size_t k = 0; k < width; ++k) frozen = frozen & count[k];
        const B inc = bit & !frozen;

        B carry = inc;
        for (std::size_t k = 0; k < width; ++k) {
            const B sum = count[k] ^ carry;
            carry = count[k] & carry;
            count[k] = sum;
        }
    }

    // count <= threshold, constant-threshold comparator.
    B gt = B::constant(ctx, false);
    B eq = B::constant(ctx, true);
    for (std::size_t k = width; k-- > 0;) {
        const bool want = ((threshold >> k) & 1u) != 0;
        if (want) {
            eq = eq & count[k];
        } else {
            gt = gt | (eq & count[k]);
            eq = eq & !count[k];
        }
    }
    return !gt;
}

namespace detail {

// Adds two 3-bit counts (each in [0,7]) and saturates the result at 7.
// Returns the three result bits in `out`.
template <class Ctx>
void add3_saturating(
    const emp::Bit_T<Ctx>& a0, const emp::Bit_T<Ctx>& a1, const emp::Bit_T<Ctx>& a2,
    const emp::Bit_T<Ctx>& b0, const emp::Bit_T<Ctx>& b1, const emp::Bit_T<Ctx>& b2,
    emp::Bit_T<Ctx>& o0, emp::Bit_T<Ctx>& o1, emp::Bit_T<Ctx>& o2)
{
    using B = emp::Bit_T<Ctx>;
    Ctx& ctx = *a0.context();

    const B s0 = a0 ^ b0;
    const B k1 = a0 & b0;

    const B s1 = a1 ^ b1 ^ k1;
    const B k2 = (a1 & b1) | (k1 & (a1 ^ b1));

    const B s2 = a2 ^ b2 ^ k2;
    const B k3 = (a2 & b2) | (k2 & (a2 ^ b2));

    // sum >= 5  <=>  bit3 set, or (bit2 set and any lower bit set)
    const B ge5 = k3 | (s2 & (s1 | s0));

    o0 = s0 | ge5;
    o1 = s1 | ge5;
    o2 = s2 | ge5;
    (void)ctx;
}

// Exact 3-bit Hamming weight of up to four bits.
template <class Ctx>
void leaf4(
    const emp::Bit_T<Ctx>* b, std::size_t n,
    emp::Bit_T<Ctx>& o0, emp::Bit_T<Ctx>& o1, emp::Bit_T<Ctx>& o2)
{
    using B = emp::Bit_T<Ctx>;
    Ctx& ctx = *b[0].context();

    B x0 = B::constant(ctx, false), x1 = B::constant(ctx, false);
    B x2 = B::constant(ctx, false), x3 = B::constant(ctx, false);
    if (n > 0) x0 = b[0];
    if (n > 1) x1 = b[1];
    if (n > 2) x2 = b[2];
    if (n > 3) x3 = b[3];

    const B p = x0 ^ x1;
    const B q = x0 & x1;
    const B r = x2 ^ x3;
    const B s = x2 & x3;

    o0 = p ^ r;
    const B k = p & r;
    o1 = q ^ s ^ k;
    o2 = (q & s) | (k & (q ^ s));
}

} // namespace detail

// Tree of 4-bit leaves and saturating 3-bit merges (previous fast path).
template <class Ctx>
emp::Bit_T<Ctx> hamming_leq_tree(
    const std::vector<emp::Bit_T<Ctx>>& diff,
    std::size_t threshold)
{
    using B = emp::Bit_T<Ctx>;
    if (diff.empty()) {
        throw std::invalid_argument("hamming_leq_tree: empty diff");
    }
    Ctx& ctx = *diff.front().context();

    if (threshold >= diff.size()) return B::constant(ctx, true);
    if (threshold > 6) return hamming_leq_saturating(diff, threshold);

    struct Node { B b0, b1, b2; };

    std::vector<Node> level;
    level.reserve(diff.size() / 4 + 1);
    for (std::size_t i = 0; i < diff.size(); i += 4) {
        const std::size_t n = std::min<std::size_t>(4, diff.size() - i);
        Node node{B::constant(ctx, false), B::constant(ctx, false), B::constant(ctx, false)};
        detail::leaf4<Ctx>(diff.data() + i, n, node.b0, node.b1, node.b2);
        level.push_back(node);
    }

    while (level.size() > 1) {
        std::vector<Node> next;
        next.reserve(level.size() / 2 + 1);
        for (std::size_t i = 0; i + 1 < level.size(); i += 2) {
            Node node{B::constant(ctx, false), B::constant(ctx, false), B::constant(ctx, false)};
            detail::add3_saturating<Ctx>(
                level[i].b0, level[i].b1, level[i].b2,
                level[i + 1].b0, level[i + 1].b1, level[i + 1].b2,
                node.b0, node.b1, node.b2);
            next.push_back(node);
        }
        if (level.size() % 2 == 1) next.push_back(level.back());
        level.swap(next);
    }

    // level[0] is a 3-bit count saturated at 7, which is > threshold.
    const Node& c = level[0];
    B gt = B::constant(ctx, false);
    B eq = B::constant(ctx, true);
    for (std::size_t k = 3; k-- > 0;) {
        const bool want = ((threshold >> k) & 1u) != 0;
        const B bit = (k == 0) ? c.b0 : (k == 1 ? c.b1 : c.b2);
        if (want) {
            eq = eq & bit;
        } else {
            gt = gt | (eq & bit);
            eq = eq & !bit;
        }
    }
    return !gt;
}

// Carry-save (Wallace) popcount truncated at weight 8.
//
// Bits are pushed through 3:2 and 2:2 compressors at weights 1, 2 and 4.
// Carries out of weight 4 mean the total count is at least 8, which is enough
// to answer "count > d" for any d <= 6, so they are OR-ed into a sticky reject
// bit instead of being propagated.  That keeps the whole circuit at ~2.7 AND
// gates per compared bit (about 347 for 128 bits), independent of the vector
// length, and avoids the saturating-adder overhead of the tree variant.
template <class Ctx>
emp::Bit_T<Ctx> hamming_leq_wallace(
    const std::vector<emp::Bit_T<Ctx>>& diff,
    std::size_t threshold)
{
    using B = emp::Bit_T<Ctx>;
    if (diff.empty()) {
        throw std::invalid_argument("hamming_leq_wallace: empty diff");
    }
    Ctx& ctx = *diff.front().context();

    if (threshold >= diff.size()) return B::constant(ctx, true);
    if (threshold > 6) return hamming_leq_saturating(diff, threshold);

    constexpr std::size_t kWeights = 3; // weights 1, 2, 4; weight 8 => reject
    std::vector<std::vector<B>> bucket(kWeights);
    bucket[0] = diff;
    std::vector<B> reject;

    for (std::size_t w = 0; w < kWeights; ++w) {
        std::vector<B>& cur = bucket[w];
        while (cur.size() >= 2) {
            B carry = B::constant(ctx, false);
            if (cur.size() >= 3) {
                const B a = cur.back(); cur.pop_back();
                const B b = cur.back(); cur.pop_back();
                const B c = cur.back(); cur.pop_back();
                cur.push_back(a ^ b ^ c);
                carry = (a & b) | (c & (a ^ b));
            } else {
                const B a = cur.back(); cur.pop_back();
                const B b = cur.back(); cur.pop_back();
                cur.push_back(a ^ b);
                carry = a & b;
            }
            if (w + 1 < kWeights) bucket[w + 1].push_back(carry);
            else reject.push_back(carry);
        }
    }

    auto bit_at = [&](std::size_t w) {
        return bucket[w].empty() ? B::constant(ctx, false) : bucket[w].front();
    };
    const B b1 = bit_at(0);
    const B b2 = bit_at(1);
    const B b4 = bit_at(2);

    // count <= threshold, with count in [0,7] unless `reject` is set.
    B gt = B::constant(ctx, false);
    B eq = B::constant(ctx, true);
    const B bits[kWeights] = {b1, b2, b4};
    for (std::size_t k = kWeights; k-- > 0;) {
        const bool want = ((threshold >> k) & 1u) != 0;
        if (want) {
            eq = eq & bits[k];
        } else {
            gt = gt | (eq & bits[k]);
            eq = eq & !bits[k];
        }
    }

    B any_reject = B::constant(ctx, false);
    for (const B& r : reject) any_reject = any_reject | r;

    return !(gt | any_reject);
}

// Dispatch used by the secure filter: Wallace for the small-threshold regime,
// generic saturating counter otherwise.
template <class Ctx>
emp::Bit_T<Ctx> hamming_leq_small(
    const std::vector<emp::Bit_T<Ctx>>& diff,
    std::size_t threshold)
{
    return hamming_leq_wallace<Ctx>(diff, threshold);
}

} // namespace approx_psi::secure
