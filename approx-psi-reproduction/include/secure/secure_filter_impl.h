#pragma once

// Fused secure filter for one RsCpsi output bin:
//
//   hit    = flag_A XOR flag_B
//   ham_ok = [H(a,b) <= d]
//   keep   = hit AND valid_A AND valid_B AND ham_ok
//   outA   = keep * a
//   outB   = keep * b
//
// Nothing about the PSI hit or the peer vector is opened before this circuit.
// The caller may reveal outA only to Alice and outB only to Bob, matching the
// "open own payload" semantics of Algorithm 1.

#include <emp-sh2pc/emp-sh2pc.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace approx_psi::secure {

template <class Ctx>
struct SecureFilterWires
{
    emp::Bit_T<Ctx> keep;
    std::vector<emp::Bit_T<Ctx>> masked_a;
    std::vector<emp::Bit_T<Ctx>> masked_b;
};

template <class Ctx>
SecureFilterWires<Ctx> secure_filter_impl(
    const emp::Bit_T<Ctx>& flag_a,
    const emp::Bit_T<Ctx>& flag_b,
    const emp::Bit_T<Ctx>& valid_a,
    const emp::Bit_T<Ctx>& valid_b,
    const std::vector<emp::Bit_T<Ctx>>& a_wires,
    const std::vector<emp::Bit_T<Ctx>>& b_wires,
    std::size_t threshold
)
{
    using B = emp::Bit_T<Ctx>;
    using U32 = emp::UInt_T<Ctx, 32>;

    if (a_wires.size() != b_wires.size() || a_wires.empty()) {
        throw std::runtime_error("secure_filter_impl: invalid vector lengths");
    }

    Ctx* ctx = a_wires.front().context();

    B hit = flag_a ^ flag_b;

    std::vector<B> diff;
    diff.reserve(a_wires.size());
    for (std::size_t i = 0; i < a_wires.size(); ++i) {
        diff.push_back(a_wires[i] ^ b_wires[i]);
    }

    U32 distance = U32::constant(*ctx, 0);
    for (const B& bit : diff) {
        U32 zero = U32::constant(*ctx, 0);
        U32 one = U32::constant(*ctx, 1);
        distance = distance + zero.select(bit, one);
    }

    U32 limit = U32::constant(*ctx, static_cast<std::uint64_t>(threshold));
    B ham_ok = distance <= limit;

    B keep = hit & valid_a & valid_b & ham_ok;

    std::vector<B> masked_a;
    std::vector<B> masked_b;
    masked_a.reserve(a_wires.size());
    masked_b.reserve(b_wires.size());

    for (std::size_t i = 0; i < a_wires.size(); ++i) {
        masked_a.push_back(keep & a_wires[i]);
        masked_b.push_back(keep & b_wires[i]);
    }

    return SecureFilterWires<Ctx>{keep, std::move(masked_a), std::move(masked_b)};
}

} // namespace approx_psi::secure
