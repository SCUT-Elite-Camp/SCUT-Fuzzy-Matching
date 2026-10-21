#pragma once

// Garbled-circuit Hamming comparison, as a context-generic template.
//
// The circuit takes this party's wire-level view of both vectors and returns the
// comparison bit as a wire, WITHOUT revealing it. Revealing is the caller's
// decision: a XOR reveal turns the wire into this party's share, and the two
// parties' shares XOR to the cleartext.
//
// Keeping the reveal out of the circuit is what makes the two parties' EMP call
// sequences symmetric: the circuit itself only issues the input exchanges, and
// every party issues exactly the same number of them.

#include "emp-sh2pc/emp-sh2pc.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace approx_psi::secure {

// Returns [hamming(a, b) <= threshold] as a wire over the session's context.
//
// `a_wires` and `b_wires` are the circuit-side views of the two vectors: the
// caller has already fed its shares in (see the callers for how the XOR shares
// are split across ALICE/BOB ownership). Both must have the same length.
template <class Ctx>
emp::Bit_T<Ctx> ss_ham_compare_impl(
    const std::vector<emp::Bit_T<Ctx>>& a_wires,
    const std::vector<emp::Bit_T<Ctx>>& b_wires,
    std::size_t threshold
)
{
    using B = emp::Bit_T<Ctx>;
    using U32 = emp::UInt_T<Ctx, 32>;

    Ctx& ctx = *a_wires.at(0).context();

    const std::size_t length = a_wires.size();

    // sum += (a_i XOR b_i) for each position.
    U32 sum = U32::constant(ctx, 0);
    for (std::size_t i = 0; i < length; i++) {
        B diff = a_wires[i] ^ b_wires[i];

        U32 zero = U32::constant(ctx, 0);
        U32 one = U32::constant(ctx, 1);

        sum = sum + zero.select(diff, one);
    }

    U32 limit = U32::constant(ctx, static_cast<std::uint64_t>(threshold));

    return sum <= limit;
}

} // namespace approx_psi::secure
