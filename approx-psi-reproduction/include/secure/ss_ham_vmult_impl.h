#pragma once

// Fused Hamming-compare + vector-mask circuit, as a context-generic template.
//
// This deliberately builds ONE circuit fragment per candidate rather than two
// composed ones. The intermediate comparison bit stays a wire inside this
// function and is never handed back to the caller, so there is no point at which
// a garbled wire produced by one fragment is fed into a separately-constructed
// fragment -- which is what the previous split (ss_ham_compare_impl followed by
// ss_vmult_impl) required, and where the two parties' gate streams diverged.
//
// Returns c ? (a XOR b) : 0 elementwise, as wires. The caller reveals.

#include "emp-sh2pc/emp-sh2pc.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace approx_psi::secure {

// a_wires / b_wires are the circuit-side views of the two vectors (already fed in
// by the caller, four share inputs per bit). Returns one wire per element.
template <class Ctx>
std::vector<emp::Bit_T<Ctx>> ss_ham_and_vmult_impl(
    const std::vector<emp::Bit_T<Ctx>>& a_wires,
    const std::vector<emp::Bit_T<Ctx>>& b_wires,
    std::size_t threshold
)
{
    using B = emp::Bit_T<Ctx>;
    using U32 = emp::UInt_T<Ctx, 32>;

    Ctx* ctx = a_wires.at(0).context();

    // 1. diff[i] = a[i] XOR b[i]
    std::vector<B> diff;
    diff.reserve(a_wires.size());
    for (std::size_t i = 0; i < a_wires.size(); i++) {
        diff.push_back(a_wires[i] ^ b_wires[i]);
    }

    // 2. sum = number of set diff bits, via a serial adder.
    U32 sum = U32::constant(*ctx, 0);
    for (std::size_t i = 0; i < diff.size(); i++) {
        U32 zero = U32::constant(*ctx, 0);
        U32 one = U32::constant(*ctx, 1);
        sum = sum + zero.select(diff[i], one);
    }

    // 3. c = [sum <= threshold]
    U32 limit = U32::constant(*ctx, static_cast<std::uint64_t>(threshold));
    B c = sum <= limit;

    // 4. masked[i] = c AND diff[i], inside the same fragment.
    std::vector<B> masked;
    masked.reserve(diff.size());
    for (std::size_t i = 0; i < diff.size(); i++) {
        masked.push_back(c & diff[i]);
    }

    return masked;
}

} // namespace approx_psi::secure
