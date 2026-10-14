#pragma once

// Garbled-circuit vector masking, as a context-generic template.
//
// Computes c ? v : 0 elementwise and returns the result as wires, WITHOUT
// revealing. The caller decides how to turn those wires into shares.
//
// Note this is the AND form (`c & v_i`), which is what the layered design needs:
// the masked vector is only meaningful when c is set, and the caller combines it
// with the shared comparison bit.

#include "emp-sh2pc/emp-sh2pc.h"

#include <cstddef>
#include <vector>

namespace approx_psi::secure {

// Elementwise (c AND v_i), over wires. Returns one wire per input element.
template <class Ctx>
std::vector<emp::Bit_T<Ctx>> ss_vmult_impl(
    const emp::Bit_T<Ctx>& c,
    const std::vector<emp::Bit_T<Ctx>>& v
)
{
    std::vector<emp::Bit_T<Ctx>> out;
    out.reserve(v.size());

    for (const auto& bit : v) {
        out.push_back(c & bit);
    }

    return out;
}

} // namespace approx_psi::secure
