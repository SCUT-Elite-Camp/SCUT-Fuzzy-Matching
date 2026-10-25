#pragma once
#include "types.h"
#include <cstddef>

namespace approx_psi {
std::size_t hammingDistance(const BitVector& a, const BitVector& b);
bool withinHammingDistance(const BitVector& a, const BitVector& b, std::size_t threshold);
} // namespace approx_psi
