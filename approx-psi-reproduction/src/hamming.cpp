#include "hamming.h"
#include <stdexcept>

namespace approx_psi {

std::size_t hammingDistance(const BitVector& a, const BitVector& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Hamming vectors must have equal length");
    std::size_t distance = 0;
    for (std::size_t i = 0; i < a.size(); ++i) distance += (a[i] != b[i]);
    return distance;
}

bool withinHammingDistance(const BitVector& a, const BitVector& b, std::size_t threshold) {
    return hammingDistance(a, b) <= threshold;
}

} // namespace approx_psi
