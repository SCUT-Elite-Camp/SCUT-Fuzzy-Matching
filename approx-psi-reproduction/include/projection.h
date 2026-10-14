#pragma once
#include "types.h"
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace approx_psi {

struct ProjectionRepresentative {
    Element original;
    BitVector projected;
};

BitIndex sampleProjection(std::size_t length, double p, std::mt19937_64& rng);
BitVector project(const BitVector& value, const BitIndex& indices);
std::string bitsToString(const BitVector& bits);

// Algorithm 1 line 4: for every repeated projection value,
// randomly keep exactly one representative and attach the original element as payload.
std::unordered_map<std::string, ProjectionRepresentative>
projectSetWithRandomRepresentative(const std::vector<Element>& input,
                                   const BitIndex& projection,
                                   std::mt19937_64& rng);

} // namespace approx_psi
