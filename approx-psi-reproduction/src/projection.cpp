#include "projection.h"
#include <stdexcept>
#include <unordered_map>

namespace approx_psi {

BitIndex sampleProjection(std::size_t length, double p, std::mt19937_64& rng) {
    if (length == 0) throw std::invalid_argument("projection length must be positive");
    if (p < 0.0 || p > 1.0) throw std::invalid_argument("p must be in [0,1]");

    std::bernoulli_distribution keep(p);
    BitIndex indices;
    indices.reserve(length);
    for (std::size_t i = 0; i < length; ++i) {
        if (keep(rng)) indices.push_back(i);
    }
    // Empty I is mathematically allowed; the empty string is then one shared bucket.
    return indices;
}

BitVector project(const BitVector& value, const BitIndex& indices) {
    BitVector result;
    result.reserve(indices.size());
    for (const auto index : indices) {
        if (index >= value.size()) throw std::out_of_range("projection index out of range");
        result.push_back(value[index]);
    }
    return result;
}

std::string bitsToString(const BitVector& bits) {
    std::string result;
    result.reserve(bits.size());
    for (const auto bit : bits) result.push_back(bit ? '1' : '0');
    return result;
}

std::unordered_map<std::string, ProjectionRepresentative>
projectSetWithRandomRepresentative(const std::vector<Element>& input,
                                   const BitIndex& projection,
                                   std::mt19937_64& rng) {
    std::unordered_map<std::string, std::vector<ProjectionRepresentative>> buckets;
    for (const auto& item : input) {
        auto projected = project(item.bits, projection);
        auto key = bitsToString(projected);
        buckets[key].push_back(ProjectionRepresentative{item, std::move(projected)});
    }

    std::unordered_map<std::string, ProjectionRepresentative> representatives;
    representatives.reserve(buckets.size());
    for (auto& [key, bucket] : buckets) {
        std::uniform_int_distribution<std::size_t> pick(0, bucket.size() - 1);
        representatives.emplace(key, std::move(bucket[pick(rng)]));
    }
    return representatives;
}

} // namespace approx_psi
