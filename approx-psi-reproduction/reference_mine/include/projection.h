#pragma once
#include "types.h"
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

struct ProjectionRepresentative {
    Element original;
    BitVector projected;
};

inline std::vector<std::size_t> sampleProjection(
    std::size_t length,
    double p,
    std::mt19937_64& rng)
{
    std::bernoulli_distribution keep(p);
    std::vector<std::size_t> indices;
    for (std::size_t i = 0; i < length; ++i) {
        if (keep(rng)) indices.push_back(i);
    }
    if (indices.empty() && length > 0) {
        std::uniform_int_distribution<std::size_t> pick(0, length - 1);
        indices.push_back(pick(rng));
    }
    return indices;
}

inline BitVector project(const BitVector& value, const std::vector<std::size_t>& indices) {
    BitVector result;
    result.reserve(indices.size());
    for (auto index : indices) {
        result.push_back(value[index]);
    }
    return result;
}

inline std::string bitsToString(const BitVector& bits) {
    std::string result;
    result.reserve(bits.size());
    for (auto bit : bits) result.push_back(bit ? '1' : '0');
    return result;
}

inline std::unordered_map<std::string, ProjectionRepresentative>
projectSetWithRandomRepresentative(
    const std::vector<Element>& input,
    const std::vector<std::size_t>& projection,
    std::mt19937_64& rng)
{
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
