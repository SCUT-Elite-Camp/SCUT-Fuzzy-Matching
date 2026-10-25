#include "dataset.h"
#include "hamming.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace approx_psi {
namespace {

BitVector randomVector(std::size_t length, std::mt19937_64& rng) {
    std::bernoulli_distribution bit(0.5);
    BitVector out(length);
    for (auto& v : out) v = static_cast<std::uint8_t>(bit(rng));
    return out;
}

BitVector mutateClose(const BitVector& source, std::size_t d, std::mt19937_64& rng) {
    BitVector out = source;
    std::vector<std::size_t> positions(source.size());
    std::iota(positions.begin(), positions.end(), 0);
    std::shuffle(positions.begin(), positions.end(), rng);

    // Use at least one flip when possible so the test exercises approximate, not only exact, matches.
    std::uniform_int_distribution<std::size_t> count_dist(1, std::min(d, source.size()));
    const auto count = count_dist(rng);
    for (std::size_t i = 0; i < count; ++i) out[positions[i]] ^= 1U;
    return out;
}

bool farFromAll(const BitVector& candidate,
                const std::vector<Element>& set,
                std::size_t far_threshold) {
    for (const auto& x : set) {
        if (hammingDistance(candidate, x.bits) < far_threshold) return false;
    }
    return true;
}

bool farFromAllExcept(const BitVector& candidate,
                      const std::vector<Element>& set,
                      std::size_t except_id,
                      std::size_t far_threshold) {
    for (const auto& x : set) {
        if (x.id == except_id) continue;
        if (hammingDistance(candidate, x.bits) < far_threshold) return false;
    }
    return true;
}

} // namespace

DatasetPair generateDataset(std::size_t n,
                            std::size_t length,
                            std::size_t d,
                            double t,
                            std::size_t num_matches,
                            std::uint64_t seed,
                            std::size_t max_attempts) {
    if (n == 0 || length == 0 || d == 0 || t <= 1.0) throw std::invalid_argument("invalid dataset parameters");
    if (num_matches > n) throw std::invalid_argument("num_matches must be <= n");
    const auto far_threshold = static_cast<std::size_t>(std::ceil(static_cast<double>(d) * t));
    if (far_threshold > length) throw std::invalid_argument("ceil(t*d) exceeds vector length");

    std::mt19937_64 rng(seed);
    DatasetPair dataset;
    dataset.A.reserve(n);
    dataset.B.reserve(n);

    // Build A so any two representatives are at least td apart.
    for (std::size_t i = 0; i < n; ++i) {
        bool placed = false;
        for (std::size_t attempt = 0; attempt < max_attempts && !placed; ++attempt) {
            auto candidate = randomVector(length, rng);
            if (farFromAll(candidate, dataset.A, far_threshold)) {
                dataset.A.push_back(Element{i, std::move(candidate)});
                placed = true;
            }
        }
        if (!placed) throw std::runtime_error("failed to construct separated set A; relax parameters or increase length");
    }

    // Plant num_matches close pairs B[i] ~ A[i].
    for (std::size_t i = 0; i < num_matches; ++i) {
        bool placed = false;
        for (std::size_t attempt = 0; attempt < max_attempts && !placed; ++attempt) {
            auto candidate = mutateClose(dataset.A[i].bits, d, rng);
            if (!farFromAllExcept(candidate, dataset.A, i, far_threshold)) continue;
            if (!farFromAll(candidate, dataset.B, far_threshold)) continue;
            dataset.B.push_back(Element{i, std::move(candidate)});
            dataset.planted_ground_truth.emplace_back(i, i);
            placed = true;
        }
        if (!placed) throw std::runtime_error("failed to plant close B element while preserving gap");
    }

    // Remaining B elements are far from every A and from each other.
    for (std::size_t i = num_matches; i < n; ++i) {
        bool placed = false;
        for (std::size_t attempt = 0; attempt < max_attempts && !placed; ++attempt) {
            auto candidate = randomVector(length, rng);
            if (!farFromAll(candidate, dataset.A, far_threshold)) continue;
            if (!farFromAll(candidate, dataset.B, far_threshold)) continue;
            dataset.B.push_back(Element{i, std::move(candidate)});
            placed = true;
        }
        if (!placed) throw std::runtime_error("failed to construct far B element; relax parameters or increase length");
    }

    return dataset;
}

bool validateGapAssumptions(const DatasetPair& dataset,
                            std::size_t d,
                            double t,
                            bool verbose) {
    const auto far_threshold = static_cast<std::size_t>(std::ceil(static_cast<double>(d) * t));
    bool ok = true;

    auto check_internal = [&](const std::vector<Element>& set, const char* name) {
        for (std::size_t i = 0; i < set.size(); ++i) {
            for (std::size_t j = i + 1; j < set.size(); ++j) {
                const auto dist = hammingDistance(set[i].bits, set[j].bits);
                if (dist < far_threshold) {
                    ok = false;
                    if (verbose) std::cerr << name << " internal gap violation: " << i << "," << j << " dist=" << dist << '\n';
                }
            }
        }
    };
    check_internal(dataset.A, "A");
    check_internal(dataset.B, "B");

    for (const auto& a : dataset.A) {
        for (const auto& b : dataset.B) {
            const auto dist = hammingDistance(a.bits, b.bits);
            if (!(dist <= d || dist >= far_threshold)) {
                ok = false;
                if (verbose) std::cerr << "cross gap violation A[" << a.id << "] B[" << b.id << "] dist=" << dist << '\n';
            }
        }
    }
    return ok;
}

std::vector<PairMatch> bruteForceGroundTruth(const std::vector<Element>& A,
                                             const std::vector<Element>& B,
                                             std::size_t d) {
    std::vector<PairMatch> result;
    for (const auto& a : A) {
        for (const auto& b : B) {
            const auto dist = hammingDistance(a.bits, b.bits);
            if (dist <= d) result.push_back(PairMatch{a.id, b.id, dist, 0});
        }
    }
    return result;
}

} // namespace approx_psi
