#pragma once
#include "types.h"
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace approx_psi {

struct DatasetPair {
    std::vector<Element> A;
    std::vector<Element> B;
    std::vector<std::pair<std::size_t, std::size_t>> planted_ground_truth;
};

DatasetPair generateDataset(std::size_t n,
                            std::size_t length,
                            std::size_t d,
                            double t,
                            std::size_t num_matches,
                            std::uint64_t seed,
                            std::size_t max_attempts = 200000);

bool validateGapAssumptions(const DatasetPair& dataset,
                            std::size_t d,
                            double t,
                            bool verbose = false);

std::vector<PairMatch> bruteForceGroundTruth(const std::vector<Element>& A,
                                             const std::vector<Element>& B,
                                             std::size_t d);

} // namespace approx_psi
