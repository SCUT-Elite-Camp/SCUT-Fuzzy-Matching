#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace approx_psi {

using BitVector = std::vector<std::uint8_t>;
using BitIndex = std::vector<std::size_t>;

struct Element {
    std::size_t id{};
    BitVector bits;
};

struct PairMatch {
    std::size_t a_id{};
    std::size_t b_id{};
    std::size_t distance{};
    std::size_t round{};
};

struct RunStats {
    std::size_t rounds{};
    std::size_t total_projection_coordinates{};
    std::size_t projection_intersections{};
    std::size_t hamming_checks{};
    // Per-stage transport counters, measured at this endpoint.
    std::size_t psi_bytes_sent{};
    std::size_t psi_bytes_received{};
    std::size_t emp_bytes_sent{};
    std::size_t emp_bytes_received{};
};

struct ExperimentResult {
    std::vector<PairMatch> matches;
    RunStats stats;
};

} // namespace approx_psi
