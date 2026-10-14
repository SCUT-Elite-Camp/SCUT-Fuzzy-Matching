#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>

using BitVector = std::vector<uint8_t>;

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

struct Parameters {
    std::size_t n{};
    std::size_t length{};
    std::size_t d{};
    double t{};
    double lambda{};
    double p{};
    double q{};
    double alpha{};
    std::size_t k{};
};
