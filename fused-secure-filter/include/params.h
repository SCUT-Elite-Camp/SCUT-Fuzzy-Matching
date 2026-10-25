#pragma once
#include <cstddef>

namespace approx_psi {

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

Parameters makeParameters(std::size_t n,
                          std::size_t length,
                          std::size_t d,
                          double t,
                          double lambda);

} // namespace approx_psi
