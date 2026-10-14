#pragma once
#include "types.h"
#include <cmath>
#include <stdexcept>

inline Parameters makeParameters(
    std::size_t n,
    std::size_t length,
    std::size_t d,
    double t,
    double lambda)
{
    const double q = 1.0 / std::pow(static_cast<double>(n) * t, 1.0 / (t - 1.0));
    const double p = 1.0 - std::pow(q, 1.0 / static_cast<double>(d));
    const double alpha = 1.0 - q + static_cast<double>(n) * std::pow(q, t);

    if (!(alpha > 0.0 && alpha < 1.0)) {
        throw std::runtime_error("alpha must be in (0,1)");
    }

    const double numerator = lambda + std::log2(static_cast<double>(n));
    const double denominator = std::log2(1.0 / alpha);
    const double k_raw = std::ceil(numerator / denominator);
    if (!(k_raw > 0.0) || k_raw > 1e9) {
        throw std::runtime_error("k out of range");
    }
    const auto k = static_cast<std::size_t>(k_raw);

    return Parameters{n, length, d, t, lambda, p, q, alpha, k};
}
