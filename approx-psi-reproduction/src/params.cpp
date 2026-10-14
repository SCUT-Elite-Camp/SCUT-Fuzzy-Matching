#include "params.h"
#include <cmath>
#include <stdexcept>

namespace approx_psi {

Parameters makeParameters(std::size_t n,
                          std::size_t length,
                          std::size_t d,
                          double t,
                          double lambda) {
    if (n == 0) throw std::invalid_argument("n must be positive");
    if (length == 0) throw std::invalid_argument("length must be positive");
    if (d == 0 || d > length) throw std::invalid_argument("d must be in [1,length]");
    if (t <= 1.0) throw std::invalid_argument("t must be greater than 1");
    if (lambda <= 0.0) throw std::invalid_argument("lambda must be positive");

    // Lemma 6.1 in the paper:
    // q = (1-p)^d = 1 / (nt)^(1/(t-1))
    const double q = 1.0 / std::pow(static_cast<double>(n) * t, 1.0 / (t - 1.0));
    const double p = 1.0 - std::pow(q, 1.0 / static_cast<double>(d));

    // Bad-projection upper bound alpha(n,t) = 1 - q + n q^t.
    const double alpha = 1.0 - q + static_cast<double>(n) * std::pow(q, t);
    if (!(alpha > 0.0 && alpha < 1.0)) {
        throw std::runtime_error("alpha must be in (0,1); check parameters");
    }

    // The paper defines log base 2:
    // k = ceil((lambda + log2 n) / log2(1/alpha)).
    const double numerator = lambda + std::log2(static_cast<double>(n));
    const double denominator = std::log2(1.0 / alpha);
    const auto k = static_cast<std::size_t>(std::ceil(numerator / denominator));

    return Parameters{n, length, d, t, lambda, p, q, alpha, k};
}

} // namespace approx_psi
