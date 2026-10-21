#pragma once
#include "params.h"
#include "types.h"
#include <cstdint>
#include <vector>

namespace approx_psi {

struct ApproxPSIConfig {
    Parameters parameters;
    std::uint64_t seed{1};
};

class ApproxPSI {
public:
    explicit ApproxPSI(ApproxPSIConfig config);

    ExperimentResult run(const std::vector<Element>& A,
                         const std::vector<Element>& B) const;

private:
    ApproxPSIConfig config_;
};

} // namespace approx_psi
