#include "approx_psi.h"
#include "dataset.h"
#include "hamming.h"
#include "params.h"
#include "projection.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

using namespace approx_psi;

int main() {
    {
        BitVector a{0,1,0,1,1}, b{0,0,0,1,0};
        assert(hammingDistance(a,b) == 2);
        assert(withinHammingDistance(a,b,2));
        assert(!withinHammingDistance(a,b,1));
    }
    {
        BitVector value{1,0,1,1,0};
        BitIndex index{0,2,4};
        const auto projected = project(value,index);
        assert((projected == BitVector{1,1,0}));
    }
    {
        // Paper Table 3 sanity check.
        auto p256 = makeParameters(256,128,4,8.0,40.0);
        auto p1024 = makeParameters(1024,128,4,10.0,40.0);
        auto p4096 = makeParameters(4096,128,4,12.0,40.0);
        assert(p256.k == 96);
        assert(p1024.k == 89);
        assert(p4096.k == 86);
    }
    {
        const auto ds = generateDataset(16,128,4,4.0,4,12345);
        assert(ds.A.size() == 16 && ds.B.size() == 16);
        assert(ds.planted_ground_truth.size() == 4);
        assert(validateGapAssumptions(ds,4,4.0,false));
        const auto gt = bruteForceGroundTruth(ds.A,ds.B,4);
        assert(gt.size() == 4);

        auto params = makeParameters(16,128,4,4.0,10.0);
        ApproxPSI protocol(ApproxPSIConfig{params,999});
        const auto result = protocol.run(ds.A,ds.B);
        assert(result.matches.size() == 4);
        assert(result.stats.hamming_checks <= params.k * params.n);
    }
    {
        // Verify random representative semantics: only one representative per identical key.
        std::vector<Element> s{{0,{1,0,1}}, {1,{1,0,1}}, {2,{0,0,0}}};
        BitIndex I{0,1,2};
        std::mt19937_64 rng(7);
        const auto reps = projectSetWithRandomRepresentative(s,I,rng);
        assert(reps.size() == 2);
    }
    std::cout << "All tests passed.\n";
    return 0;
}
