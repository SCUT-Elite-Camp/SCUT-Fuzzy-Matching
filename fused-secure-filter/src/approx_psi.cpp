#include "approx_psi.h"
#include "hamming.h"
#include "projection.h"
#include <algorithm>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace approx_psi {

ApproxPSI::ApproxPSI(ApproxPSIConfig config) : config_(std::move(config)) {
    if (config_.parameters.n == 0 || config_.parameters.length == 0) {
        throw std::invalid_argument("invalid ApproxPSI parameters");
    }
}

ExperimentResult ApproxPSI::run(const std::vector<Element>& A,
                                const std::vector<Element>& B) const {
    if (A.size() != config_.parameters.n || B.size() != config_.parameters.n) {
        throw std::invalid_argument("input set size does not match configured n");
    }
    for (const auto& x : A) if (x.bits.size() != config_.parameters.length) throw std::invalid_argument("invalid A vector length");
    for (const auto& x : B) if (x.bits.size() != config_.parameters.length) throw std::invalid_argument("invalid B vector length");

    std::mt19937_64 rng(config_.seed);
    ExperimentResult result;
    result.stats.rounds = config_.parameters.k;

    std::unordered_map<std::uint64_t, PairMatch> unique_matches;

    for (std::size_t round = 0; round < config_.parameters.k; ++round) {
        const auto I = sampleProjection(config_.parameters.length, config_.parameters.p, rng);
        result.stats.total_projection_coordinates += I.size();

        // Algorithm 1 line 4: each side randomly keeps one representative per projection value.
        const auto Aproj = projectSetWithRandomRepresentative(A, I, rng);
        const auto Bproj = projectSetWithRandomRepresentative(B, I, rng);

        // Plain substitute for F_ssPSI: exact equality of projection keys.
        const auto* smaller = &Aproj;
        const auto* larger = &Bproj;
        bool swapped = false;
        if (Aproj.size() > Bproj.size()) {
            smaller = &Bproj;
            larger = &Aproj;
            swapped = true;
        }

        for (const auto& [key, left_rep] : *smaller) {
            const auto it = larger->find(key);
            if (it == larger->end()) continue;
            ++result.stats.projection_intersections;
            ++result.stats.hamming_checks;

            const auto& a = swapped ? it->second.original : left_rep.original;
            const auto& b = swapped ? left_rep.original : it->second.original;
            const auto dist = hammingDistance(a.bits, b.bits);

            // Plain substitute for F_ssHamCom + F_ssVMult.
            if (dist > config_.parameters.d) continue;

            const std::uint64_t pair_key =
                (static_cast<std::uint64_t>(a.id) << 32U) |
                static_cast<std::uint64_t>(b.id & 0xffffffffULL);
            unique_matches.try_emplace(pair_key, PairMatch{a.id, b.id, dist, round});
        }
    }

    result.matches.reserve(unique_matches.size());
    for (const auto& [_, match] : unique_matches) result.matches.push_back(match);
    std::sort(result.matches.begin(), result.matches.end(), [](const PairMatch& x, const PairMatch& y) {
        return x.a_id < y.a_id || (x.a_id == y.a_id && x.b_id < y.b_id);
    });
    return result;
}

} // namespace approx_psi
