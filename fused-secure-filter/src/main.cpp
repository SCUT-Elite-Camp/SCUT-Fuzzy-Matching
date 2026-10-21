#include "approx_psi.h"
#include "dataset.h"
#include "params.h"
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>

using namespace approx_psi;

namespace {
struct Cli {
    std::size_t n = 16;
    std::size_t length = 128;
    std::size_t d = 4;
    double t = 4.0;
    double lambda = 10.0;
    std::size_t matches = 4;
    std::size_t trials = 1;
    std::uint64_t seed = 42;
    std::size_t k_override = 0;
};

void printHelp() {
    std::cout <<
        "Merged plain reproduction of Approx-PSI Algorithm 1\n\n"
        "Options:\n"
        "  --n N           set size (default 16)\n"
        "  --ell L         vector length (default 128)\n"
        "  --d D           Hamming threshold (default 4)\n"
        "  --t T           gap (default 4)\n"
        "  --lambda X      statistical security parameter (default 10)\n"
        "  --matches M     planted close pairs (default 4)\n"
        "  --trials R      Monte Carlo trials (default 1)\n"
        "  --seed S        RNG seed (default 42)\n"
        "  --k K           override paper-derived rounds\n";
}

Cli parse(int argc, char** argv) {
    Cli c;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const std::string& name) {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + name);
            return std::string(argv[++i]);
        };
        if (arg == "--help") { printHelp(); std::exit(0); }
        else if (arg == "--n") c.n = std::stoull(need(arg));
        else if (arg == "--ell") c.length = std::stoull(need(arg));
        else if (arg == "--d") c.d = std::stoull(need(arg));
        else if (arg == "--t") c.t = std::stod(need(arg));
        else if (arg == "--lambda") c.lambda = std::stod(need(arg));
        else if (arg == "--matches") c.matches = std::stoull(need(arg));
        else if (arg == "--trials") c.trials = std::stoull(need(arg));
        else if (arg == "--seed") c.seed = std::stoull(need(arg));
        else if (arg == "--k") c.k_override = std::stoull(need(arg));
        else throw std::runtime_error("unknown argument: " + arg);
    }
    return c;
}

std::set<std::pair<std::size_t, std::size_t>> pairSet(const std::vector<PairMatch>& v) {
    std::set<std::pair<std::size_t, std::size_t>> out;
    for (const auto& x : v) out.emplace(x.a_id, x.b_id);
    return out;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto cli = parse(argc, argv);
        if (cli.matches > cli.n) throw std::runtime_error("--matches must be <= n");
        if (cli.trials == 0) throw std::runtime_error("--trials must be positive");

        auto params = makeParameters(cli.n, cli.length, cli.d, cli.t, cli.lambda);
        if (cli.k_override > 0) params.k = cli.k_override;

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "=== Paper parameters ===\n"
                  << "n=" << params.n << ", ell=" << params.length
                  << ", d=" << params.d << ", t=" << params.t
                  << ", lambda=" << params.lambda << '\n'
                  << "q=" << params.q << '\n'
                  << "p=" << params.p << '\n'
                  << "alpha=" << params.alpha << '\n'
                  << "k=" << params.k << '\n';

        std::size_t total_true = 0, total_found_true = 0, total_false_positive = 0;
        std::size_t total_checks = 0, total_intersections = 0;
        double total_ms = 0.0;

        for (std::size_t trial = 0; trial < cli.trials; ++trial) {
            const auto trial_seed = cli.seed + trial * 1000003ULL;
            auto ds = generateDataset(cli.n, cli.length, cli.d, cli.t, cli.matches, trial_seed);
            if (!validateGapAssumptions(ds, cli.d, cli.t, false)) {
                throw std::runtime_error("generated dataset violates paper gap assumptions");
            }

            const auto gt = bruteForceGroundTruth(ds.A, ds.B, cli.d);
            const auto gt_set = pairSet(gt);

            ApproxPSI protocol(ApproxPSIConfig{params, trial_seed + 1});
            const auto begin = std::chrono::steady_clock::now();
            const auto result = protocol.run(ds.A, ds.B);
            const auto end = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(end - begin).count();

            const auto found_set = pairSet(result.matches);
            std::size_t found_true = 0;
            for (const auto& pair : found_set) if (gt_set.count(pair)) ++found_true;
            const auto false_pos = found_set.size() - found_true;

            total_true += gt_set.size();
            total_found_true += found_true;
            total_false_positive += false_pos;
            total_checks += result.stats.hamming_checks;
            total_intersections += result.stats.projection_intersections;
            total_ms += ms;

            if (cli.trials <= 5) {
                std::cout << "\nTrial " << trial + 1
                          << ": true=" << gt_set.size()
                          << ", found_true=" << found_true
                          << ", false_positive=" << false_pos
                          << ", Hamming_checks=" << result.stats.hamming_checks
                          << ", time_ms=" << ms << '\n';
            }
        }

        const double fnr = total_true == 0 ? 0.0 : 1.0 - static_cast<double>(total_found_true) / total_true;
        std::cout << "\n=== Monte Carlo summary ===\n"
                  << "trials=" << cli.trials << '\n'
                  << "total_true_matches=" << total_true << '\n'
                  << "found_true_matches=" << total_found_true << '\n'
                  << "false_negatives=" << (total_true - total_found_true) << '\n'
                  << "FNR=" << fnr << '\n'
                  << "false_positives=" << total_false_positive << '\n'
                  << "avg_projection_intersections=" << static_cast<double>(total_intersections) / cli.trials << '\n'
                  << "avg_Hamming_checks=" << static_cast<double>(total_checks) / cli.trials << '\n'
                  << "bruteforce_checks_per_trial=" << static_cast<double>(cli.n) * cli.n << '\n'
                  << "avg_runtime_ms=" << total_ms / cli.trials << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << '\n';
        return 1;
    }
}
