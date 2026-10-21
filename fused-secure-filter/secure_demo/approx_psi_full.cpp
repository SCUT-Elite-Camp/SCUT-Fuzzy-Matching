#include <coproto/Socket/AsioSocket.h>
#include <emp-tool/runtime/io/net_io_channel.h>
#include "secure/secure_approx_psi.h"
#include "secure/ss_psi.h"
#include "secure/ss_hamming.h"
#include "secure/ss_vmult.h"
#include "secure/emp/emp_context.h"

#include "params.h"
#include "projection.h"
#include "types.h"


#ifdef APPROX_PSI_HAS_VOLEPSI
#endif

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace approx_psi;
using namespace approx_psi::secure;

namespace {

constexpr std::size_t kPlantedMatches = 4;

// First kPlantedMatches pairs are genuine matches (Bob's copy of Alice's vector
// with d bits flipped); the rest are independent random vectors.
void buildDatasets(
    std::size_t n,
    std::size_t ell,
    std::size_t d,
    std::vector<Element>& A,
    std::vector<Element>& B
)
{
    A.clear();
    B.clear();
    A.reserve(n);
    B.reserve(n);

    std::mt19937_64 rng_shared(42);
    std::mt19937_64 rng_alice(111);
    std::mt19937_64 rng_bob(222);

    const std::size_t planted = std::min(kPlantedMatches, n);

    for (std::size_t i = 0; i < planted; i++) {
        BitVector bits(ell);
        for (auto& x : bits) x = static_cast<std::uint8_t>(rng_shared() & 1);

        A.push_back(Element{ i, bits });

        BitVector bits2 = bits;
        std::vector<std::size_t> positions(ell);
        std::iota(positions.begin(), positions.end(), 0);
        std::shuffle(positions.begin(), positions.end(), rng_shared);

        for (std::size_t k = 0; k < d && k < ell; k++) {
            bits2[positions[k]] ^= 1;
        }

        B.push_back(Element{ i, bits2 });
    }

    for (std::size_t i = planted; i < n; i++) {
        BitVector bits(ell);
        for (auto& x : bits) x = static_cast<std::uint8_t>(rng_alice() & 1);
        A.push_back(Element{ i, bits });

        BitVector bits2(ell);
        for (auto& x : bits2) x = static_cast<std::uint8_t>(rng_bob() & 1);
        B.push_back(Element{ i, bits2 });
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cerr
            << "Usage:\n"
            << "  " << argv[0]
            << " <party> <emp_port> <psi_port> [alice_ip] [n] [ell] [d] [t] [lambda]\n"
            << "    party = 1 (ALICE) or 2 (BOB)\n";
        return 1;
    }

    const int party = std::stoi(argv[1]);
    const int emp_port = std::stoi(argv[2]);
    const int psi_port = std::stoi(argv[3]);

    std::string alice_ip = "127.0.0.1";
    if (argc > 4) alice_ip = argv[4];

    std::size_t n = 16;
    std::size_t ell = 128;
    std::size_t d = 4;
    double t = 4.0;
    double lambda = 10.0;

    if (argc > 5) n = static_cast<std::size_t>(std::stoull(argv[5]));
    if (argc > 6) ell = static_cast<std::size_t>(std::stoull(argv[6]));
    if (argc > 7) d = static_cast<std::size_t>(std::stoull(argv[7]));
    if (argc > 8) t = std::stod(argv[8]);
    if (argc > 9) lambda = std::stod(argv[9]);

    if (party != 1 && party != 2) {
        std::cerr << "ERROR: party must be 1 (ALICE) or 2 (BOB)\n";
        return 1;
    }

    const Role role = (party == 1) ? Role::ALICE : Role::BOB;
    const char* party_name = (party == 1) ? "ALICE" : "BOB";

    Parameters params = makeParameters(n, ell, d, t, lambda);

    std::vector<Element> A;
    std::vector<Element> B;
    buildDatasets(n, ell, d, A, B);

    ApproxPSIConfig config;
    config.parameters = params;
    config.seed = 1;

    EMPContext ctx(role, alice_ip, emp_port);

#ifdef APPROX_PSI_HAS_VOLEPSI
    const std::string psi_addr = "localhost:" + std::to_string(psi_port);
    auto psi_sock = coproto::asioConnect(psi_addr, /*server=*/party == 1);

    SSPSI ssPsi(&ctx);
    SSHamming ssHam(&ctx);
    SSVMult ssVMult(&ctx);

    SecureApproxPSI proto(config, ssPsi, ssHam, ssVMult);

    std::cerr << "[" << (party == 1 ? "alice" : "bob")
              << "-main] entering " << (party == 1 ? "run_alice" : "run_bob")
              << std::endl;

    ExperimentResult result;
    try {
        result = (party == 1)
            ? proto.run_alice(A, psi_sock)
            : proto.run_bob(B, psi_sock);
    } catch (const std::exception& e) {
        std::cerr << "[" << (party == 1 ? "alice" : "bob")
                  << "-main] EXCEPTION: " << e.what() << std::endl;
        std::cerr << "[" << (party == 1 ? "alice" : "bob")
                  << "-main] returning 1" << std::endl;
        return 1;
    }

    std::cerr << "[" << (party == 1 ? "alice" : "bob")
              << "-main] run returned" << std::endl;

    std::cout << "party=" << party_name << "\n";
    std::cout << "matched_local_elements=" << result.matches.size() << "\n";

    for (const auto& match : result.matches) {
        const std::size_t local_id = (party == 1) ? match.a_id : match.b_id;
        std::cout << "local_id=" << local_id
                  << " first_match_round=" << match.round << "\n";
    }

    std::cout << "stats: rounds=" << result.stats.rounds
              << " projection_coordinates=" << result.stats.total_projection_coordinates
              << " secure_bins_processed=" << result.stats.hamming_checks << "\n";
    std::cout << "stage_bytes:"
              << " psi_sent=" << result.stats.psi_bytes_sent
              << " psi_recv=" << result.stats.psi_bytes_received
              << " emp_sent=" << result.stats.emp_bytes_sent
              << " emp_recv=" << result.stats.emp_bytes_received << "\n";
#else
    (void)psi_port;
    std::cout << "party=" << party_name << "\n";
    std::cout << "ERROR: built without APPROX_PSI_HAS_VOLEPSI\n";
#endif

    // Final flush before printing statistics.  Redundant with per-round
    // flush, but keeps the last EMP exchange clean before NetIO teardown.
    ctx.flush();

    if (ctx.io() != nullptr) {
        std::cout << ctx.io()->get_statistics_string();
    }

    return 0;
}
