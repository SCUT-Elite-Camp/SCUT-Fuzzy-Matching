// Micro-benchmark for the EMP Hamming-predicate circuit variants.
//
// Usage: bench_circuit <party> <port> <mode> [ell] [candidates] [d]
//   party = 1 (ALICE, garbler) or 2 (BOB, evaluator)
//   mode  = u16 | sat | input_only
//
// Prints the AND-gate count and the EMP transport totals, so the per-candidate
// byte cost of each circuit shape can be compared directly.

#include <emp-sh2pc/emp-sh2pc.h>
#include <emp-tool/runtime/io/net_io_channel.h>

#include "secure/hamming_threshold.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace emp;

namespace {

using Ctx = SH2PCSession::ctx_t;
using B = Bit_T<Ctx>;

// Threshold test using `W = ceil(log2(d + 2))` counter bits.
template <std::size_t W>
B sat_leq(const std::vector<B>& diff, std::size_t threshold)
{
    Ctx& ctx = *diff.front().context();
    const std::uint64_t cap = static_cast<std::uint64_t>(threshold) + 1;

    std::vector<B> c;
    c.reserve(W);
    for (std::size_t k = 0; k < W; ++k) c.push_back(B::constant(ctx, false));

    for (const B& bit : diff) {
        // freeze = [c == cap]
        B freeze = B::constant(ctx, true);
        for (std::size_t k = 0; k < W; ++k) {
            const bool want = (cap >> k) & 1u;
            freeze = freeze & (want ? c[k] : !c[k]);
        }
        const B inc = bit & !freeze;
        B carry = inc;
        for (std::size_t k = 0; k < W; ++k) {
            const B sum = c[k] ^ carry;
            carry = c[k] & carry;
            c[k] = sum;
        }
        (void)ctx;
    }

    B at_cap = B::constant(ctx, true);
    for (std::size_t k = 0; k < W; ++k) {
        const bool want = (cap >> k) & 1u;
        at_cap = at_cap & (want ? c[k] : !c[k]);
    }
    return !at_cap;
}

// Current production circuit: 16-bit ripple counter fed by a 16-bit mux.
B u16_leq(const std::vector<B>& a, const std::vector<B>& b, std::size_t threshold)
{
    using U16 = UInt_T<Ctx, 16>;
    Ctx& ctx = *a.front().context();
    U16 sum = U16::constant(ctx, 0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        const B diff = a[i] ^ b[i];
        sum = sum + U16::constant(ctx, 0).select(diff, U16::constant(ctx, 1));
    }
    return sum <= U16::constant(ctx, static_cast<std::uint64_t>(threshold));
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cerr << "usage: bench_circuit <party> <port> "
                     "<u16|sat|tree|wallace|generic_sat|input_only> "
                     "[ell] [candidates] [d] [check]\n";
        return 1;
    }
    const int party = std::atoi(argv[1]);
    const int port = std::atoi(argv[2]);
    const std::string mode = argv[3];
    const std::size_t ell = (argc > 4) ? std::stoull(argv[4]) : 128;
    const std::size_t m = (argc > 5) ? std::stoull(argv[5]) : 1;
    const std::size_t d = (argc > 6) ? std::stoull(argv[6]) : 4;
    const bool check = (argc > 7) && std::string(argv[7]) == "check";

    auto io = (party == ALICE) ? NetIO::listen(port)
                               : NetIO::connect("127.0.0.1", port);
    SH2PCSession sess(io.get(), party);
    const bool alice = (party == ALICE);

    const auto start = std::chrono::steady_clock::now();
    const std::uint64_t gates0 = sess.num_and();

    std::vector<B> keep_wires;
    keep_wires.reserve(m);

    std::vector<std::uint8_t> av(ell), bv(ell);
    std::vector<bool> expected(m, false);
    std::size_t mismatches = 0;

    for (std::size_t c = 0; c < m; ++c) {
        B flag_a = sess.input<B>(ALICE, alice);
        B flag_b = sess.input<B>(BOB, false);
        B valid_a = sess.input<B>(ALICE, alice);
        B valid_b = sess.input<B>(BOB, !alice);

        if (check) {
            std::mt19937_64 rng(0x9e3779b97f4a7c15ull * (c + 1));
            const std::size_t ones = c % (d + 5); // exercise 0..d+4 mismatches
            for (std::size_t i = 0; i < ell; ++i) {
                av[i] = static_cast<std::uint8_t>(rng() & 1);
                bv[i] = av[i];
            }
            for (std::size_t k = 0; k < ones && ell > 0; ++k) {
                bv[(rng() % ell)] ^= 1;
            }
        } else {
            for (std::size_t i = 0; i < ell; ++i) {
                av[i] = static_cast<std::uint8_t>((i + c) & 1);
                bv[i] = static_cast<std::uint8_t>((i * 3 + c) & 1);
            }
        }

        std::vector<B> a(ell);
        std::vector<B> b(ell);
        for (std::size_t i = 0; i < ell; ++i) {
            a[i] = sess.input<B>(ALICE, alice && (av[i] != 0));
            b[i] = sess.input<B>(BOB, !alice && (bv[i] != 0));
        }

        B ok = B::constant(sess.ctx(), false);
        if (mode == "u16") {
            ok = u16_leq(a, b, d);
        } else if (mode == "sat") {
            std::vector<B> diff(ell);
            for (std::size_t i = 0; i < ell; ++i) diff[i] = a[i] ^ b[i];
            ok = sat_leq<3>(diff, d);
        } else if (mode == "wallace") {
            std::vector<B> diff(ell);
            for (std::size_t i = 0; i < ell; ++i) diff[i] = a[i] ^ b[i];
            ok = approx_psi::secure::hamming_leq_wallace<Ctx>(diff, d);
        } else if (mode == "tree") {
            std::vector<B> diff(ell);
            for (std::size_t i = 0; i < ell; ++i) diff[i] = a[i] ^ b[i];
            ok = approx_psi::secure::hamming_leq_tree<Ctx>(diff, d);
        } else if (mode == "generic_sat") {
            std::vector<B> diff(ell);
            for (std::size_t i = 0; i < ell; ++i) diff[i] = a[i] ^ b[i];
            ok = approx_psi::secure::hamming_leq_saturating<Ctx>(diff, d);
        } else if (mode == "input_only") {
            ok = a[0] & b[0];
        } else {
            std::cerr << "unknown mode\n";
            return 1;
        }
        keep_wires.push_back((flag_a ^ flag_b) & valid_a & valid_b & ok);
        if (check) {
            std::size_t dist = 0;
            for (std::size_t i = 0; i < ell; ++i) dist += (av[i] ^ bv[i]) != 0;
            expected[c] = (dist <= d);
        }
    }

    std::vector<bool> out(m);
    for (std::size_t i = 0; i < m; ++i) {
        auto ra = sess.reveal(keep_wires[i], ALICE);
        auto rb = sess.reveal(keep_wires[i], BOB);
        out[i] = alice ? (ra.has_value() && *ra) : (rb.has_value() && *rb);
        if (check && out[i] != expected[i]) ++mismatches;
    }
    sess.finalize();
    io->flush();

    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start).count();
    const std::uint64_t gates = sess.num_and() - gates0;

    std::cerr << "mode=" << mode << " check=" << check
              << " mismatches=" << mismatches
              << " ell=" << ell << " m=" << m
              << " and_gates_per_candidate=" << (gates / m)
              << " and_gates_total=" << gates
              << " wall_ms=" << ms << "\n";
    std::cerr << io->get_statistics_string();
    (void)out;
    return 0;
}
