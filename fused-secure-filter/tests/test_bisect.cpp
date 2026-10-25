// Bisection probe: does a preceding RsCpsi round disturb the EMP session?
//
// Each process brings up both channels that approx_psi_full uses:
//   - psi_port: a coproto socket carrying one RsCpsi join
//   - emp_port: an EMP SH2PCSession
// then builds the same 128-bit compare+mask circuit as secureVerify and reports
// the AND-gate count. If both parties agree and both exit, RsCpsi and the EMP
// session do not interfere; if one blocks, they do.

#include <emp-sh2pc/emp-sh2pc.h>

#include "coproto/Socket/AsioSocket.h"
#include "volePSI/RsCpsi.h"
#include "cryptoTools/Common/BitVector.h"
#include "cryptoTools/Crypto/PRNG.h"
#include "cryptoTools/Crypto/RandomOracle.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace emp;

namespace {

constexpr std::size_t kPayloadBytes = sizeof(oc::block);

oc::block keyToBlock(const std::string& key)
{
    oc::RandomOracle ro(sizeof(oc::block));
    ro.Update(key.data(), key.size());
    oc::block b;
    ro.Final(b);
    return b;
}

} // namespace

int main(int argc, char** argv)
{
    int party = std::stoi(argv[1]);
    int emp_port = std::stoi(argv[2]);
    int psi_port = std::stoi(argv[3]);

    const bool am_alice = (party == ALICE);

    const std::string addr = "localhost:" + std::to_string(psi_port);
    auto psi_sock = coproto::asioConnect(addr, /*server=*/am_alice);

    using Ctx = SH2PCSession::ctx_t;
    using B = Bit_T<Ctx>;
    using U32 = UInt_T<Ctx, 32>;

    // Round 0's PSI runs BEFORE the EMP session exists (the previously-passing
    // ordering); the session is then built once and reused for both rounds.
    auto run_psi = [&](int round) {
        std::vector<oc::block> keys;
        for (int i = 0; i < 8; i++) {
            keys.push_back(keyToBlock("key" + std::to_string(i)));
        }

        if (am_alice) {
            volePSI::RsCpsiSender sender;
            volePSI::RsCpsiSender::Sharing sShare;
            sender.init(keys.size(), keys.size(), kPayloadBytes, 40,
                        oc::sysRandomSeed(), 1,
                        volePSI::ValueShareType::Xor, 4);

            oc::Matrix<oc::u8> values(keys.size(), kPayloadBytes);
            auto p = sender.send(keys, values, sShare, psi_sock);
            coproto::sync_wait(p);
            std::cerr << "[party " << party << "] round " << round
                      << " RsCpsi sender done" << std::endl;
        } else {
            volePSI::RsCpsiReceiver recver;
            volePSI::RsCpsiReceiver::Sharing rShare;
            recver.init(keys.size(), keys.size(), kPayloadBytes, 40,
                        oc::sysRandomSeed(), 1,
                        volePSI::ValueShareType::Xor, 4);
            auto p = recver.receive(keys, rShare, psi_sock);
            coproto::sync_wait(p);
            std::cerr << "[party " << party << "] round " << round
                      << " RsCpsi receiver done" << std::endl;
        }
    };

    run_psi(0);

    auto io = am_alice
        ? NetIO::listen(emp_port)
        : NetIO::connect("127.0.0.1", emp_port);

    SH2PCSession sess(io.get(), party);

    std::cerr << "[party " << party << "] session up" << std::endl;

    for (int round = 0; round < 2; round++) {
        if (round > 0) {
            run_psi(round);
        }

        // ---- Circuit for this round, on the SAME EMP session ----
        std::uint64_t start = sess.num_and();

        std::vector<B> diff;
        for (int i = 0; i < 128; i++) {
            B a_s = sess.input<B>(ALICE, am_alice ? ((i & 1) != 0) : false);
            B a_r = sess.input<B>(BOB, am_alice ? false : ((i & 1) != 0));
            B b_s = sess.input<B>(ALICE, am_alice ? ((i & 2) != 0) : false);
            B b_r = sess.input<B>(BOB, am_alice ? false : ((i & 2) != 0));
            diff.push_back((a_s ^ a_r) ^ (b_s ^ b_r));
        }

        U32 sum = U32::constant(sess.ctx(), 0);
        for (int i = 0; i < 128; i++) {
            U32 zero = U32::constant(sess.ctx(), 0);
            U32 one = U32::constant(sess.ctx(), 1);
            sum = sum + zero.select(diff[i], one);
        }

        U32 limit = U32::constant(sess.ctx(), 4);
        B c = sum <= limit;

        std::vector<B> masked;
        for (int i = 0; i < 128; i++) {
            masked.push_back(c & diff[i]);
        }

        std::uint64_t used = sess.num_and() - start;

        auto r = sess.reveal(masked[0], XOR);
        int revealed = (r.has_value() && *r) ? 1 : 0;

        std::cerr << "[party " << party << "] round " << round
                  << " num_and=" << used << " reveal0=" << revealed << std::endl;

        // ---- A psi_sock exchange, as run_round does after the circuit ----
        std::uint8_t mine = static_cast<std::uint8_t>(revealed);
        std::uint8_t peer = 0;

        if (am_alice) {
            auto s = psi_sock.send(coproto::copy(mine));
            coproto::sync_wait(s);
            auto rr = psi_sock.recv(peer);
            coproto::sync_wait(rr);
        } else {
            auto rr = psi_sock.recv(peer);
            coproto::sync_wait(rr);
            auto s = psi_sock.send(coproto::copy(mine));
            coproto::sync_wait(s);
        }

        std::cerr << "[party " << party << "] round " << round
                  << " exchange done, peer=" << (int)peer << std::endl;
    }

    sess.finalize();
    return 0;
}
