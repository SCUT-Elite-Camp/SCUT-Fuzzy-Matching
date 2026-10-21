#include <emp-sh2pc/emp-sh2pc.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace emp;

namespace {

bool valid_bit_string(const std::string& s)
{
    if (s.empty()) return false;
    for (char c : s) {
        if (c != '0' && c != '1') return false;
    }
    return true;
}

void usage(const char* prog)
{
    std::cerr
        << "Usage:\n"
        << "  Alice: " << prog << " 1 <port> <alice_bits> <threshold>\n"
        << "  Bob:   " << prog << " 2 <port> <bob_bits>   <threshold> [alice_ip]\n\n"
        << "Example:\n"
        << "  " << prog << " 1 12345 101101 1\n"
        << "  " << prog << " 2 12345 101001 1 127.0.0.1\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 5 || argc > 6) {
        usage(argv[0]);
        return 1;
    }

    const int party = std::stoi(argv[1]);
    const int port = std::stoi(argv[2]);
    const std::string local_bits = argv[3];
    const std::uint32_t threshold = static_cast<std::uint32_t>(std::stoul(argv[4]));

    if (party != ALICE && party != BOB) {
        throw std::invalid_argument("party must be 1 (ALICE) or 2 (BOB)");
    }
    if (!valid_bit_string(local_bits)) {
        throw std::invalid_argument("bit string must contain only 0 and 1");
    }
    if (threshold > local_bits.size()) {
        throw std::invalid_argument("threshold cannot exceed vector length");
    }

    const char* alice_ip = "127.0.0.1";
    if (party == BOB && argc == 6) {
        alice_ip = argv[5];
    }

    // Alice listens; Bob connects. Vector length and threshold are public protocol
    // parameters, so checking them outside the garbled circuit does not leak inputs.
    auto io = (party == ALICE) ? NetIO::listen(port) : NetIO::connect(alice_ip, port);

    std::uint64_t peer_len = 0;
    std::uint32_t peer_threshold = 0;
    if (party == ALICE) {
        const std::uint64_t len = local_bits.size();
        io->send_data(&len, sizeof(len));
        io->send_data(&threshold, sizeof(threshold));
        io->flush();
    } else {
        io->recv_data(&peer_len, sizeof(peer_len));
        io->recv_data(&peer_threshold, sizeof(peer_threshold));
        if (peer_len != local_bits.size()) {
            std::cerr << "ERROR: Alice/Bob vector lengths differ: Alice="
                      << peer_len << ", Bob=" << local_bits.size() << "\n";
            return 2;
        }
        if (peer_threshold != threshold) {
            std::cerr << "ERROR: Alice/Bob thresholds differ: Alice="
                      << peer_threshold << ", Bob=" << threshold << "\n";
            return 2;
        }
    }

    const auto begin = std::chrono::steady_clock::now();

    SH2PCSession sess(io.get(), party);
    using Ctx = SH2PCSession::ctx_t;
    using B = Bit_T<Ctx>;
    using U32 = UInt_T<Ctx, 32>;

    // Each process supplies only its own private vector. For an input owned by the
    // other party, the local clear value is a dummy; EMP ignores it for that party.
    std::vector<B> diff;
    diff.reserve(local_bits.size());

    for (std::size_t i = 0; i < local_bits.size(); ++i) {
        const bool local_value = (local_bits[i] == '1');

        B a_bit = sess.input<B>(ALICE, (party == ALICE) ? local_value : false);
        B b_bit = sess.input<B>(BOB, (party == BOB) ? local_value : false);

        diff.push_back(a_bit ^ b_bit);
    }

    // Secure Hamming weight. The count itself is never revealed.
    U32 distance = U32::constant(sess.ctx(), 0);
    for (const B& d : diff) {
        auto zero = U32::constant(sess.ctx(), 0);
        auto one = U32::constant(sess.ctx(), 1);
        distance = distance + zero.select(d, one);
    }

    U32 limit = U32::constant(sess.ctx(), threshold);
    B is_close = distance <= limit;

    // Produce XOR shares of [H(a,b) <= threshold] without opening the bit:
    //   Alice share = r
    //   Bob share   = result XOR r
    // where r is fresh random data known only to Alice.
    bool alice_random_share = false;
    if (party == ALICE) {
        std::random_device rd;
        alice_random_share = static_cast<bool>(rd() & 1U);
    }

    B r = B::constant(sess.ctx(), alice_random_share);
    B bob_share_wire = is_close ^ r;

    // Both parties must execute the reveal protocol, but only Bob is the recipient.
    auto out = sess.reveal(bob_share_wire, BOB);
    const bool bob_share = (party == BOB) ? *out : false;

    sess.finalize();

    io->flush();
    const auto end = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(end - begin).count();

    if (party == ALICE) {
        std::cout << "party=ALICE\n";
        std::cout << "vector_length=" << local_bits.size() << "\n";
        std::cout << "threshold=" << threshold << "\n";
        std::cout << "local_output_share=" << static_cast<int>(alice_random_share) << "\n";
        std::cout << "protocol_ms=" << ms << "\n";
        std::cout << "NOTE: Alice does NOT learn the comparison result.\n";
    } else {
        std::cout << "party=BOB\n";
        std::cout << "vector_length=" << local_bits.size() << "\n";
        std::cout << "threshold=" << threshold << "\n";
        std::cout << "local_output_share=" << static_cast<int>(bob_share) << "\n";
        std::cout << "protocol_ms=" << ms << "\n";
        std::cout << "NOTE: Bob does NOT learn the comparison result.\n";
    }

    return 0;
}
