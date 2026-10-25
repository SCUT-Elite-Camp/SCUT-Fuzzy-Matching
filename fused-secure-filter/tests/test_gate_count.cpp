#include <emp-sh2pc/emp-sh2pc.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace emp;

int main(int argc, char** argv)
{
    int party = std::stoi(argv[1]);
    int port = std::stoi(argv[2]);
    int local_bit = std::stoi(argv[3]);

    auto io = (party == ALICE)
        ? NetIO::listen(port)
        : NetIO::connect("127.0.0.1", port);

    SH2PCSession sess(io.get(), party);

    using Ctx = SH2PCSession::ctx_t;
    using B = Bit_T<Ctx>;
    using U32 = UInt_T<Ctx, 32>;

    std::uint64_t start = sess.num_and();

    // Mirror the circuit that secureVerify builds.
    std::vector<B> diff;
    for (int i = 0; i < 128; i++) {
        B a = sess.input<B>(
            ALICE, party == ALICE ? (local_bit & (i & 1)) : false);
        B b = sess.input<B>(
            BOB, party == BOB ? (local_bit & (i & 1)) : false);
        diff.push_back(a ^ b);
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
    std::cerr << "[party " << party << "] num_and=" << used << std::endl;

    sess.finalize();
    return 0;
}
