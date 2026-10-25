#include <emp-sh2pc/emp-sh2pc.h>

#include <iostream>
#include <string>

using namespace emp;

int main(int argc, char** argv)
{
    int party = std::stoi(argv[1]);
    int port = std::stoi(argv[2]);

    auto io = (party == ALICE)
        ? NetIO::listen(port)
        : NetIO::connect("127.0.0.1", port);

    SH2PCSession sess(io.get(), party);

    using Ctx = SH2PCSession::ctx_t;
    using B = Bit_T<Ctx>;

    // One private bit from each party; cleartext is 1 for both.
    B a = sess.input<B>(ALICE, 1);
    B b = sess.input<B>(BOB, 1);

    // AND gate output; cleartext is 1.
    B v = a & b;
    auto out = sess.reveal(v, XOR);
    std::cerr << "[party " << party << "] AND out="
              << (out.has_value() ? (int)out.value() : -1) << std::endl;

    // XOR gate output; cleartext is 0.
    B w = a ^ b;
    auto out2 = sess.reveal(w, XOR);
    std::cerr << "[party " << party << "] XOR out="
              << (out2.has_value() ? (int)out2.value() : -1) << std::endl;

    sess.finalize();
    return 0;
}
