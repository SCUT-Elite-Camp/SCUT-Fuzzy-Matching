#include "secure_approx_psi.h"
#include "projection.h"
#include <iostream>
#include <random>
#include <numeric>
#include <algorithm>
#include <thread>
#include <future>
#include <cstring>
#include "coproto/Socket/LocalAsyncSock.h"
#include "cryptoTools/Crypto/PRNG.h"


int main(int argc, char** argv) {
    std::size_t n = 16, ell = 128;
    std::size_t d = 4;
    double t = 4.0, lambda = 10.0;
    std::size_t planted = 4;
    if (argc > 1) n = std::stoull(argv[1]);
    if (argc > 2) ell = std::stoull(argv[2]);
    if (argc > 3) d = std::stoull(argv[3]);
    if (argc > 4) t = std::stod(argv[4]);
    if (argc > 5) lambda = std::stod(argv[5]);
    if (argc > 6) planted = std::stoull(argv[6]);

    Parameters params = makeParameters(n, ell, d, t, lambda);
    std::cerr << "params: n=" << n << " ell=" << ell << " d=" << d
              << " t=" << t << " lambda=" << lambda << " k=" << params.k << std::endl;

    // 数据集
    std::vector<Element> A, B;
    A.reserve(n); B.reserve(n);
    std::mt19937_64 rng_shared(42), rng_alice(111), rng_bob(222);
    for (std::size_t i = 0; i < planted; ++i) {
        BitVector bits(ell);
        for (auto& x : bits) x = rng_shared() & 1;
        A.push_back(Element{i, bits});
        BitVector bits2 = bits;
        std::vector<std::size_t> pos(ell);
        std::iota(pos.begin(), pos.end(), 0);
        std::shuffle(pos.begin(), pos.end(), rng_shared);
        for (std::size_t k = 0; k < d; ++k) bits2[pos[k]] ^= 1;
        B.push_back(Element{i, bits2});
    }
    for (std::size_t i = planted; i < n; ++i) {
        BitVector bits(ell);
        for (auto& x : bits) x = rng_alice() & 1;
        A.push_back(Element{i, bits});
        BitVector bits2(ell);
        for (auto& x : bits2) x = rng_bob() & 1;
        B.push_back(Element{i, bits2});
    }

    // 单进程双线程 + LocalAsyncSocket
    auto sockets = coproto::LocalAsyncSocket::makePair();
    // sockets[0] 给 Bob，sockets[1] 给 Alice

    // 两个 SecureApproxPSI，Alice 侧和 Bob 侧
    // 因为不用 EMP，SecureApproxPSI 不需要 SH2PCSession
    // 用一个新的接口：run(A, sock, isAlice)
    SecureApproxPSI proto(params);

    std::vector<PairMatch> aliceMatches, bobMatches;
    std::thread aliceThread([&]() {
        aliceMatches = proto.run_alice(A, sockets[1]);
    });
    std::thread bobThread([&]() {
        bobMatches = proto.run_bob(B, sockets[0]);
    });
    aliceThread.join();
    bobThread.join();

    std::cout << "Alice matches: " << aliceMatches.size() << "\n";
    std::cout << "Bob matches: " << bobMatches.size() << "\n";
    return 0;
}
