#include "secure/common/share.h"

#include <cassert>
#include <iostream>
#include <random>

using namespace approx_psi;
using namespace approx_psi::secure;

int main()
{
    std::mt19937_64 rng(12345);

    for (int i = 0; i < 1000; i++) {
        std::uint8_t value = rng() % 2;

        auto shares = makeVectorShare(BitVector{ value }, rng());

        auto reconstructed = reconstructXOR(
            shares.first.local,
            shares.second.local
        );

        assert(reconstructed[0] == value);
    }

    for (int i = 0; i < 100; i++) {
        BitVector input(128, 0);
        for (auto& x : input) x = rng() % 2;

        auto shares = makeVectorShare(input, rng());

        auto output = reconstructXOR(
            shares.first.local,
            shares.second.local
        );

        assert(output == input);
    }

    std::cout << "Secret sharing test passed\n";
    return 0;
}