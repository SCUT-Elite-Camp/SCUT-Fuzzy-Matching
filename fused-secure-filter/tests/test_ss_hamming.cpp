#include "secure/ss_hamming.h"

#include <cassert>
#include <iostream>

using namespace approx_psi;
using namespace approx_psi::secure;

int main()
{
    BitVector a{ 1,0,1,1, 0,1,0,0 };
    BitVector b{ 1,1,1,0, 0,0,0,1 };

    auto a_share = makeVectorShare(a, 123);
    auto b_share = makeVectorShare(b, 456);

    SSHamming ham;

    auto result = ham.compare(
        a_share.first.local,
        a_share.second.local,
        b_share.first.local,
        b_share.second.local,
        4
    );

    assert(result.reconstruct());

    std::cout << "SS-Hamming test passed\n";
    return 0;
}
