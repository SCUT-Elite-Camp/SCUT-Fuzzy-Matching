#include "secure/ss_vmult.h"

#include <cassert>
#include <iostream>

using namespace approx_psi;
using namespace approx_psi::secure;

int main()
{
    BitVector v{ 1,0,1,1, 1,0 };

    auto shares = makeVectorShare(v, 111);

    SharedBit c;
    c.sender_share = 1;
    c.receiver_share = 0;

    SSVMult vm;

    auto result = vm.multiply(
        c,
        shares.first.local,
        shares.second.local
    );

    auto plain = result.reconstruct();

    assert(plain == v);

    std::cout << "SS-VMult test passed\n";
    return 0;
}