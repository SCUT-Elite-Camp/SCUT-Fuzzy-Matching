#include "secure/ss_psi.h"

#include <cassert>
#include <iostream>


using namespace approx_psi::secure;


int main()
{
    std::vector<std::pair<std::string, std::string>> A = {
        { "1010", "payload_A" },
        { "1111", "payload_B" }
    };

    std::vector<std::pair<std::string, std::string>> B = {
        { "1010", "payload_C" }
    };

    SSPSI psi;

    auto result = psi.run(A, B);

#ifndef APPROX_PSI_HAS_VOLEPSI
    assert(result.size() == 1);
#endif

    std::cout << "SS-PSI test passed\n";
    return 0;
}
