#include "secure/secure_approx_psi.h"
#include "secure/emp/emp_context.h"

#include <iostream>


using namespace approx_psi::secure;


int main()
{
    EMPContext ctx(Role::ALICE, "", 12345);

    std::cout << "Secure Approx PSI Alice ready\n";

    ctx.flush();

    return 0;
}
