#include "secure/emp/emp_context.h"

#include <iostream>


using namespace approx_psi::secure;


int main()
{
    EMPContext ctx(Role::ALICE, "", 12345);

    std::cout << "Alice EMP context ready\n";

    ctx.flush();

    return 0;
}
