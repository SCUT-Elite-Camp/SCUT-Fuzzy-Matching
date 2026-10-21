#include "secure/emp/emp_context.h"

#include <iostream>


using namespace approx_psi::secure;


int main()
{
    EMPContext ctx(Role::BOB, "127.0.0.1", 12345);

    std::cout << "Bob EMP context ready\n";

    ctx.flush();

    return 0;
}
