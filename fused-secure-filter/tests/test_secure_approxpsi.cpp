#include "secure/secure_approx_psi.h"
#include "dataset.h"
#include "params.h"

#include <iostream>

using namespace approx_psi;
using namespace approx_psi::secure;

int main()
{
    auto params = makeParameters(16, 128, 4, 5, 5);

    ApproxPSIConfig config;
    config.parameters = params;

    SSPSI psi;
    SSHamming ham;
    SSVMult vmult;

    SecureApproxPSI securePsi(config, psi, ham, vmult);

    auto dataset = generateDataset(16, 128, 4, 5, 4, 1234);

    auto result = securePsi.run(dataset.A, dataset.B);

    std::cout << "Secure ApproxPSI test finished\n";
    std::cout << "Recovered matches: " << result.matches.size() << std::endl;

    return 0;
}
