#include "secure/volepsi_adapter.h"


namespace approx_psi {
namespace secure {


VolePSIAdapter::VolePSIAdapter()
:
initialized_(false)
{
}


VolePSIAdapter::~VolePSIAdapter()
{
}


void VolePSIAdapter::initSender(int port)
{
    (void)port;
    initialized_ = true;
}


void VolePSIAdapter::initReceiver(const char* host, int port)
{
    (void)host;
    (void)port;
    initialized_ = true;
}


std::vector<VolePSIShare>
VolePSIAdapter::run(
    const std::vector<VolePSIInput>& sender,
    const std::vector<VolePSIInput>& receiver
)
{
    (void)sender;
    (void)receiver;

    return {};
}


void VolePSIAdapter::finalize()
{
    initialized_ = false;
}


}
}
