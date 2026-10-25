#include "secure/ss_psi.h"

#include <random>


#ifdef APPROX_PSI_HAS_VOLEPSI
#include <volePSI/RsCpsi.h>
#endif


namespace approx_psi::secure {


SSPSI::SSPSI()
:
context_(nullptr)
{
}


SSPSI::SSPSI(EMPContext* context)
:
context_(context)
{
}


SSPSI::~SSPSI()
{
}


void SSPSI::setContext(EMPContext* context)
{
    context_ = context;
}


std::vector<PSIPayloadShare>
SSPSI::run(
    const std::vector<std::pair<std::string, std::string>>& sender_input,
    const std::vector<std::pair<std::string, std::string>>& receiver_input
)
{
#ifndef APPROX_PSI_HAS_VOLEPSI
    std::vector<PSIPayloadShare> output;

    for (const auto& a : sender_input) {
        for (const auto& b : receiver_input) {
            if (a.first == b.first) {
                PSIPayloadShare share;
                share.my_id = 0;
                share.peer_id = 0;
                share.local_valid = true;
                share.flag_share = 1;
                share.sender_payload_share = a.second;
                share.receiver_payload_share = b.second;
                output.push_back(std::move(share));
            }
        }
    }

    return output;
#else
    return ss_psi_volepsi_run(sender_input, receiver_input);
#endif
}


std::vector<PSIPayloadShare>
SSPSI::run_as_sender(
    const std::vector<std::pair<std::string, std::string>>& my_input,
    coproto::Socket& psi_sock,
    std::uint64_t seed
)
{
#ifndef APPROX_PSI_HAS_VOLEPSI
    (void)psi_sock;
    (void)seed;
    return run(my_input, {});
#else
    return ss_psi_volepsi_run_sender(my_input, psi_sock, seed);
#endif
}


std::vector<PSIPayloadShare>
SSPSI::run_as_receiver(
    const std::vector<std::pair<std::string, std::string>>& my_input,
    coproto::Socket& psi_sock,
    std::uint64_t seed
)
{
#ifndef APPROX_PSI_HAS_VOLEPSI
    (void)psi_sock;
    (void)seed;
    return run({}, my_input);
#else
    return ss_psi_volepsi_run_receiver(my_input, psi_sock, seed);
#endif
}


} // namespace approx_psi::secure
