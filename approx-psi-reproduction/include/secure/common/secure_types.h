#pragma once


#include "types.h"

#include <string>
#include <vector>


namespace approx_psi::secure {


struct PayloadShare
{
    std::string sender_share;
    std::string receiver_share;
};


struct SecureCommunicationStats
{
    std::size_t bytes_sent{0};
    std::size_t bytes_received{0};
    std::size_t rounds{0};
};


} // namespace approx_psi::secure
