#pragma once


#include "types.h"

#include <string>
#include <vector>


namespace approx_psi {
namespace secure {


struct VolePSIInput
{
    std::string key;
    std::string payload;
};


struct VolePSIShare
{
    std::string sender_share;
    std::string receiver_share;
};


class VolePSIAdapter
{
public:
    VolePSIAdapter();
    ~VolePSIAdapter();

    void initSender(int port);
    void initReceiver(const char* host, int port);

    std::vector<VolePSIShare>
    run(
        const std::vector<VolePSIInput>& sender,
        const std::vector<VolePSIInput>& receiver
    );

    void finalize();

private:
    bool initialized_;
};


}
}
