#pragma once


#include "secure/common/role.h"

#include <string>


namespace emp {
class NetIO;
class SH2PCSession;
}


namespace approx_psi::secure {


class EMPContext
{
public:
    EMPContext(
        Role role,
        const std::string& address,
        int port
    );

    ~EMPContext();

    emp::NetIO* io();

    emp::SH2PCSession* session();

    Role role() const;

    void flush();

private:
    Role role_;
    emp::NetIO* io_;
    emp::SH2PCSession* sess_;
};


} // namespace approx_psi::secure
