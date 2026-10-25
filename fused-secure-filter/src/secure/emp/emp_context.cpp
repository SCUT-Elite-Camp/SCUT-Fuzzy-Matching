#include <emp-tool/runtime/io/net_io_channel.h>
#include "secure/emp/emp_context.h"


#ifdef APPROX_PSI_HAS_EMP
#include <emp-sh2pc/emp-sh2pc.h>
#endif


namespace approx_psi::secure {


EMPContext::EMPContext(
    Role role,
    const std::string& address,
    int port
)
:
role_(role),
io_(nullptr),
sess_(nullptr)
{
#ifdef APPROX_PSI_HAS_EMP
    const int party = (role == Role::ALICE) ? emp::ALICE : emp::BOB;

    auto io = (role == Role::ALICE)
        ? emp::NetIO::listen(port)
        : emp::NetIO::connect(address.c_str(), port);

    io_ = io.release();

    sess_ = new emp::SH2PCSession(io_, party);
#else
    (void)address;
    (void)port;
#endif
}


EMPContext::~EMPContext()
{
#ifdef APPROX_PSI_HAS_EMP
    delete sess_;
    sess_ = nullptr;
    delete io_;
    io_ = nullptr;
#endif
}


emp::NetIO* EMPContext::io()
{
    return io_;
}


emp::SH2PCSession* EMPContext::session()
{
    return sess_;
}


Role EMPContext::role() const
{
    return role_;
}


void EMPContext::flush()
{
#ifdef APPROX_PSI_HAS_EMP
    if (io_ != nullptr) {
        io_->flush();
    }
#endif
}


} // namespace approx_psi::secure
