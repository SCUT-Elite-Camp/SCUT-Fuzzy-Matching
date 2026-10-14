#pragma once


#include "types.h"
#include "secure/common/share.h"
#include "secure/emp/emp_context.h"

#include <cstddef>
#include <cstdint>


namespace approx_psi::secure {


struct SharedBit
{
    std::uint8_t sender_share{0};
    std::uint8_t receiver_share{0};

    bool reconstruct() const
    {
        return sender_share ^ receiver_share;
    }
};


class SSHamming
{
public:
    SSHamming();
    explicit SSHamming(EMPContext* context);
    ~SSHamming();

    void setContext(EMPContext* context);

    // The session this instance is bound to, so a caller that already holds the
    // sub-protocol can reach the circuit context without a second pointer.
    EMPContext* context() const
    {
        return context_;
    }

    SharedBit compare(
        const BitVector& a_sender_share,
        const BitVector& a_receiver_share,
        const BitVector& b_sender_share,
        const BitVector& b_receiver_share,
        std::size_t threshold
    );

private:
    EMPContext* context_;
};


} // namespace approx_psi::secure
