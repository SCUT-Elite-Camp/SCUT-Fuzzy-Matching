#pragma once


#include "types.h"
#include "secure/common/share.h"
#include "secure/ss_hamming.h"
#include "secure/emp/emp_context.h"

#include <cstddef>


namespace approx_psi::secure {


struct SharedVector
{
    BitVector sender_share;
    BitVector receiver_share;

    BitVector reconstruct() const;
};


class SSVMult
{
public:
    SSVMult();
    explicit SSVMult(EMPContext* context);
    ~SSVMult();

    void setContext(EMPContext* context);

    SharedVector multiply(
        const SharedBit& c,
        const BitVector& v_sender_share,
        const BitVector& v_receiver_share
    );

private:
    EMPContext* context_;
};


} // namespace approx_psi::secure
