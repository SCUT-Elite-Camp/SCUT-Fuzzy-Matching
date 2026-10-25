#include "secure/ss_vmult.h"


#ifdef APPROX_PSI_HAS_EMP
#include <emp-sh2pc/emp-sh2pc.h>
#endif

#include <iostream>
#include <random>


namespace approx_psi::secure {


BitVector SharedVector::reconstruct() const
{
    return reconstructXOR(sender_share, receiver_share);
}


SSVMult::SSVMult()
:
context_(nullptr)
{
}


SSVMult::SSVMult(EMPContext* context)
:
context_(context)
{
}


SSVMult::~SSVMult()
{
}


void SSVMult::setContext(EMPContext* context)
{
    context_ = context;
}


SharedVector SSVMult::multiply(
    const SharedBit& c,
    const BitVector& v_sender_share,
    const BitVector& v_receiver_share
)
{
#ifndef APPROX_PSI_HAS_EMP
    BitVector v = reconstructXOR(v_sender_share, v_receiver_share);

    bool bit = c.reconstruct();

    BitVector result(v.size(), 0);
    if (bit) result = v;

    auto shares = makeVectorShare(result, 123);

    return {
        shares.first.local,
        shares.second.local
    };
#else
    using namespace emp;

    int party =
        context_->role() == Role::ALICE ? ALICE : BOB;

    std::size_t length = v_sender_share.size();

    auto* sess = context_->session();

    using Ctx = SH2PCSession::ctx_t;
    using B = Bit_T<Ctx>;

    // The choice bit is shared: ALICE holds c.sender_share, BOB holds
    // c.receiver_share, and c = sender ^ receiver. The vector is shared the same
    // way. Each process supplies only its own share.
    B c_sender = sess->input<B>(
        ALICE,
        (party == ALICE) ? (c.sender_share != 0) : false
    );
    B c_receiver = sess->input<B>(
        BOB,
        (party == BOB) ? (c.receiver_share != 0) : false
    );
    B choice = c_sender ^ c_receiver;

    std::cerr << "[" << (party == ALICE ? "alice" : "bob")
              << "] multiply start, length=" << length << std::endl;

    BitVector output(length, 0);

    for (std::size_t i = 0; i < length; i++) {
        B v_sender = sess->input<B>(
            ALICE,
            (party == ALICE) ? (v_sender_share[i] != 0) : false
        );
        B v_receiver = sess->input<B>(
            BOB,
            (party == BOB) ? (v_receiver_share[i] != 0) : false
        );
        B v = v_sender ^ v_receiver;

        // c ? v : 0, without opening either operand.
        B zero = B::constant(sess->ctx(), false);
        B result = zero.select(choice, v);

        // XOR reveal inside the loop: no wire traffic, so 128 iterations cannot
        // accumulate the trailing exchanges a recipient reveal would add.
        auto out = sess->reveal(result, XOR);
        output[i] = out.has_value() ? static_cast<std::uint8_t>(*out) : 0;
    }

    std::cerr << "[" << (party == ALICE ? "alice" : "bob")
              << "] multiply returned" << std::endl;

    // Each party fills only its own half of the vector share; the peer half is
    // collected by the caller and the two halves XOR to the true product.
    if (party == ALICE) {
        return { output, BitVector(length, 0) };
    }
    return { BitVector(length, 0), output };
#endif
}


} // namespace approx_psi::secure
