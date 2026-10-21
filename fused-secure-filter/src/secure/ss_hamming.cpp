#include "secure/ss_hamming.h"

#ifdef APPROX_PSI_HAS_EMP
#include <emp-sh2pc/emp-sh2pc.h>
#endif

#include <iostream>
#include <random>

namespace approx_psi::secure {

SSHamming::SSHamming()
:
context_(nullptr)
{
}

SSHamming::SSHamming(EMPContext* context)
:
context_(context)
{
}

SSHamming::~SSHamming()
{
}

void SSHamming::setContext(EMPContext* context)
{
    context_ = context;
}

SharedBit SSHamming::compare(
    const BitVector& a_sender_share,
    const BitVector& a_receiver_share,
    const BitVector& b_sender_share,
    const BitVector& b_receiver_share,
    std::size_t threshold
)
{
#ifndef APPROX_PSI_HAS_EMP
    BitVector a = reconstructXOR(a_sender_share, a_receiver_share);
    BitVector b = reconstructXOR(b_sender_share, b_receiver_share);

    std::size_t dist = 0;
    for (std::size_t i = 0; i < a.size(); i++) {
        if (a[i] ^ b[i]) dist++;
    }

    bool result = dist <= threshold;

    return {
        static_cast<uint8_t>(result),
        0
    };
#else
    using namespace emp;

    int party =
        context_->role() == Role::ALICE ? ALICE : BOB;

    const std::size_t length = a_sender_share.size();

    auto* sess = context_->session();

    using Ctx = SH2PCSession::ctx_t;
    using B = Bit_T<Ctx>;
    using U32 = UInt_T<Ctx, 32>;

    std::cerr << "[" << (party == ALICE ? "alice" : "bob")
              << "] compare start, length=" << length << std::endl;

    std::vector<B> diff;
    diff.reserve(length);

    for (std::size_t i = 0; i < length; i++) {
        // Sender shares are owned by ALICE, receiver shares by BOB; each process
        // passes its own clear value and a dummy for the peer's input.
        B a_bit = sess->input<B>(
            ALICE,
            (party == ALICE) ? (a_sender_share[i] != 0) : false
        );
        B ar_bit = sess->input<B>(
            BOB,
            (party == BOB) ? (a_receiver_share[i] != 0) : false
        );

        B b_bit = sess->input<B>(
            ALICE,
            (party == ALICE) ? (b_sender_share[i] != 0) : false
        );
        B br_bit = sess->input<B>(
            BOB,
            (party == BOB) ? (b_receiver_share[i] != 0) : false
        );

        B secret_a = a_bit ^ ar_bit;
        B secret_b = b_bit ^ br_bit;

        diff.push_back(secret_a ^ secret_b);
    }

    U32 sum = U32::constant(sess->ctx(), 0);
    for (std::size_t i = 0; i < length; i++) {
        auto zero = U32::constant(sess->ctx(), 0);
        auto one = U32::constant(sess->ctx(), 1);
        sum = sum + zero.select(diff[i], one);
    }

    U32 limit = U32::constant(sess->ctx(), static_cast<std::uint64_t>(threshold));
    B result = sum <= limit;

    std::cerr << "[" << (party == ALICE ? "alice" : "bob")
              << "] compare returned" << std::endl;

    // XOR reveal: no wire traffic, each party gets its own wire LSB as its share
    // of the comparison bit. A recipient-based reveal would add a trailing
    // exchange per call and desync the two parties' streams.
    auto mine = sess->reveal(result, XOR);
    const std::uint8_t my_lsb =
        mine.has_value() ? static_cast<std::uint8_t>(*mine) : 0;

    // Each party fills only its own half; the peer half is collected by the
    // caller. The two halves XOR to the true comparison bit.
    if (party == ALICE) {
        return { my_lsb, 0 };
    }
    return { 0, my_lsb };
#endif
}

} // namespace approx_psi::secure
