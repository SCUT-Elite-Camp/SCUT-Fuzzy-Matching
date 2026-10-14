#pragma once


#include "types.h"

#include <cstdint>
#include <vector>


namespace approx_psi::secure {


struct BitShare
{
    std::uint8_t local{0};

    BitShare() = default;

    explicit BitShare(std::uint8_t value)
        : local(value)
    {
    }
};


struct VectorShare
{
    BitVector local;

    VectorShare() = default;

    explicit VectorShare(BitVector value)
        : local(std::move(value))
    {
    }
};


BitVector reconstructXOR(
    const BitVector& a,
    const BitVector& b
);


std::pair<VectorShare, VectorShare>
makeVectorShare(
    const BitVector& value,
    std::uint64_t seed = 0
);


} // namespace approx_psi::secure
