#include "secure/common/share.h"

#include <random>


namespace approx_psi::secure {


BitVector reconstructXOR(
    const BitVector& a,
    const BitVector& b
)
{
    BitVector result;
    result.resize(a.size());

    for (std::size_t i = 0; i < a.size(); i++) {
        result[i] = a[i] ^ b[i];
    }

    return result;
}


std::pair<VectorShare, VectorShare>
makeVectorShare(
    const BitVector& value,
    std::uint64_t seed
)
{
    std::mt19937_64 rng(seed);

    BitVector first;
    first.resize(value.size());

    BitVector second;
    second.resize(value.size());

    for (std::size_t i = 0; i < value.size(); i++) {
        first[i] = static_cast<std::uint8_t>(rng() & 1);
        second[i] = first[i] ^ value[i];
    }

    return {
        VectorShare(first),
        VectorShare(second)
    };
}


} // namespace approx_psi::secure
