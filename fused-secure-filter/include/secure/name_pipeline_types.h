#pragma once

// Shared types for the end-to-end secure name-matching pipeline.
//
// The pipeline consumes the binary files written by
// ``experiments/single_attribute/rldata_comparison/scripts/secure_export.py``.
// Names are encoded locally by each party;
// this layer only ever sees 256-bit vectors, identities (server side), opaque
// 64-bit record tokens (server side) and query indices (client side).

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace approx_psi::secure::namepipeline {

inline constexpr std::uint32_t kMagic = 0x4E505349u;  // "NPSI"
inline constexpr std::uint32_t kVersion = 1;
inline constexpr std::uint32_t kPartyServer = 1;
inline constexpr std::uint32_t kPartyClient = 2;
inline constexpr std::uint32_t kEllBits = 256;
inline constexpr std::size_t kEllBytes = kEllBits / 8;
inline constexpr std::uint16_t kNoIdentity = 0xFFFFu;

struct NameRecord
{
    std::uint32_t index{};
    std::uint32_t identity{};
    std::uint64_t token{};
    std::array<std::uint8_t, kEllBytes> bits{};
};

struct NameQuery
{
    std::uint32_t index{};
    std::array<std::uint8_t, kEllBytes> bits{};
};

struct NameInput
{
    std::uint32_t party{};
    std::uint32_t ell_bits{};
    std::uint32_t ell_bytes{};
    std::uint32_t distance{};
    std::uint32_t rounds{};
    std::uint32_t slots{};
    std::uint64_t seed{};
    std::vector<std::vector<std::uint32_t>> projections;
    std::vector<NameRecord> records;  // server input only
    std::vector<NameQuery> queries;   // client input only
};

// One candidate row as it enters the Top-1 circuit.  ``rid`` is the server's
// row index and is used only inside the circuit as the deterministic tie-break
// for equal distances; it is never revealed.
struct Top1Candidate
{
    std::array<std::uint8_t, kEllBytes> bits{};
    std::uint16_t identity{};
    std::uint16_t rid{};
};

struct Prediction
{
    std::uint32_t query_index{};
    std::int32_t identity{-1};
    std::uint32_t candidates{};
};

struct PipelineStats
{
    double phase1_sec{0.0};
    double phase2_sec{0.0};
    double phase3_sec{0.0};
    double total_sec{0.0};
    std::uint64_t psi_bytes_sent{0};
    std::uint64_t psi_bytes_received{0};
    std::uint64_t emp_bytes_sent{0};
    std::uint64_t emp_bytes_received{0};
    std::uint64_t psi_rounds{0};
    std::uint64_t psi_input_items{0};
    std::uint64_t psi_bins{0};
    std::uint64_t hit_bins{0};
    std::uint64_t candidates{0};
    std::uint64_t groups{0};
    std::uint64_t and_gates{0};
};

// numpy.packbits layout: bit j lives in byte j/8 at position 7-(j%8).
inline bool bit_at(const std::array<std::uint8_t, kEllBytes>& packed, std::size_t j)
{
    return ((packed[j >> 3] >> (7 - (j & 7))) & 1u) != 0;
}

} // namespace approx_psi::secure::namepipeline
