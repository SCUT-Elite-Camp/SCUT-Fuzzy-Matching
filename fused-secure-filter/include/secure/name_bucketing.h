#pragma once

// Projection bucketing and slot expansion for the secure PSI stage.
//
// Per round both parties project their 256-bit vectors onto the public
// coordinate set and bucket by the projected value.  A bucket key is
//
//     H("NPSI2" || round || projected_bytes || slot)
//
// The server keeps at most ``slots`` members per bucket (rank = position in
// ascending row order, which is the order the plaintext reference uses) and
// the client enumerates every slot 0..slots-1 for each of its own buckets.
// A PSI hit therefore means "this client projection matches the server bucket
// entry at this rank", and the PSI value channel carries the server record's
// opaque token.

#include "secure/name_pipeline_types.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace approx_psi::secure::namepipeline {

inline void append_le32(std::string& out, std::uint32_t value)
{
    out.push_back(static_cast<char>(value & 0xFFu));
    out.push_back(static_cast<char>((value >> 8) & 0xFFu));
    out.push_back(static_cast<char>((value >> 16) & 0xFFu));
    out.push_back(static_cast<char>((value >> 24) & 0xFFu));
}

inline void append_le16(std::string& out, std::uint16_t value)
{
    out.push_back(static_cast<char>(value & 0xFFu));
    out.push_back(static_cast<char>((value >> 8) & 0xFFu));
}

// Pack the selected bits, in coordinate order, with the numpy.packbits layout.
inline std::array<std::uint8_t, kEllBytes> project_packed(
    const std::array<std::uint8_t, kEllBytes>& bits,
    const std::vector<std::uint32_t>& coordinates)
{
    std::array<std::uint8_t, kEllBytes> out{};
    for (std::size_t position = 0; position < coordinates.size(); ++position)
    {
        if (bit_at(bits, coordinates[position]))
        {
            out[position >> 3] |= static_cast<std::uint8_t>(1u << (7 - (position & 7)));
        }
    }
    return out;
}

inline std::string slot_key(
    std::uint32_t round,
    const std::array<std::uint8_t, kEllBytes>& projection,
    std::uint32_t slot)
{
    std::string key;
    key.reserve(5 + 4 + kEllBytes + 2);
    key.append("NPSI2", 5);
    append_le32(key, round);
    key.append(reinterpret_cast<const char*>(projection.data()), kEllBytes);
    append_le16(key, static_cast<std::uint16_t>(slot));
    return key;
}

struct RoundItems
{
    std::vector<std::pair<std::string, std::string>> items;
    // Server: record ordinal per item.  Client: query indices per item.
    std::vector<std::uint32_t> record_ordinals;
    std::vector<std::vector<std::uint32_t>> row_queries;
};

// Server round input: every bucket's first ``slots`` members by ascending row
// order, keyed by (round, projection, rank) and carrying the record token.
inline RoundItems build_server_round(
    const std::vector<NameRecord>& records,
    const std::vector<std::uint32_t>& coordinates,
    std::uint32_t round,
    std::uint32_t slots)
{
    std::unordered_map<std::string, std::vector<std::uint32_t>> buckets;
    buckets.reserve(records.size());
    for (std::uint32_t ordinal = 0; ordinal < records.size(); ++ordinal)
    {
        const auto projection = project_packed(records[ordinal].bits, coordinates);
        buckets[std::string(reinterpret_cast<const char*>(projection.data()), kEllBytes)]
            .push_back(ordinal);
    }

    RoundItems out;
    out.items.reserve(records.size());
    out.record_ordinals.reserve(records.size());
    for (const auto& [projection_string, members] : buckets)
    {
        const std::size_t keep =
            std::min<std::size_t>(members.size(), static_cast<std::size_t>(slots));
        for (std::size_t rank = 0; rank < keep; ++rank)
        {
            const std::uint32_t ordinal = members[rank];
            std::array<std::uint8_t, kEllBytes> projection{};
            std::memcpy(projection.data(), projection_string.data(), kEllBytes);
            out.items.emplace_back(
                slot_key(round, projection, static_cast<std::uint32_t>(rank)),
                std::to_string(records[ordinal].token));
            out.record_ordinals.push_back(ordinal);
        }
    }
    return out;
}

// Client round input: every distinct projection of the client's queries,
// expanded to all ``slots`` ranks.  The PSI payload is the item's row index so
// the receiver can map a hit bin back to its own projection.
inline RoundItems build_client_round(
    const std::vector<NameQuery>& queries,
    const std::vector<std::uint32_t>& coordinates,
    std::uint32_t round,
    std::uint32_t slots)
{
    std::unordered_map<std::string, std::vector<std::uint32_t>> buckets;
    buckets.reserve(queries.size());
    for (std::uint32_t ordinal = 0; ordinal < queries.size(); ++ordinal)
    {
        const auto projection = project_packed(queries[ordinal].bits, coordinates);
        buckets[std::string(reinterpret_cast<const char*>(projection.data()), kEllBytes)]
            .push_back(ordinal);
    }

    RoundItems out;
    out.items.reserve(buckets.size() * slots);
    out.row_queries.reserve(buckets.size() * slots);
    for (const auto& [projection_string, query_indices] : buckets)
    {
        std::array<std::uint8_t, kEllBytes> projection{};
        std::memcpy(projection.data(), projection_string.data(), kEllBytes);
        for (std::uint32_t slot = 0; slot < slots; ++slot)
        {
            const std::uint32_t row = static_cast<std::uint32_t>(out.items.size());
            out.items.emplace_back(slot_key(round, projection, slot), std::to_string(row));
            out.row_queries.push_back(query_indices);
        }
    }
    return out;
}

} // namespace approx_psi::secure::namepipeline
