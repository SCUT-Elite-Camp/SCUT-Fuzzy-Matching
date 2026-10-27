// Bucketing / slot-expansion tests for the secure PSI stage.
//
// These pin the two properties the plaintext reference relies on:
//   * the server keeps the first `slots` members of every bucket in ascending
//     row order (the same order the reference uses), and
//   * the client enumerates every slot for each of its buckets, with the item
//     row mapping back to all queries that share that projection.

#include "secure/name_bucketing.h"
#include "secure/name_pipeline_types.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

using namespace approx_psi::secure;
using namespace approx_psi::secure::namepipeline;

namespace {

int failures = 0;

#define CHECK(condition, message)                                              \
    do {                                                                       \
        if (!(condition)) {                                                    \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, message);      \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

using Bits = std::array<std::uint8_t, kEllBytes>;

Bits with_bit(std::size_t coordinate)
{
    Bits bits{};
    bits[coordinate >> 3] = static_cast<std::uint8_t>(1u << (7 - (coordinate & 7)));
    return bits;
}

void test_project_packed_layout()
{
    const Bits bits = with_bit(0);
    const auto projected = project_packed(bits, {7, 0, 9});
    // Selected bit 0 lands at position 1 -> 0x40; the others are zero.
    CHECK(projected[0] == 0x40, "selected coordinates must be packed big-endian in order");

    const auto empty = project_packed(bits, {});
    CHECK(std::all_of(empty.begin(), empty.end(), [](std::uint8_t b) { return b == 0; }),
          "an empty coordinate set must project to zeros");
}

void test_slot_keys_are_distinct()
{
    const Bits projection = with_bit(3);
    const auto k0 = slot_key(0, projection, 0);
    const auto k1 = slot_key(0, projection, 1);
    const auto r1 = slot_key(1, projection, 0);
    CHECK(k0 != k1, "slots within a round must differ");
    CHECK(k0 != r1, "rounds must be domain-separated");
    CHECK(k0.rfind("NPSI2", 0) == 0, "keys must carry the domain prefix");
}

void test_server_truncates_to_slots()
{
    std::vector<NameRecord> records;
    const Bits shared = with_bit(5);
    for (std::uint32_t row = 0; row < 12; ++row)
    {
        NameRecord record;
        record.index = row;
        record.identity = 100 + row;
        record.token = 9000 + row;
        record.bits = shared;  // all 12 fall into one bucket
        records.push_back(record);
    }
    const auto round = build_server_round(records, {5}, /*round=*/0, /*slots=*/4);

    CHECK(round.items.size() == 4, "a 12-member bucket must be truncated to 4 slots");
    CHECK(round.record_ordinals == std::vector<std::uint32_t>({0, 1, 2, 3}),
          "the server must keep the lowest row ordinals");
    CHECK(round.items[0].second == "9000", "the payload must be the record token");
    CHECK(round.items[3].second == "9003", "slot 3 must carry the fourth token");

    std::set<std::string> keys;
    for (const auto& item : round.items)
    {
        keys.insert(item.first);
    }
    CHECK(keys.size() == round.items.size(), "server keys must be unique");
}

void test_client_enumerates_all_slots()
{
    std::vector<NameQuery> queries;
    const Bits shared = with_bit(5);
    for (std::uint32_t row = 0; row < 3; ++row)
    {
        NameQuery query;
        query.index = row;
        query.bits = shared;
        queries.push_back(query);
    }
    NameQuery other;
    other.index = 3;
    other.bits = with_bit(6);
    queries.push_back(other);

    const auto round = build_client_round(queries, {5}, /*round=*/0, /*slots=*/8);
    CHECK(round.items.size() == 16, "two distinct buckets x 8 slots");
    CHECK(round.row_queries.size() == round.items.size(), "every item needs a row mapping");

    std::size_t shared_rows = 0;
    for (std::size_t row = 0; row < round.items.size(); ++row)
    {
        if (round.row_queries[row].size() == 3)
        {
            ++shared_rows;
            CHECK(round.row_queries[row] == std::vector<std::uint32_t>({0, 1, 2}),
                  "all queries sharing a projection must map to the same bucket");
        }
        else
        {
            CHECK(round.row_queries[row] == std::vector<std::uint32_t>({3}),
                  "a unique projection maps to its single query");
        }
        CHECK(round.items[row].second == std::to_string(row),
              "the client payload must be its own item row");
    }
    CHECK(shared_rows == 8, "every slot of the shared bucket must be present");
}

void test_server_and_client_keys_intersect()
{
    std::vector<NameRecord> records;
    std::vector<NameQuery> queries;
    const Bits shared = with_bit(5);
    for (std::uint32_t row = 0; row < 12; ++row)
    {
        NameRecord record;
        record.index = row;
        record.identity = row;
        record.token = row;
        record.bits = shared;
        records.push_back(record);
    }
    NameQuery query;
    query.index = 0;
    query.bits = shared;
    queries.push_back(query);

    const auto server = build_server_round(records, {5}, 7, 8);
    const auto client = build_client_round(queries, {5}, 7, 8);

    std::set<std::string> server_keys;
    for (const auto& item : server.items)
    {
        server_keys.insert(item.first);
    }
    std::size_t hits = 0;
    for (const auto& item : client.items)
    {
        if (server_keys.count(item.first) != 0)
        {
            ++hits;
        }
    }
    CHECK(server.items.size() == 8, "8 slots kept");
    CHECK(hits == 8, "the client must hit all 8 server slots for a shared bucket");
}

} // namespace

int main()
{
    test_project_packed_layout();
    test_slot_keys_are_distinct();
    test_server_truncates_to_slots();
    test_client_enumerates_all_slots();
    test_server_and_client_keys_intersect();
    if (failures != 0)
    {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("test_name_slot_keys: all checks passed\n");
    return 0;
}
