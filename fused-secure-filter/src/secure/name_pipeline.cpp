#include "secure/name_pipeline.h"

#include <coproto/Socket/AsioSocket.h>
#include <emp-sh2pc/emp-sh2pc.h>
#include <emp-tool/runtime/io/net_io_channel.h>

#include "secure/emp/emp_context.h"
#include "secure/name_bucketing.h"
#include "secure/ss_psi.h"
#include "secure/top1_argmin.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace approx_psi::secure::namepipeline {

namespace {

using Clock = std::chrono::steady_clock;

// XOR payload share width (one oc::block).
constexpr std::size_t kShareBytes = 16;

double elapsed_sec(Clock::time_point start, Clock::time_point stop)
{
    return std::chrono::duration<double>(stop - start).count();
}

std::uint64_t load_le64_xor(const std::uint8_t* a, const std::uint8_t* b)
{
    std::uint64_t value = 0;
    for (int byte = 0; byte < 8; ++byte)
    {
        value |= static_cast<std::uint64_t>(a[byte] ^ b[byte]) << (8 * byte);
    }
    return value;
}

// Server -> client: its per-bin flag share and payload share.  These are
// already secret shares; opening them to the client only reveals the PSI
// intersection, which is the protocol's intended output.
void send_psi_shares(emp::NetIO* io, const std::vector<PSIPayloadShare>& bins)
{
    const std::uint32_t count = static_cast<std::uint32_t>(bins.size());
    io->send_data(&count, sizeof(count));

    std::vector<std::uint8_t> flags(count, 0);
    std::vector<std::uint8_t> payloads(static_cast<std::size_t>(count) * kShareBytes, 0);
    for (std::size_t bin = 0; bin < bins.size(); ++bin)
    {
        flags[bin] = bins[bin].flag_share;
        const std::size_t length =
            std::min(kShareBytes, bins[bin].sender_payload_share.size());
        std::memcpy(
            payloads.data() + bin * kShareBytes,
            bins[bin].sender_payload_share.data(),
            length);
    }
    if (count != 0)
    {
        io->send_data(flags.data(), static_cast<std::int64_t>(flags.size()));
        io->send_data(payloads.data(), static_cast<std::int64_t>(payloads.size()));
    }
    io->flush();
}

void recv_psi_shares(
    emp::NetIO* io,
    std::size_t expected_bins,
    std::vector<std::uint8_t>& flags,
    std::vector<std::uint8_t>& payloads)
{
    std::uint32_t count = 0;
    io->recv_data(&count, sizeof(count));
    if (count != expected_bins)
    {
        throw std::runtime_error("name_pipeline: PSI bin count mismatch between parties");
    }
    flags.assign(count, 0);
    payloads.assign(static_cast<std::size_t>(count) * kShareBytes, 0);
    if (count != 0)
    {
        io->recv_data(flags.data(), static_cast<std::int64_t>(flags.size()));
        io->recv_data(payloads.data(), static_cast<std::int64_t>(payloads.size()));
    }
}

struct ClientGroup
{
    std::uint64_t pseudonym{};
    std::uint32_t query_index{};
    std::vector<std::uint64_t> tokens;
};

struct ServerGroup
{
    std::uint64_t pseudonym{};
    std::vector<std::uint32_t> record_ordinals;
};

} // namespace

NamePipelineResult run_name_pipeline(
    const NamePipelineOptions& options,
    const NameInput& input)
{
    const bool am_server = options.role == Role::ALICE;

    if ((am_server && input.party != kPartyServer) ||
        (!am_server && input.party != kPartyClient))
    {
        throw std::runtime_error("name_pipeline: input file is for the other party");
    }

    NamePipelineResult result;
    const auto total_start = Clock::now();

    EMPContext context(options.role, options.alice_ip, options.emp_port);
    const std::string psi_address = "localhost:" + std::to_string(options.psi_port);
    auto psi_socket = coproto::asioConnect(psi_address, /*server=*/am_server);

    SSPSI ss_psi(&context);
    emp::NetIO* io = context.io();
    emp::SH2PCSession* session = context.session();
    if (io == nullptr || session == nullptr)
    {
        throw std::runtime_error("name_pipeline: EMP context is not available");
    }

    const std::uint64_t emp_sent_start = io->send_counter;
    const std::uint64_t emp_recv_start = io->recv_counter;
    const std::uint64_t psi_sent_start = psi_socket.bytesSent();
    const std::uint64_t psi_recv_start = psi_socket.bytesReceived();
    const std::uint64_t and_start = session->num_and();

    const std::uint32_t rounds = options.max_rounds > 0
        ? std::min(options.max_rounds, input.rounds)
        : input.rounds;

    // ------------------------------------------------------------------
    // Phase 1: approximate PSI, one pass per round.
    // ------------------------------------------------------------------
    std::vector<std::unordered_set<std::uint64_t>> candidate_tokens;
    if (!am_server)
    {
        candidate_tokens.resize(input.queries.size());
    }

    const auto phase1_start = Clock::now();
    for (std::uint32_t round = 0; round < rounds; ++round)
    {
        const auto& coordinates = input.projections[round];
        const std::uint64_t round_seed =
            input.seed + static_cast<std::uint64_t>(round);

        if (am_server)
        {
            auto round_items =
                build_server_round(input.records, coordinates, round, input.slots);
            auto bins = ss_psi.run_as_sender(round_items.items, psi_socket, round_seed);

            result.stats.psi_input_items += round_items.items.size();
            result.stats.psi_bins += bins.size();
            ++result.stats.psi_rounds;
            send_psi_shares(io, bins);
        }
        else
        {
            auto round_items =
                build_client_round(input.queries, coordinates, round, input.slots);
            auto bins =
                ss_psi.run_as_receiver(round_items.items, psi_socket, round_seed);

            result.stats.psi_input_items += round_items.items.size();
            result.stats.psi_bins += bins.size();
            ++result.stats.psi_rounds;

            std::vector<std::uint8_t> flags;
            std::vector<std::uint8_t> payloads;
            recv_psi_shares(io, bins.size(), flags, payloads);

            for (const auto& bin : bins)
            {
                if (!bin.local_valid) continue;
                const std::uint32_t row = static_cast<std::uint32_t>(bin.my_id);
                if (row >= round_items.row_queries.size())
                {
                    throw std::runtime_error("name_pipeline: PSI returned an unknown row id");
                }
                if (bin.bin >= flags.size())
                {
                    throw std::runtime_error("name_pipeline: PSI bin id out of range");
                }
                const bool hit = (bin.flag_share ^ flags[bin.bin]) != 0;
                if (!hit) continue;

                ++result.stats.hit_bins;
                const std::uint8_t* sender_share =
                    payloads.data() + static_cast<std::size_t>(bin.bin) * kShareBytes;
                const std::uint8_t* receiver_share =
                    reinterpret_cast<const std::uint8_t*>(
                        bin.receiver_payload_share.data());
                const std::uint64_t token = load_le64_xor(sender_share, receiver_share);
                for (const std::uint32_t query_index : round_items.row_queries[row])
                {
                    candidate_tokens[query_index].insert(token);
                }
            }
        }
    }
    const auto phase1_end = Clock::now();
    result.stats.phase1_sec = elapsed_sec(phase1_start, phase1_end);

    // ------------------------------------------------------------------
    // Phase 2: candidate alignment.  The client sends pseudonymous groups of
    // opaque tokens; the server maps tokens to its own rows.
    // ------------------------------------------------------------------
    const auto phase2_start = Clock::now();
    std::vector<ClientGroup> client_groups;
    std::vector<ServerGroup> server_groups;

    if (am_server)
    {
        std::unordered_map<std::uint64_t, std::uint32_t> token_to_ordinal;
        token_to_ordinal.reserve(input.records.size() * 2);
        for (std::uint32_t ordinal = 0; ordinal < input.records.size(); ++ordinal)
        {
            token_to_ordinal.emplace(input.records[ordinal].token, ordinal);
        }

        std::uint32_t group_count = 0;
        io->recv_data(&group_count, sizeof(group_count));
        if (group_count > 1000000u)
        {
            throw std::runtime_error("name_pipeline: implausible group count");
        }
        server_groups.resize(group_count);
        for (auto& group : server_groups)
        {
            io->recv_data(&group.pseudonym, sizeof(group.pseudonym));
            std::uint32_t count = 0;
            io->recv_data(&count, sizeof(count));
            if (count > input.records.size())
            {
                throw std::runtime_error("name_pipeline: group larger than the database");
            }
            std::vector<std::uint64_t> tokens(count);
            if (count != 0)
            {
                io->recv_data(tokens.data(),
                              static_cast<std::int64_t>(count * sizeof(std::uint64_t)));
            }
            group.record_ordinals.reserve(count);
            for (const std::uint64_t token : tokens)
            {
                const auto it = token_to_ordinal.find(token);
                if (it == token_to_ordinal.end())
                {
                    throw std::runtime_error("name_pipeline: unknown candidate token");
                }
                group.record_ordinals.push_back(it->second);
            }
            result.stats.candidates += count;
        }
        result.stats.groups = server_groups.size();
    }
    else
    {
        for (std::uint32_t query_index = 0; query_index < input.queries.size(); ++query_index)
        {
            if (candidate_tokens[query_index].empty()) continue;
            ClientGroup group;
            group.query_index = query_index;
            group.tokens.assign(
                candidate_tokens[query_index].begin(), candidate_tokens[query_index].end());
            std::sort(group.tokens.begin(), group.tokens.end());
            client_groups.push_back(std::move(group));
        }

        // The pseudonym is an opaque group id: the server never uses it, and a
        // fully random value keeps the query index out of what the server sees.
        std::mt19937_64 rng(input.seed ^ 0xC0FFEEULL);
        std::shuffle(client_groups.begin(), client_groups.end(), rng);
        for (auto& group : client_groups)
        {
            group.pseudonym = rng();
        }

        std::uint32_t group_count = static_cast<std::uint32_t>(client_groups.size());
        io->send_data(&group_count, sizeof(group_count));
        for (const auto& group : client_groups)
        {
            io->send_data(&group.pseudonym, sizeof(group.pseudonym));
            const std::uint32_t count = static_cast<std::uint32_t>(group.tokens.size());
            io->send_data(&count, sizeof(count));
            io->send_data(
                group.tokens.data(),
                static_cast<std::int64_t>(count * sizeof(std::uint64_t)));
            result.stats.candidates += count;
        }
        io->flush();
        result.stats.groups = client_groups.size();
    }
    const auto phase2_end = Clock::now();
    result.stats.phase2_sec = elapsed_sec(phase2_start, phase2_end);

    // ------------------------------------------------------------------
    // Phase 3: one garbled circuit per group.  The client is the only party
    // that learns the winning identity.
    // ------------------------------------------------------------------
    const auto phase3_start = Clock::now();
    if (am_server)
    {
        const std::size_t limit = options.max_groups > 0
            ? std::min<std::size_t>(options.max_groups, server_groups.size())
            : server_groups.size();
        std::array<std::uint8_t, kEllBytes> no_query{};
        for (std::size_t index = 0; index < limit; ++index)
        {
            const auto& group = server_groups[index];
            std::vector<Top1Candidate> candidates;
            candidates.reserve(group.record_ordinals.size());
            for (const std::uint32_t ordinal : group.record_ordinals)
            {
                const NameRecord& record = input.records[ordinal];
                Top1Candidate candidate;
                candidate.bits = record.bits;
                candidate.identity = static_cast<std::uint16_t>(record.identity);
                candidate.rid = static_cast<std::uint16_t>(record.index);
                candidates.push_back(candidate);
            }
            (void)top1_group(
                *session, true, no_query, candidates.size(), &candidates, input.distance);
            context.flush();
        }
    }
    else
    {
        const std::size_t limit = options.max_groups > 0
            ? std::min<std::size_t>(options.max_groups, client_groups.size())
            : client_groups.size();
        for (std::size_t index = 0; index < limit; ++index)
        {
            const ClientGroup& group = client_groups[index];
            const auto identity = top1_group(
                *session,
                false,
                input.queries[group.query_index].bits,
                group.tokens.size(),
                nullptr,
                input.distance);
            if (!identity.has_value())
            {
                throw std::runtime_error("name_pipeline: client received no Top-1 output");
            }

            Prediction prediction;
            prediction.query_index = group.query_index;
            prediction.candidates = static_cast<std::uint32_t>(group.tokens.size());
            prediction.identity = (*identity == kNoIdentity)
                ? -1
                : static_cast<std::int32_t>(*identity);
            result.predictions.push_back(prediction);
            context.flush();
        }

        // Queries without candidates are rejected locally; add them only when
        // every group was processed (a max_groups smoke run is partial).
        if (limit == client_groups.size())
        {
            std::vector<bool> covered(input.queries.size(), false);
            for (const auto& prediction : result.predictions)
            {
                if (prediction.query_index < covered.size())
                {
                    covered[prediction.query_index] = true;
                }
            }
            for (std::uint32_t query_index = 0; query_index < input.queries.size(); ++query_index)
            {
                if (!covered[query_index])
                {
                    result.predictions.push_back(Prediction{query_index, -1, 0});
                }
            }
        }
        std::sort(
            result.predictions.begin(),
            result.predictions.end(),
            [](const Prediction& left, const Prediction& right) {
                return left.query_index < right.query_index;
            });
    }
    const auto phase3_end = Clock::now();
    result.stats.phase3_sec = elapsed_sec(phase3_start, phase3_end);

    result.stats.psi_bytes_sent = psi_socket.bytesSent() - psi_sent_start;
    result.stats.psi_bytes_received = psi_socket.bytesReceived() - psi_recv_start;
    result.stats.emp_bytes_sent = io->send_counter - emp_sent_start;
    result.stats.emp_bytes_received = io->recv_counter - emp_recv_start;
    result.stats.and_gates = session->num_and() - and_start;
    result.stats.total_sec = elapsed_sec(total_start, Clock::now());

    return result;
}

} // namespace approx_psi::secure::namepipeline
