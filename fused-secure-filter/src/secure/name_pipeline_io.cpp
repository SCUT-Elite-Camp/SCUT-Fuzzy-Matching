#include "secure/name_pipeline_io.h"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace approx_psi::secure::namepipeline {

namespace {

template <class T>
T read_scalar(std::ifstream& in)
{
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in)
    {
        throw std::runtime_error("name_pipeline_io: unexpected end of input file");
    }
    return value;
}

void read_bytes(std::ifstream& in, void* buffer, std::size_t length)
{
    in.read(reinterpret_cast<char*>(buffer), static_cast<std::streamsize>(length));
    if (!in)
    {
        throw std::runtime_error("name_pipeline_io: truncated input file");
    }
}

void write_stats(std::ostream& out, const PipelineStats& stats)
{
    out << std::fixed << std::setprecision(6);
    out << "  \"stats\": {\n";
    out << "    \"phase1_sec\": " << stats.phase1_sec << ",\n";
    out << "    \"phase2_sec\": " << stats.phase2_sec << ",\n";
    out << "    \"phase3_sec\": " << stats.phase3_sec << ",\n";
    out << "    \"total_sec\": " << stats.total_sec << ",\n";
    out << "    \"psi_bytes_sent\": " << stats.psi_bytes_sent << ",\n";
    out << "    \"psi_bytes_received\": " << stats.psi_bytes_received << ",\n";
    out << "    \"emp_bytes_sent\": " << stats.emp_bytes_sent << ",\n";
    out << "    \"emp_bytes_received\": " << stats.emp_bytes_received << ",\n";
    out << "    \"psi_rounds\": " << stats.psi_rounds << ",\n";
    out << "    \"psi_input_items\": " << stats.psi_input_items << ",\n";
    out << "    \"psi_bins\": " << stats.psi_bins << ",\n";
    out << "    \"hit_bins\": " << stats.hit_bins << ",\n";
    out << "    \"candidates\": " << stats.candidates << ",\n";
    out << "    \"groups\": " << stats.groups << ",\n";
    out << "    \"and_gates\": " << stats.and_gates << "\n";
    out << "  }\n";
}

} // namespace

NameInput read_name_input(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error("name_pipeline_io: cannot open input " + path);
    }

    NameInput input;
    const std::uint32_t magic = read_scalar<std::uint32_t>(in);
    const std::uint32_t version = read_scalar<std::uint32_t>(in);
    input.party = read_scalar<std::uint32_t>(in);
    input.ell_bits = read_scalar<std::uint32_t>(in);
    input.ell_bytes = read_scalar<std::uint32_t>(in);
    input.distance = read_scalar<std::uint32_t>(in);
    input.rounds = read_scalar<std::uint32_t>(in);
    input.slots = read_scalar<std::uint32_t>(in);
    // Header field kept for format compatibility with secure_export.py.
    (void)read_scalar<std::uint32_t>(in);
    input.seed = read_scalar<std::uint64_t>(in);
    const std::uint32_t count = read_scalar<std::uint32_t>(in);
    const std::uint32_t projection_count = read_scalar<std::uint32_t>(in);

    if (magic != kMagic || version != kVersion)
    {
        throw std::runtime_error("name_pipeline_io: bad magic/version");
    }
    if (input.ell_bits != kEllBits || input.ell_bytes != kEllBytes)
    {
        throw std::runtime_error("name_pipeline_io: this build expects 256-bit vectors");
    }
    if (projection_count != input.rounds)
    {
        throw std::runtime_error("name_pipeline_io: projection count != rounds");
    }

    input.projections.resize(projection_count);
    for (auto& projection : input.projections)
    {
        const std::uint32_t size = read_scalar<std::uint32_t>(in);
        if (size > input.ell_bits)
        {
            throw std::runtime_error("name_pipeline_io: projection wider than the vector");
        }
        projection.resize(size);
        for (auto& coordinate : projection)
        {
            coordinate = read_scalar<std::uint32_t>(in);
            if (coordinate >= input.ell_bits)
            {
                throw std::runtime_error("name_pipeline_io: projection coordinate out of range");
            }
        }
    }

    if (input.party == kPartyServer)
    {
        input.records.resize(count);
        for (auto& record : input.records)
        {
            record.index = read_scalar<std::uint32_t>(in);
            record.identity = read_scalar<std::uint32_t>(in);
            record.token = read_scalar<std::uint64_t>(in);
            read_bytes(in, record.bits.data(), kEllBytes);
            if (record.identity > 0xFFFFu)
            {
                throw std::runtime_error("name_pipeline_io: identity does not fit in 16 bits");
            }
        }
    }
    else if (input.party == kPartyClient)
    {
        input.queries.resize(count);
        for (std::uint32_t ordinal = 0; ordinal < count; ++ordinal)
        {
            NameQuery& query = input.queries[ordinal];
            query.index = read_scalar<std::uint32_t>(in);
            if (query.index != ordinal)
            {
                throw std::runtime_error(
                    "name_pipeline_io: query indices must be contiguous from zero");
            }
            read_bytes(in, query.bits.data(), kEllBytes);
        }
    }
    else
    {
        throw std::runtime_error("name_pipeline_io: unknown party id in input file");
    }

    return input;
}

void write_client_output(
    const std::string& path,
    const NameInput& input,
    const std::vector<Prediction>& predictions,
    const PipelineStats& stats)
{
    std::ofstream out(path);
    if (!out)
    {
        throw std::runtime_error("name_pipeline_io: cannot write " + path);
    }
    out << "{\n";
    out << "  \"party\": \"client\",\n";
    out << "  \"seed\": " << input.seed << ",\n";
    out << "  \"rounds\": " << input.rounds << ",\n";
    out << "  \"slots\": " << input.slots << ",\n";
    out << "  \"distance_threshold\": " << input.distance << ",\n";
    out << "  \"queries\": " << input.queries.size() << ",\n";
    out << "  \"predictions\": [\n";
    for (std::size_t i = 0; i < predictions.size(); ++i)
    {
        const auto& row = predictions[i];
        out << "    {\"query_index\": " << row.query_index
            << ", \"identity\": " << row.identity
            << ", \"identities\": [";
        for (std::size_t k = 0; k < row.identities.size(); ++k)
        {
            out << (k == 0 ? "" : ", ") << row.identities[k];
        }
        out << "]"
            << ", \"candidates\": " << row.candidates << "}";
        out << (i + 1 == predictions.size() ? "\n" : ",\n");
    }
    out << "  ],\n";
    write_stats(out, stats);
    out << "}\n";
}

void write_server_output(
    const std::string& path,
    const NameInput& input,
    const PipelineStats& stats)
{
    std::ofstream out(path);
    if (!out)
    {
        throw std::runtime_error("name_pipeline_io: cannot write " + path);
    }
    out << "{\n";
    out << "  \"party\": \"server\",\n";
    out << "  \"seed\": " << input.seed << ",\n";
    out << "  \"rounds\": " << input.rounds << ",\n";
    out << "  \"slots\": " << input.slots << ",\n";
    out << "  \"distance_threshold\": " << input.distance << ",\n";
    out << "  \"records\": " << input.records.size() << ",\n";
    write_stats(out, stats);
    out << "}\n";
}

} // namespace approx_psi::secure::namepipeline
