// Two-process driver for the end-to-end secure name pipeline.
//
//   server (party 1 / EMP ALICE / PSI sender): database side
//   client (party 2 / EMP BOB  / PSI receiver): query side
//
// Usage:
//   name_secure_pipeline --party server --emp-port P --psi-port Q
//       --input server.bin --output server.json
//   name_secure_pipeline --party client --emp-port P --psi-port Q
//       --alice-ip 127.0.0.1 --input client.bin --output client.json
//
// Optional smoke-test limits: --max-rounds K, --max-groups G (must match on
// both parties).

#include "secure/name_pipeline.h"
#include "secure/name_pipeline_io.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace approx_psi::secure;
using namespace approx_psi::secure::namepipeline;

namespace {

void usage(const char* program)
{
    std::cerr
        << "Usage:\n"
        << "  " << program
        << " --party server|client --emp-port P --psi-port Q\n"
        << "     --input <file.bin> --output <file.json>\n"
        << "     [--alice-ip 127.0.0.1] [--max-rounds K] [--max-groups G]\n"
        << "     [--top-k K]  (both parties must use the same K; default 1)\n";
}

} // namespace

int main(int argc, char** argv)
{
    std::string party;
    std::string alice_ip = "127.0.0.1";
    std::string input_path;
    std::string output_path;
    int emp_port = 0;
    int psi_port = 0;
    std::uint32_t max_rounds = 0;
    std::uint32_t max_groups = 0;
    std::uint32_t top_k = 1;

    try
    {
        for (int index = 1; index < argc; ++index)
        {
            const std::string argument = argv[index];
            auto value = [&](const char* name) -> std::string {
                if (index + 1 >= argc)
                {
                    throw std::runtime_error(std::string("missing value for ") + name);
                }
                return argv[++index];
            };

            if (argument == "--party") party = value("--party");
            else if (argument == "--emp-port") emp_port = std::stoi(value("--emp-port"));
            else if (argument == "--psi-port") psi_port = std::stoi(value("--psi-port"));
            else if (argument == "--alice-ip") alice_ip = value("--alice-ip");
            else if (argument == "--input") input_path = value("--input");
            else if (argument == "--output") output_path = value("--output");
            else if (argument == "--max-rounds")
                max_rounds = static_cast<std::uint32_t>(std::stoul(value("--max-rounds")));
            else if (argument == "--max-groups")
                max_groups = static_cast<std::uint32_t>(std::stoul(value("--max-groups")));
            else if (argument == "--top-k")
                top_k = static_cast<std::uint32_t>(std::stoul(value("--top-k")));
            else if (argument == "--help") { usage(argv[0]); return 0; }
            else throw std::runtime_error("unknown argument: " + argument);
        }

        if (party != "server" && party != "client")
        {
            throw std::runtime_error("--party must be 'server' or 'client'");
        }
        if (input_path.empty() || output_path.empty())
        {
            throw std::runtime_error("--input and --output are required");
        }
        if (emp_port <= 0 || psi_port <= 0)
        {
            throw std::runtime_error("--emp-port and --psi-port are required");
        }
        if (top_k == 0 || top_k > 16)
        {
            throw std::runtime_error("--top-k must be in 1..16");
        }
    }
    catch (const std::exception& error)
    {
        std::cerr << "argument error: " << error.what() << "\n";
        usage(argv[0]);
        return 1;
    }

    const bool am_server = party == "server";
    NamePipelineOptions options;
    options.role = am_server ? Role::ALICE : Role::BOB;
    options.alice_ip = alice_ip;
    options.emp_port = emp_port;
    options.psi_port = psi_port;
    options.max_rounds = max_rounds;
    options.max_groups = max_groups;
    options.top_k = top_k;

    try
    {
        const NameInput input = read_name_input(input_path);
        const NamePipelineResult result = run_name_pipeline(options, input);
        if (am_server)
        {
            write_server_output(output_path, input, result.stats);
        }
        else
        {
            write_client_output(output_path, input, result.predictions, result.stats);
        }

        std::cout << "party=" << party
                  << " phase1_sec=" << result.stats.phase1_sec
                  << " phase2_sec=" << result.stats.phase2_sec
                  << " phase3_sec=" << result.stats.phase3_sec
                  << " total_sec=" << result.stats.total_sec
                  << " groups=" << result.stats.groups
                  << " candidates=" << result.stats.candidates
                  << " hit_bins=" << result.stats.hit_bins
                  << " and_gates=" << result.stats.and_gates
                  << std::endl;
    }
    catch (const std::exception& error)
    {
        std::cerr << "[name-pipeline:" << party << "] ERROR: " << error.what()
                  << std::endl;
        return 1;
    }

    return 0;
}
