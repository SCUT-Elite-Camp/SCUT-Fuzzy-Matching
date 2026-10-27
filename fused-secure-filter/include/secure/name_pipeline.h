#pragma once

// End-to-end two-party secure name matching.
//
// Server (EMP ALICE, PSI sender) holds the 10,000-row database; client (EMP
// BOB, PSI receiver) holds 200 query names.  Three phases:
//
//   1. 80 PSI rounds produce candidate tokens for each query.  Only the
//      per-round hit bins and the server's value shares cross the wire.
//   2. The client sends its (pseudonym, token list) groups; the server maps
//      tokens back to its rows.  Query vectors never leave the client.
//   3. One garbled circuit per group computes the Top-1 identity and reveals
//      it to the client only.

#include "secure/common/role.h"
#include "secure/name_pipeline_types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace approx_psi::secure::namepipeline {

struct NamePipelineOptions
{
    Role role = Role::ALICE;  // ALICE = server (PSI sender), BOB = client
    std::string alice_ip = "127.0.0.1";
    int emp_port = 0;
    int psi_port = 0;
    std::uint32_t max_rounds = 0;  // 0 = every round (smoke tests set a limit)
    std::uint32_t max_groups = 0;  // 0 = every group
};

struct NamePipelineResult
{
    std::vector<Prediction> predictions;  // client only
    PipelineStats stats;
};

NamePipelineResult run_name_pipeline(
    const NamePipelineOptions& options,
    const NameInput& input);

} // namespace approx_psi::secure::namepipeline
