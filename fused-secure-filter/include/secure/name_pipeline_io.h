#pragma once

// Binary input / JSON output helpers for the end-to-end secure name pipeline.

#include "secure/name_pipeline_types.h"

#include <string>
#include <vector>

namespace approx_psi::secure::namepipeline {

// Reads the binary file written by
// ``experiments/single_attribute/rldata_comparison/scripts/secure_export.py``
// and checks the header against the parameters this build supports.
NameInput read_name_input(const std::string& path);

// Client output: the only place predictions ever appear.
void write_client_output(
    const std::string& path,
    const NameInput& input,
    const std::vector<Prediction>& predictions,
    const PipelineStats& stats);

// Server output: aggregate statistics only; no identities, no predictions.
void write_server_output(
    const std::string& path,
    const NameInput& input,
    const PipelineStats& stats);

} // namespace approx_psi::secure::namepipeline
