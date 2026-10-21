#pragma once

#include "secure/common/secure_types.h"
#include "secure/emp/emp_context.h"

#ifdef APPROX_PSI_HAS_VOLEPSI
#include "coproto/Socket/Socket.h"
#else
namespace coproto { class Socket; }
#endif

#include <cstdint>
#include <string>
#include <vector>

namespace approx_psi::secure {

// Local view of one RsCpsi output bin.
//
// IMPORTANT: in the two-process secure path this structure contains ONLY local
// information.  In particular, peer_id is never filled and the peer's flag share
// is never exchanged.  flag_share is fed directly into EMP, where the real hit
// bit is reconstructed as flag_A XOR flag_B.
struct PSIPayloadShare
{
    std::uint64_t bin = 0;
    std::uint64_t my_id = 0;
    std::uint64_t peer_id = 0; // single-process/debug path only

    // Per-party, local-only predicate: "this party has at least one element
    // mapped to this bin".  Its *count* is not comparable across parties: the
    // RsCpsi sender uses 3 cuckoo slots per element, the receiver uses 1.
    // It only feeds valid_A / valid_B into the fused EMP circuit.
    bool local_valid = false;
    std::uint8_t flag_share = 0;

    // Kept for the single-process/debug backend and for inspecting RsCpsi output.
    // The end-to-end two-process pipeline does not open or exchange these fields.
    std::string sender_payload_share;
    std::string receiver_payload_share;
};

class SSPSI
{
public:
    SSPSI();
    explicit SSPSI(EMPContext* context);
    ~SSPSI();

    void setContext(EMPContext* context);

    std::vector<PSIPayloadShare>
    run(
        const std::vector<std::pair<std::string, std::string>>& sender_input,
        const std::vector<std::pair<std::string, std::string>>& receiver_input
    );

    // Two-process entry points. Each party supplies only its own projected set.
    // They return ALL RsCpsi output bins in ascending bin order.  No hit bit,
    // peer id, or peer vector is opened here.  Each returned row contains this
    // party's local flag share and local bin->element mapping only.
    //
    // seed must be identical on both parties for the round (otherwise the
    // RsCpsi cuckoo hashing desynchronizes); it is not a secret.
    std::vector<PSIPayloadShare>
    run_as_sender(
        const std::vector<std::pair<std::string, std::string>>& my_input,
        coproto::Socket& psi_sock,
        std::uint64_t seed
    );

    std::vector<PSIPayloadShare>
    run_as_receiver(
        const std::vector<std::pair<std::string, std::string>>& my_input,
        coproto::Socket& psi_sock,
        std::uint64_t seed
    );

private:
    EMPContext* context_;
};

std::vector<PSIPayloadShare> ss_psi_volepsi_run(
    const std::vector<std::pair<std::string, std::string>>& sender_input,
    const std::vector<std::pair<std::string, std::string>>& receiver_input
);

std::vector<PSIPayloadShare> ss_psi_volepsi_run_sender(
    const std::vector<std::pair<std::string, std::string>>& my_input,
    coproto::Socket& psi_sock,
    std::uint64_t seed
);

std::vector<PSIPayloadShare> ss_psi_volepsi_run_receiver(
    const std::vector<std::pair<std::string, std::string>>& my_input,
    coproto::Socket& psi_sock,
    std::uint64_t seed
);

} // namespace approx_psi::secure
