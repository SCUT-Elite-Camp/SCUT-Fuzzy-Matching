#pragma once

#include "approx_psi.h"
#include "secure/ss_psi.h"
#include "secure/ss_hamming.h"
#include "secure/ss_vmult.h"

#ifdef APPROX_PSI_HAS_VOLEPSI
#include "coproto/Socket/Socket.h"
#else
namespace coproto { class Socket; }
#endif

#include <cstdint>
#include <vector>

namespace approx_psi {
namespace secure {

// Legacy/single-process verification result retained for the existing fallback
// tests.  The end-to-end two-process path below does not exchange these shares.
struct VerifyShare
{
    std::uint8_t c_my_share = 0;
    std::vector<std::uint8_t> masked_my_share;
};

// Output visible to one party after the fused secure filter for one RsCpsi bin.
// matched is the final Approx-PSI membership bit for this bin. masked_local is
// either this party's own element or the all-zero vector; the peer's element is
// never revealed.
struct SecureFilterResult
{
    bool matched = false;
    std::vector<std::uint8_t> masked_local;
};

class SecureApproxPSI
{
public:
    SecureApproxPSI(
        ApproxPSIConfig config,
        SSPSI& ssPsi,
        SSHamming& ssHam,
        SSVMult& ssVMult
    );

    // Single-process fallback/debug path.
    ExperimentResult run(
        const std::vector<Element>& A,
        const std::vector<Element>& B
    );

    // Two-process secure path.  Each party supplies only its local set.  The
    // returned PairMatch records contain ONLY the local id: Alice uses a_id,
    // Bob uses b_id; the peer id and exact distance stay hidden.
    ExperimentResult run_alice(
        const std::vector<Element>& A,
        coproto::Socket& psi_sock
    );

    ExperimentResult run_bob(
        const std::vector<Element>& B,
        coproto::Socket& psi_sock
    );

    // Existing standalone/fallback verifier.
    VerifyShare secureVerify(
        const Element& a,
        const Element& b,
        std::size_t round
    );

    // Secure composition for one RsCpsi output bin.  local_element and
    // local_flag_share are private inputs of the current party.  The peer's
    // values are supplied only by the peer process to the same EMP session.
    SecureFilterResult secureFilterBin(
        const Element& local_element,
        bool local_valid,
        std::uint8_t local_flag_share
    );

    // Diagnostic only: the shared EMP context, for explicit NetIO flushes.
    EMPContext* empContext() const;

private:
    ApproxPSIConfig config_;
    SSPSI& ssPsi_;
    SSHamming& ssHam_;
    SSVMult& ssVMult_;
};

} // namespace secure
} // namespace approx_psi
