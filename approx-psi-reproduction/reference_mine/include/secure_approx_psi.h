#pragma once
#include <vector>
#include <cstdint>
#include "types.h"
#include "params.h"
#include "coproto/Socket/Socket.h"

class SecureApproxPSI {
public:
    explicit SecureApproxPSI(Parameters params);

    std::vector<PairMatch> run_alice(
        const std::vector<Element>& A,
        coproto::Socket& sock);

    std::vector<PairMatch> run_bob(
        const std::vector<Element>& B,
        coproto::Socket& sock);

private:
    Parameters params_;
};
