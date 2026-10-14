#include <emp-tool/runtime/io/net_io_channel.h>
#include "secure/secure_approx_psi.h"

#include "projection.h"
#include "hamming.h"

#ifdef APPROX_PSI_HAS_EMP
#include "secure/ss_ham_vmult_impl.h"
#include "secure/secure_filter_impl.h"
#include <emp-sh2pc/emp-sh2pc.h>
#endif

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace approx_psi {
namespace secure {

namespace {

std::unordered_map<std::size_t, const Element*>
indexById(const std::vector<Element>& elements)
{
    std::unordered_map<std::size_t, const Element*> index;
    index.reserve(elements.size());
    for (const auto& element : elements) {
        index.emplace(element.id, &element);
    }
    return index;
}

std::uint64_t roundSeed(std::uint64_t base, std::size_t round)
{
    return base * 1000003ull + round;
}

std::vector<std::pair<std::string, std::string>>
buildSecureInput(
    const std::vector<Element>& elements,
    const BitIndex& projection,
    std::mt19937_64& rng)
{
    auto projected = projectSetWithRandomRepresentative(elements, projection, rng);

    std::vector<std::pair<std::string, std::string>> input;
    input.reserve(projected.size());
    for (const auto& item : projected) {
        input.push_back({item.first, std::to_string(item.second.original.id)});
    }

    std::sort(input.begin(), input.end(),
        [](const auto& x, const auto& y) { return x.first < y.first; });
    return input;
}

static const char* diagTag(bool am_alice)
{
    return am_alice ? "alice" : "bob";
}

} // namespace

SecureApproxPSI::SecureApproxPSI(
    ApproxPSIConfig config,
    SSPSI& ssPsi,
    SSHamming& ssHam,
    SSVMult& ssVMult)
    : config_(config), ssPsi_(ssPsi), ssHam_(ssHam), ssVMult_(ssVMult)
{
}

// -----------------------------------------------------------------------------
// Single-process fallback/debug path.  Kept to preserve the existing unit tests.
// This is NOT the end-to-end secure protocol.
// -----------------------------------------------------------------------------
ExperimentResult SecureApproxPSI::run(
    const std::vector<Element>& A,
    const std::vector<Element>& B)
{
    ExperimentResult result;
    const auto& params = config_.parameters;
    const std::size_t d = params.d;

    auto aIndex = indexById(A);
    auto bIndex = indexById(B);
    std::set<std::pair<std::size_t, std::size_t>> seen;
    std::mt19937_64 rng(config_.seed);

    for (std::size_t round = 0; round < params.k; ++round) {
        auto projection = sampleProjection(params.length, params.p, rng);
        auto Aproj = projectSetWithRandomRepresentative(A, projection, rng);
        auto Bproj = projectSetWithRandomRepresentative(B, projection, rng);

        std::vector<std::pair<std::string, std::string>> secureInputA;
        std::vector<std::pair<std::string, std::string>> secureInputB;
        for (auto& item : Aproj) {
            secureInputA.push_back({item.first, std::to_string(item.second.original.id)});
        }
        for (auto& item : Bproj) {
            secureInputB.push_back({item.first, std::to_string(item.second.original.id)});
        }

        auto candidates = ssPsi_.run(secureInputA, secureInputB);
        result.stats.projection_intersections += candidates.size();

        for (const auto& candidate : candidates) {
            const std::size_t a_id = static_cast<std::size_t>(candidate.my_id);
            const std::size_t b_id = static_cast<std::size_t>(candidate.peer_id);
            auto aIt = aIndex.find(a_id);
            auto bIt = bIndex.find(b_id);
            if (aIt == aIndex.end() || bIt == bIndex.end()) continue;

            VerifyShare share = secureVerify(*aIt->second, *bIt->second, round);
            result.stats.hamming_checks++;

            const bool accepted = (share.c_my_share != 0);
            std::size_t dist = d + 1;
            if (accepted) {
                std::size_t count = 0;
                for (std::uint8_t bit : share.masked_my_share) {
                    if (bit) ++count;
                }
                dist = count;
            }

            if (dist <= d && seen.insert({a_id, b_id}).second) {
                result.matches.push_back(PairMatch{a_id, b_id, dist, round});
            }
        }
    }

    result.stats.rounds = params.k;
    return result;
}

// -----------------------------------------------------------------------------
// End-to-end two-process secure composition.
// -----------------------------------------------------------------------------
namespace {

void run_round_secure(
    SecureApproxPSI& self,
    const std::vector<Element>& local,
    coproto::Socket& psi_sock,
    bool am_alice,
    std::size_t round,
    SSPSI& ssPsi,
    const ApproxPSIConfig& config,
    ExperimentResult& result,
    std::set<std::size_t>& seen_local)
{
    const auto& params = config.parameters;

    std::cerr << "[" << diagTag(am_alice) << "-round " << round
              << "] enter, local_size=" << local.size() << std::endl;

    std::mt19937_64 rng(roundSeed(config.seed, round));
    auto projection = sampleProjection(params.length, params.p, rng);
    result.stats.total_projection_coordinates += projection.size();

    std::mt19937_64 projRng(roundSeed(config.seed, round));
    auto secureInput = buildSecureInput(local, projection, projRng);

    std::cerr << "[" << diagTag(am_alice) << "-round " << round
              << "] RsCpsi start" << std::endl;

    auto bins = am_alice
        ? ssPsi.run_as_sender(secureInput, psi_sock, roundSeed(config.seed, round))
        : ssPsi.run_as_receiver(secureInput, psi_sock, roundSeed(config.seed, round));

    std::sort(bins.begin(), bins.end(),
        [](const PSIPayloadShare& x, const PSIPayloadShare& y) {
            return x.bin < y.bin;
        });

    std::cerr << "[" << diagTag(am_alice) << "-round " << round
              << "] RsCpsi done, secure_bins=" << bins.size() << std::endl;

    std::size_t numLocalValid = 0;
    for (const auto& bin : bins) {
        if (bin.local_valid) ++numLocalValid;
    }

    std::cerr << "[" << diagTag(am_alice) << "-round " << round
              << "] psi sizes: my_input=" << secureInput.size()
              << ", bins=" << bins.size()
              << ", local_valid=" << numLocalValid << std::endl;

    auto localIndex = indexById(local);

    // SECURITY CRITICAL: iterate over ALL bins.  We never branch on the cleartext
    // PSI hit, because the cleartext hit never exists outside EMP.
    for (const auto& bin : bins) {
        bool have_local = bin.local_valid;
        Element localElement;

        if (have_local) {
            auto it = localIndex.find(static_cast<std::size_t>(bin.my_id));
            if (it != localIndex.end()) {
                localElement = *it->second;
            } else {
                have_local = false;
            }
        }

        if (!have_local) {
            localElement.id = static_cast<std::size_t>(bin.my_id);
            localElement.bits.assign(params.length, 0);
        }

        SecureFilterResult out = self.secureFilterBin(
            localElement, have_local, bin.flag_share);

        result.stats.hamming_checks++;

        // matched is already the FINAL secure predicate:
        //   hit && valid_A && valid_B && H(a,b)<=d
        // No peer id, peer vector, hit bit, or exact distance is reconstructed.
        if (out.matched && have_local && seen_local.insert(localElement.id).second) {
            const std::size_t hidden = std::numeric_limits<std::size_t>::max();
            const std::size_t hidden_dist = std::numeric_limits<std::size_t>::max();

            if (am_alice) {
                result.matches.push_back(PairMatch{
                    localElement.id, hidden, hidden_dist, round});
            } else {
                result.matches.push_back(PairMatch{
                    hidden, localElement.id, hidden_dist, round});
            }
        }
    }

    std::cerr << "[" << diagTag(am_alice) << "-round " << round
              << "] secureFilterBin done" << std::endl;
    std::cerr << "[" << diagTag(am_alice) << "-round " << round
              << "] done, matches=" << result.matches.size() << std::endl;

    // EMP NetIO requires flush() at the end of any protocol step that ends
    // in sends, before blocking on a different channel.  Without this, the
    // sender's last bytes stay in the 32 KiB send buffer, the receiver waits
    // for them, and the next RsCpsi handshake on psi_sock desynchronizes.
    // This was the actual cause of the round-0/bin-54 hang.
    {
        auto* ctx = self.empContext();
        if (ctx != nullptr && ctx->io() != nullptr) {
            ctx->io()->flush();
        }
    }
}

ExperimentResult run_two_party(
    SecureApproxPSI& self,
    const std::vector<Element>& local,
    coproto::Socket& psi_sock,
    bool am_alice,
    SSPSI& ssPsi,
    const ApproxPSIConfig& config)
{
    ExperimentResult result;
    std::set<std::size_t> seen_local;

    // Diagnostic: surface k so we can tell a last-round teardown failure from a
    // mid-run one.
    std::cerr << "[" << diagTag(am_alice) << "] starting rounds, k="
              << config.parameters.k << ", p=" << config.parameters.p
              << std::endl;

    for (std::size_t round = 0; round < config.parameters.k; ++round) {
        if (round > 0) {
            std::cerr << "[" << diagTag(am_alice) << "] between round "
                      << (round - 1) << " and " << round << std::endl;
        }
        run_round_secure(
            self, local, psi_sock, am_alice, round,
            ssPsi, config, result, seen_local);

        std::cerr << "[" << diagTag(am_alice) << "] round " << round << "/"
                  << config.parameters.k << " done" << std::endl;
    }

    result.stats.rounds = config.parameters.k;
    return result;
}

} // namespace

ExperimentResult SecureApproxPSI::run_alice(
    const std::vector<Element>& A,
    coproto::Socket& psi_sock)
{
    return run_two_party(*this, A, psi_sock, true, ssPsi_, config_);
}

ExperimentResult SecureApproxPSI::run_bob(
    const std::vector<Element>& B,
    coproto::Socket& psi_sock)
{
    return run_two_party(*this, B, psi_sock, false, ssPsi_, config_);
}

EMPContext* SecureApproxPSI::empContext() const
{
    return ssHam_.context();
}

SecureFilterResult SecureApproxPSI::secureFilterBin(
    const Element& local_element,
    bool local_valid,
    std::uint8_t local_flag_share)
{
#ifdef APPROX_PSI_HAS_EMP
    using namespace emp;
    using Ctx = SH2PCSession::ctx_t;
    using B = Bit_T<Ctx>;

    auto* context = ssHam_.context();
    auto* sess = context ? context->session() : nullptr;
    if (sess == nullptr) {
        throw std::runtime_error("secureFilterBin requires an active EMP session");
    }

    const bool am_alice = (context->role() == Role::ALICE);
    const std::size_t length = config_.parameters.length;

    // Each party contributes only its local RsCpsi flag share.
    B flag_a = sess->input<B>(
        ALICE, am_alice ? (local_flag_share != 0) : false);
    B flag_b = sess->input<B>(
        BOB, !am_alice ? (local_flag_share != 0) : false);

    // Mapping validity stays private as well.  This prevents empty/padded bins
    // from accidentally matching the all-zero vector.
    B valid_a = sess->input<B>(ALICE, am_alice ? local_valid : false);
    B valid_b = sess->input<B>(BOB, !am_alice ? local_valid : false);

    std::vector<B> a_wires;
    std::vector<B> b_wires;
    a_wires.reserve(length);
    b_wires.reserve(length);

    for (std::size_t i = 0; i < length; ++i) {
        const bool local_bit =
            local_valid && i < local_element.bits.size() &&
            ((local_element.bits[i] & 1u) != 0);

        B a = sess->input<B>(ALICE, am_alice ? local_bit : false);
        B b = sess->input<B>(BOB, !am_alice ? local_bit : false);
        a_wires.push_back(a);
        b_wires.push_back(b);
    }

    auto wires = secure_filter_impl<Ctx>(
        flag_a, flag_b, valid_a, valid_b,
        a_wires, b_wires, config_.parameters.d);

    // The final membership bit is legitimate output to both parties.  Revealing
    // it does NOT reveal whether failure came from PSI non-hit or Hamming reject.
    auto keep_to_alice = sess->reveal(wires.keep, ALICE);
    auto keep_to_bob = sess->reveal(wires.keep, BOB);

    const bool matched = am_alice
        ? (keep_to_alice.has_value() && *keep_to_alice)
        : (keep_to_bob.has_value() && *keep_to_bob);

    // F_ssVMult output: open only each party's own masked payload.
    std::vector<std::uint8_t> masked_local(length, 0);
    for (std::size_t i = 0; i < length; ++i) {
        auto out_a = sess->reveal(wires.masked_a[i], ALICE);
        auto out_b = sess->reveal(wires.masked_b[i], BOB);

        if (am_alice) {
            masked_local[i] =
                (out_a.has_value() && *out_a) ? std::uint8_t{1} : std::uint8_t{0};
        } else {
            masked_local[i] =
                (out_b.has_value() && *out_b) ? std::uint8_t{1} : std::uint8_t{0};
        }
    }

    // Internal consistency check: if the final bit is one, the owner-only
    // F_ssVMult output must equal that owner's original vector.
    if (matched && local_valid) {
        for (std::size_t i = 0; i < length; ++i) {
            const std::uint8_t expected =
                (i < local_element.bits.size())
                    ? static_cast<std::uint8_t>(local_element.bits[i] & 1u)
                    : std::uint8_t{0};
            if (masked_local[i] != expected) {
                throw std::runtime_error(
                    "secureFilterBin: owner masked payload mismatch");
            }
        }
    }

    return SecureFilterResult{matched, std::move(masked_local)};
#else
    (void)local_element;
    (void)local_valid;
    (void)local_flag_share;
    throw std::runtime_error("secureFilterBin built without EMP support");
#endif
}

// -----------------------------------------------------------------------------
// Legacy standalone/fallback verifier retained for existing tests.
// -----------------------------------------------------------------------------
VerifyShare SecureApproxPSI::secureVerify(
    const Element& a,
    const Element& b,
    std::size_t round)
{
#ifdef APPROX_PSI_HAS_EMP
    using namespace emp;
    using Ctx = SH2PCSession::ctx_t;
    using B = Bit_T<Ctx>;

    auto* context = ssHam_.context();
    auto* sess = context->session();
    if (sess == nullptr) {
        return VerifyShare{};
    }

    const int party = (context->role() == Role::ALICE) ? ALICE : BOB;
    const std::size_t length = a.bits.size();

    auto a_share = makeVectorShare(a.bits, config_.seed + round * 2);
    auto b_share = makeVectorShare(b.bits, config_.seed + round * 2 + 1);

    std::vector<B> a_wires;
    std::vector<B> b_wires;
    a_wires.reserve(length);
    b_wires.reserve(length);

    for (std::size_t i = 0; i < length; ++i) {
        B a_s = sess->input<B>(
            ALICE, (party == ALICE) ? (a_share.first.local[i] != 0) : false);
        B a_r = sess->input<B>(
            BOB, (party == BOB) ? (a_share.second.local[i] != 0) : false);
        B b_s = sess->input<B>(
            ALICE, (party == ALICE) ? (b_share.first.local[i] != 0) : false);
        B b_r = sess->input<B>(
            BOB, (party == BOB) ? (b_share.second.local[i] != 0) : false);

        a_wires.push_back(a_s ^ a_r);
        b_wires.push_back(b_s ^ b_r);
    }

    auto masked_wires =
        ss_ham_and_vmult_impl<Ctx>(a_wires, b_wires, config_.parameters.d);

    std::vector<std::uint8_t> masked_my(masked_wires.size(), 0);
    for (std::size_t i = 0; i < masked_wires.size(); ++i) {
        auto r = sess->reveal(masked_wires[i], XOR);
        masked_my[i] = (r.has_value() && *r) ? std::uint8_t{1} : std::uint8_t{0};
    }

    VerifyShare ret;
    ret.masked_my_share = std::move(masked_my);
    return ret;
#else
    const std::size_t d = config_.parameters.d;

    BitVector diff(a.bits.size(), 0);
    for (std::size_t i = 0; i < a.bits.size() && i < b.bits.size(); ++i) {
        diff[i] = static_cast<std::uint8_t>((a.bits[i] ^ b.bits[i]) & 1u);
    }

    std::size_t dist = 0;
    for (std::uint8_t v : diff) {
        if (v) ++dist;
    }

    VerifyShare ret;
    ret.c_my_share = (dist <= d) ? std::uint8_t{1} : std::uint8_t{0};
    ret.masked_my_share = std::move(diff);
    return ret;
#endif
}

} // namespace secure
} // namespace approx_psi
