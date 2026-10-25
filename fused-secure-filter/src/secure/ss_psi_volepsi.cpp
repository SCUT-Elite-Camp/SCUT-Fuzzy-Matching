#include "secure/ss_psi.h"

#ifdef APPROX_PSI_HAS_VOLEPSI

#include "volePSI/RsCpsi.h"

#include "coproto/Socket/LocalAsyncSock.h"
#include "cryptoTools/Common/BitVector.h"
#include "cryptoTools/Crypto/PRNG.h"
#include "cryptoTools/Crypto/RandomOracle.h"

#include <array>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#endif

namespace approx_psi::secure {

#ifdef APPROX_PSI_HAS_VOLEPSI

namespace {

constexpr std::size_t kPayloadBytes = sizeof(oc::block);
constexpr std::uint64_t kStatSecParam = 40;
constexpr std::uint64_t kNumThreads = 1;
constexpr std::uint64_t kHammingThreshold = 4;

oc::block keyToBlock(const std::string& key)
{
    oc::RandomOracle ro(sizeof(oc::block));
    ro.Update(key.data(), key.size());
    oc::block b;
    ro.Final(b);
    return b;
}

oc::block payloadToBlock(const std::string& payload)
{
    std::uint64_t value = 0;
    try {
        value = std::stoull(payload);
    } catch (const std::exception&) {
        value = 0;
    }

    oc::block b = oc::ZeroBlock;
    std::memcpy(reinterpret_cast<std::uint8_t*>(&b), &value, sizeof(value));
    return b;
}

std::string blockToPayloadShare(const oc::block& b)
{
    return std::string(reinterpret_cast<const char*>(&b), kPayloadBytes);
}

std::vector<oc::block> keysOf(
    const std::vector<std::pair<std::string, std::string>>& input)
{
    std::vector<oc::block> keys;
    keys.reserve(input.size());
    for (const auto& [key, payload] : input) {
        (void)payload;
        keys.push_back(keyToBlock(key));
    }
    return keys;
}

std::vector<std::uint64_t> idsOf(
    const std::vector<std::pair<std::string, std::string>>& input)
{
    std::vector<std::uint64_t> ids;
    ids.reserve(input.size());
    for (const auto& [key, payload] : input) {
        (void)key;
        std::uint64_t value = 0;
        try {
            value = std::stoull(payload);
        } catch (const std::exception&) {
            value = 0;
        }
        ids.push_back(value);
    }
    return ids;
}

struct BinMapping
{
    std::vector<std::uint64_t> id;
    std::vector<bool> valid;
};

BinMapping invertSenderMapping(
    const std::vector<std::array<std::uint64_t, 3>>& mapping,
    const std::vector<std::uint64_t>& ids,
    std::size_t numBins)
{
    BinMapping out;
    out.id.assign(numBins, 0);
    out.valid.assign(numBins, false);

    for (std::size_t row = 0; row < mapping.size(); ++row) {
        if (row >= ids.size()) continue;
        for (std::size_t slot = 0; slot < mapping[row].size(); ++slot) {
            const std::uint64_t bin = mapping[row][slot];
            if (bin >= numBins) continue;
            out.id[bin] = ids[row];
            out.valid[bin] = true;
        }
    }
    return out;
}

BinMapping invertReceiverMapping(
    const std::vector<std::uint64_t>& mapping,
    const std::vector<std::uint64_t>& ids,
    std::size_t numBins)
{
    BinMapping out;
    out.id.assign(numBins, 0);
    out.valid.assign(numBins, false);

    for (std::size_t row = 0; row < mapping.size(); ++row) {
        const std::uint64_t bin = mapping[row];
        if (bin >= numBins) continue;
        if (row >= ids.size()) continue;
        out.id[bin] = ids[row];
        out.valid[bin] = true;
    }
    return out;
}

} // namespace

#endif

std::vector<PSIPayloadShare> ss_psi_volepsi_run(
    const std::vector<std::pair<std::string, std::string>>& sender_input,
    const std::vector<std::pair<std::string, std::string>>& receiver_input)
{
    std::vector<PSIPayloadShare> result;

#ifdef APPROX_PSI_HAS_VOLEPSI
    std::vector<oc::block> sendKeys;
    std::vector<oc::block> sendPayloads;
    sendKeys.reserve(sender_input.size());
    sendPayloads.reserve(sender_input.size());

    std::vector<std::size_t> sendIds;
    sendIds.reserve(sender_input.size());

    for (const auto& [key, payload] : sender_input) {
        sendKeys.push_back(keyToBlock(key));
        sendPayloads.push_back(payloadToBlock(payload));
        sendIds.push_back(static_cast<std::size_t>(std::stoull(payload)));
    }

    std::vector<oc::block> recvKeys;
    recvKeys.reserve(receiver_input.size());

    std::vector<std::size_t> recvIds;
    recvIds.reserve(receiver_input.size());

    for (const auto& [key, payload] : receiver_input) {
        recvKeys.push_back(keyToBlock(key));
        recvIds.push_back(static_cast<std::size_t>(std::stoull(payload)));
    }

    auto sockets = coproto::LocalAsyncSocket::makePair();

    volePSI::RsCpsiSender sender;
    volePSI::RsCpsiReceiver receiver;
    volePSI::RsCpsiSender::Sharing sShare;
    volePSI::RsCpsiReceiver::Sharing rShare;

    oc::Matrix<oc::u8> senderValues(sendPayloads.size(), kPayloadBytes);
    for (std::size_t i = 0; i < sendPayloads.size(); ++i) {
        std::memcpy(&senderValues(i, 0), &sendPayloads[i], kPayloadBytes);
    }

    const std::uint64_t senderSize = sendKeys.size();
    const std::uint64_t recverSize = recvKeys.size();

    std::thread senderThread([&]() {
        sender.init(
            senderSize, recverSize, kPayloadBytes, kStatSecParam,
            oc::sysRandomSeed(), kNumThreads,
            volePSI::ValueShareType::Xor);
        auto p = sender.send(sendKeys, senderValues, sShare, sockets[0]);
        coproto::sync_wait(p);
    });

    receiver.init(
        senderSize, recverSize, kPayloadBytes, kStatSecParam,
        oc::sysRandomSeed(), kNumThreads,
        volePSI::ValueShareType::Xor);
    auto p0 = receiver.receive(recvKeys, rShare, sockets[1]);
    coproto::sync_wait(p0);
    senderThread.join();

    const std::size_t numBins = sShare.mFlagBits.size();

    std::vector<std::uint64_t> binToSendId(numBins, 0);
    std::vector<bool> binMapped(numBins, false);
    for (std::size_t row = 0; row < sShare.mMapping.size(); ++row) {
        for (std::size_t slot = 0; slot < sShare.mMapping[row].size(); ++slot) {
            const std::uint64_t bin = sShare.mMapping[row][slot];
            if (bin >= numBins || row >= sendIds.size()) continue;
            binToSendId[bin] = static_cast<std::uint64_t>(sendIds[row]);
            binMapped[bin] = true;
        }
    }

    std::vector<std::uint64_t> binToRecvId(numBins, 0);
    std::vector<bool> recvBinMapped(numBins, false);
    for (std::size_t row = 0; row < rShare.mMapping.size(); ++row) {
        const std::uint64_t bin = rShare.mMapping[row];
        if (bin >= numBins || row >= recvIds.size()) continue;
        binToRecvId[bin] = static_cast<std::uint64_t>(recvIds[row]);
        recvBinMapped[bin] = true;
    }

    // Single-process/debug path: reconstruct hit bins because both shares live in
    // this process.  This is intentionally NOT used by run_as_sender/receiver.
    for (std::size_t k = 0; k < numBins; ++k) {
        const bool hit = sShare.mFlagBits[k] ^ rShare.mFlagBits[k];
        if (!hit) continue;

        oc::block senderShare = oc::ZeroBlock;
        oc::block receiverShare = oc::ZeroBlock;
        std::memcpy(&senderShare, &sShare.mValues(k, 0), kPayloadBytes);
        std::memcpy(&receiverShare, &rShare.mValues(k, 0), kPayloadBytes);

        PSIPayloadShare share;
        share.bin = static_cast<std::uint64_t>(k);
        share.my_id = binMapped[k] ? binToSendId[k] : 0;
        share.peer_id = recvBinMapped[k] ? binToRecvId[k] : 0;
        share.local_valid = binMapped[k];
        share.flag_share = static_cast<std::uint8_t>(sShare.mFlagBits[k] ? 1 : 0);
        share.sender_payload_share = blockToPayloadShare(senderShare);
        share.receiver_payload_share = blockToPayloadShare(receiverShare);
        result.push_back(std::move(share));
    }
#else
    (void)sender_input;
    (void)receiver_input;
#endif

    return result;
}

std::vector<PSIPayloadShare> ss_psi_volepsi_run_sender(
    const std::vector<std::pair<std::string, std::string>>& my_input,
    coproto::Socket& psi_sock,
    std::uint64_t seed)
{
    std::vector<PSIPayloadShare> result;

#ifdef APPROX_PSI_HAS_VOLEPSI
    const std::uint64_t senderSize = my_input.size();
    std::uint64_t recverSize = 0;

    auto ss = psi_sock.send(coproto::copy(senderSize));
    coproto::sync_wait(ss);
    auto rs = psi_sock.recv(recverSize);
    coproto::sync_wait(rs);

    std::vector<oc::block> sendKeys = keysOf(my_input);
    std::vector<std::uint64_t> sendIds = idsOf(my_input);

    oc::Matrix<oc::u8> senderValues(sendKeys.size(), kPayloadBytes);
    for (std::size_t i = 0; i < my_input.size(); ++i) {
        const oc::block b = payloadToBlock(my_input[i].second);
        std::memcpy(&senderValues(i, 0), &b, kPayloadBytes);
    }

    volePSI::RsCpsiSender sender;
    volePSI::RsCpsiSender::Sharing sShare;

    sender.init(
        senderSize, recverSize, kPayloadBytes, kStatSecParam,
        oc::toBlock(seed, std::uint64_t{0x51}), kNumThreads,
        volePSI::ValueShareType::Xor);

    auto p = sender.send(sendKeys, senderValues, sShare, psi_sock);
    coproto::sync_wait(p);

    const std::size_t numBins = sShare.mFlagBits.size();
    const BinMapping mapping = invertSenderMapping(sShare.mMapping, sendIds, numBins);

    // SECURITY CRITICAL: return every bin and only the local flag share.  Do NOT
    // exchange mFlagBits, hit-bin lists, peer ids, or peer vectors here.
    result.reserve(numBins);
    for (std::size_t k = 0; k < numBins; ++k) {
        PSIPayloadShare share;
        share.bin = static_cast<std::uint64_t>(k);
        share.my_id = mapping.id[k];
        share.local_valid = mapping.valid[k];
        share.flag_share = static_cast<std::uint8_t>(sShare.mFlagBits[k] ? 1 : 0);

        oc::block senderShare = oc::ZeroBlock;
        std::memcpy(&senderShare, &sShare.mValues(k, 0), kPayloadBytes);
        share.sender_payload_share = blockToPayloadShare(senderShare);
        share.receiver_payload_share.assign(kPayloadBytes, '\0');

        result.push_back(std::move(share));
    }
#else
    (void)my_input;
    (void)psi_sock;
#endif

    return result;
}

std::vector<PSIPayloadShare> ss_psi_volepsi_run_receiver(
    const std::vector<std::pair<std::string, std::string>>& my_input,
    coproto::Socket& psi_sock,
    std::uint64_t seed)
{
    std::vector<PSIPayloadShare> result;

#ifdef APPROX_PSI_HAS_VOLEPSI
    const std::uint64_t recverSize = my_input.size();
    std::uint64_t senderSize = 0;

    auto rs = psi_sock.recv(senderSize);
    coproto::sync_wait(rs);
    auto ss = psi_sock.send(coproto::copy(recverSize));
    coproto::sync_wait(ss);

    std::vector<oc::block> recvKeys = keysOf(my_input);
    std::vector<std::uint64_t> recvIds = idsOf(my_input);

    volePSI::RsCpsiReceiver receiver;
    volePSI::RsCpsiReceiver::Sharing rShare;

    receiver.init(
        senderSize, recverSize, kPayloadBytes, kStatSecParam,
        oc::toBlock(seed, std::uint64_t{0x52}), kNumThreads,
        volePSI::ValueShareType::Xor);

    auto p = receiver.receive(recvKeys, rShare, psi_sock);
    coproto::sync_wait(p);

    const std::size_t numBins = rShare.mFlagBits.size();
    const BinMapping mapping = invertReceiverMapping(rShare.mMapping, recvIds, numBins);

    result.reserve(numBins);
    for (std::size_t k = 0; k < numBins; ++k) {
        PSIPayloadShare share;
        share.bin = static_cast<std::uint64_t>(k);
        share.my_id = mapping.id[k];
        share.local_valid = mapping.valid[k];
        share.flag_share = static_cast<std::uint8_t>(rShare.mFlagBits[k] ? 1 : 0);

        oc::block receiverShare = oc::ZeroBlock;
        std::memcpy(&receiverShare, &rShare.mValues(k, 0), kPayloadBytes);
        share.sender_payload_share.assign(kPayloadBytes, '\0');
        share.receiver_payload_share = blockToPayloadShare(receiverShare);

        result.push_back(std::move(share));
    }
#else
    (void)my_input;
    (void)psi_sock;
#endif

    return result;
}

} // namespace approx_psi::secure
