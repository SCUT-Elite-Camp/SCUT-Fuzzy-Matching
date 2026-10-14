#include "secure_approx_psi.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>
#include "projection.h"
#include "volePSI/RsCpsi.h"
#include "cryptoTools/Crypto/PRNG.h"
#include "cryptoTools/Common/BitVector.h"

namespace {

oc::block keyToBlock(const std::string& key) {
    oc::RandomOracle ro(16);
    ro.Update(key.data(), key.size());
    oc::block b;
    ro.Final(b);
    return b;
}

oc::block bitsToBlock(const BitVector& bits) {
    oc::block b = oc::ZeroBlock;
    auto* p = reinterpret_cast<std::uint8_t*>(&b);
    for (std::size_t i = 0; i < bits.size() && i < 128; ++i)
        if (bits[i]) p[i / 8] |= (1u << (i % 8));
    return b;
}

BitVector blockToBits(const oc::block& b, std::size_t len) {
    BitVector out(len);
    const auto* p = reinterpret_cast<const std::uint8_t*>(&b);
    for (std::size_t i = 0; i < len && i < 128; ++i)
        out[i] = (p[i / 8] >> (i % 8)) & 1u;
    return out;
}

std::size_t hammingDist(const BitVector& a, const BitVector& b) {
    std::size_t d = 0;
    for (std::size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) ++d;
    return d;
}

} // namespace

SecureApproxPSI::SecureApproxPSI(Parameters params)
    : params_(std::move(params)) {}

std::vector<PairMatch> SecureApproxPSI::run_alice(
    const std::vector<Element>& A,
    coproto::Socket& sock)
{
    std::vector<PairMatch> matches;
    std::mt19937_64 rng(12345);
    std::size_t totalComm = 0;
    std::size_t numBinsTotal = 0;

    for (std::size_t round = 0; round < params_.k; ++round) {
        auto projection = sampleProjection(params_.length, params_.p, rng);
        auto Aproj = projectSetWithRandomRepresentative(A, projection, rng);

        std::vector<oc::block> sendSet;
        std::vector<std::size_t> sendIds;
        std::vector<oc::block> sendPayloads;
        for (const auto& [key, rep] : Aproj) {
            sendSet.push_back(keyToBlock(key));
            sendIds.push_back(rep.original.id);
            sendPayloads.push_back(bitsToBlock(rep.original.bits));
        }

        volePSI::RsCpsiSender sender;
        auto byteLength = sizeof(oc::block);
        oc::Matrix<oc::u8> senderValues(sendSet.size(), byteLength);
        for (std::size_t i = 0; i < sendPayloads.size(); ++i)
            std::memcpy(&senderValues(i, 0), &sendPayloads[i], byteLength);

        oc::PRNG prng(oc::sysRandomSeed());
        sender.init(sendSet.size(), params_.n, byteLength, 40, prng.get(), 1,
                    volePSI::ValueShareType::Xor, params_.d);

        volePSI::RsCpsiSender::Sharing sShare;
        auto p1 = sender.send(sendSet, senderValues, sShare, sock);
        coproto::sync_wait(p1);

        // 交换 mFlagBits（先 Alice 发，Bob 收；再 Bob 发，Alice 收）
        auto flagBytes = sShare.mFlagBits.sizeBytes();
        std::vector<oc::u8> myFlagBuf(flagBytes);
        std::memcpy(myFlagBuf.data(), sShare.mFlagBits.data(), flagBytes);
        auto s1 = sock.send(coproto::copy(myFlagBuf));
        coproto::sync_wait(s1);

        std::vector<oc::u8> peerFlagBuf(flagBytes);
        auto r1 = sock.recv(peerFlagBuf);
        coproto::sync_wait(r1);

        oc::BitVector peerFlag;
        peerFlag.resize(flagBytes * 8);
        std::memcpy(peerFlag.data(), peerFlagBuf.data(), flagBytes);

        {
            std::size_t cnt = 0;
            for (std::size_t k = 0; k < sShare.mFlagBits.size(); ++k)
                if (sShare.mFlagBits[k] ^ peerFlag[k]) ++cnt;
            std::cerr << "[alice round " << round << "] hit bins = " << cnt
                      << " numBins=" << sShare.mFlagBits.size()
                      << " mMapping.size=" << sShare.mMapping.size() << std::endl;
        }

        // 对每个 bin，若命中，用 sShare.mMapping 反查是哪个 send element
        // sShare.mMapping[j] = {k0, k1, k2}，第 j 个 send 元素可能映射到 3 个 bin
        // 命中的 bin k 对应的 send 元素 j 就是我们要的
        // 但 sender 的元素和 receiver 的元素要按 bin 配对
        // 所以 Alice 侧发 (bin, a_id)，Bob 侧发 (bin, b_id)
        std::size_t numBins = sShare.mFlagBits.size();
        struct Rec { std::uint64_t bin; std::uint64_t id; oc::block payload; };
        std::vector<Rec> myRecs;
        for (std::size_t k = 0; k < numBins; ++k) {
            bool hit = sShare.mFlagBits[k] ^ peerFlag[k];
            if (!hit) continue;
            Rec r;
            r.bin = k;
            r.id = 0;
            std::memcpy(&r.payload, &sShare.mValues(k, 0), sizeof(oc::block));
            myRecs.push_back(r);
        }
        std::size_t maxRecs = numBins * 3;
        std::vector<oc::u8> myHitBufFull(maxRecs * sizeof(Rec), 0);
        std::size_t myRecCount = 0;
        for (std::size_t i = 0; i < myRecs.size() && i < maxRecs; ++i) {
            std::memcpy(myHitBufFull.data() + i * sizeof(Rec), &myRecs[i], sizeof(Rec));
            ++myRecCount;
        }

        totalComm += 8 + myRecCount * sizeof(Rec);
        numBinsTotal += numBins;

        // 变长收发：先发本方实际条数，再发对应字节数
        std::uint64_t myCount64 = myRecCount;
        std::array<oc::u8, 8> myCountBuf{};
        std::memcpy(myCountBuf.data(), &myCount64, 8);
        auto s2 = sock.send(coproto::copy(myCountBuf));
        coproto::sync_wait(s2);
        if (myRecCount > 0) {
            std::vector<oc::u8> myHitBuf(myHitBufFull.data(),
                                         myHitBufFull.data() + myRecCount * sizeof(Rec));
            auto s3 = sock.send(coproto::copy(myHitBuf));
            coproto::sync_wait(s3);
        }

        std::array<oc::u8, 8> peerCountBuf{};
        auto r2 = sock.recv(peerCountBuf);
        coproto::sync_wait(r2);
        std::uint64_t peerCount64 = 0;
        std::memcpy(&peerCount64, peerCountBuf.data(), 8);
        std::size_t peerRecCount = static_cast<std::size_t>(peerCount64);
        if (peerRecCount > maxRecs) peerRecCount = maxRecs;

        std::vector<oc::u8> peerHitBuf(peerRecCount * sizeof(Rec));
        if (peerRecCount > 0) {
            auto r3 = sock.recv(peerHitBuf);
            coproto::sync_wait(r3);
        }
        totalComm += 8 + peerRecCount * sizeof(Rec);

        std::vector<Rec> peerRecs;
        for (std::size_t i = 0; i < peerRecCount; ++i) {
            Rec rec;
            std::memcpy(&rec, peerHitBuf.data() + i * sizeof(Rec), sizeof(Rec));
            peerRecs.push_back(rec);
        }

        for (auto& aRec : myRecs) {
            for (auto& bRec : peerRecs) {
                if (aRec.bin != bRec.bin) continue;
                oc::block xorBlock = aRec.payload ^ bRec.payload;
                auto elemBits = blockToBits(xorBlock, params_.length);

                for (const auto& e : A) {
                    std::size_t dd = hammingDist(e.bits, elemBits);
                    if (dd <= params_.d) {
                        matches.push_back(PairMatch{e.id, 0, dd, round});
                        break;
                    }
                }
            }
        }
    }

    std::sort(matches.begin(), matches.end(),
        [](const PairMatch& x, const PairMatch& y) {
            if (x.a_id != y.a_id) return x.a_id < y.a_id;
            return x.b_id < y.b_id;
        });
    matches.erase(std::unique(matches.begin(), matches.end(),
        [](const PairMatch& x, const PairMatch& y) {
            return x.a_id == y.a_id && x.b_id == y.b_id;
        }), matches.end());

    // 理论通信量
    std::size_t kBits = 40 + 8;  // 近似
    std::size_t perBinAnds = kBits + 2;  // isZeroCircuit: OR 归约 + 收尾 NOT
    std::size_t gcBytes = numBinsTotal * perBinAnds * 32;
    std::size_t opprfBytes = numBinsTotal * 3 * (kBits/8 + sizeof(oc::block)) * 2;
    std::cerr << "[theory-alice] GC = " << gcBytes << " bytes" << std::endl;
    std::cerr << "[theory-alice] OPPRF ≈ " << opprfBytes << " bytes" << std::endl;
    std::cerr << "[theory-alice] hit-exchange = " << totalComm << " bytes" << std::endl;
    std::cerr << "[theory-alice] total ≈ " << (gcBytes + opprfBytes + totalComm) << " bytes" << std::endl;
    return matches;
}

std::vector<PairMatch> SecureApproxPSI::run_bob(
    const std::vector<Element>& Bset,
    coproto::Socket& sock)
{
    std::vector<PairMatch> matches;
    std::mt19937_64 rng(12345);
    std::size_t totalComm = 0;
    std::size_t numBinsTotal = 0;

    for (std::size_t round = 0; round < params_.k; ++round) {
        auto projection = sampleProjection(params_.length, params_.p, rng);
        auto Bproj = projectSetWithRandomRepresentative(Bset, projection, rng);

        std::vector<oc::block> recvSet;
        std::vector<std::size_t> recvIds;
        for (const auto& [key, rep] : Bproj) {
            recvSet.push_back(keyToBlock(key));
            recvIds.push_back(rep.original.id);
        }

        volePSI::RsCpsiReceiver recver;
        auto byteLength = sizeof(oc::block);
        oc::PRNG prng(oc::sysRandomSeed());
        recver.init(recvSet.size(), recvSet.size(), byteLength, 40, prng.get(), 1,
                    volePSI::ValueShareType::Xor, params_.d);

        volePSI::RsCpsiReceiver::Sharing rShare;
        auto p0 = recver.receive(recvSet, rShare, sock);
        coproto::sync_wait(p0);

        // 交换 mFlagBits（Bob 先收 Alice 的，再发自己的）
        auto flagBytes = rShare.mFlagBits.sizeBytes();
        std::vector<oc::u8> peerFlagBuf(flagBytes);
        auto r1 = sock.recv(peerFlagBuf);
        coproto::sync_wait(r1);

        std::vector<oc::u8> myFlagBuf(flagBytes);
        std::memcpy(myFlagBuf.data(), rShare.mFlagBits.data(), flagBytes);
        auto s1 = sock.send(coproto::copy(myFlagBuf));
        coproto::sync_wait(s1);

        oc::BitVector peerFlag;
        peerFlag.resize(flagBytes * 8);
        std::memcpy(peerFlag.data(), peerFlagBuf.data(), flagBytes);

        {
            std::size_t cnt = 0;
            for (std::size_t k = 0; k < rShare.mFlagBits.size(); ++k)
                if (rShare.mFlagBits[k] ^ peerFlag[k]) ++cnt;
            std::cerr << "[bob round " << round << "] hit bins = " << cnt
                      << " numBins=" << rShare.mFlagBits.size()
                      << " mMapping.size=" << rShare.mMapping.size() << std::endl;
        }

        std::size_t numBins = rShare.mFlagBits.size();
        struct Rec { std::uint64_t bin; std::uint64_t id; oc::block payload; };
        std::vector<Rec> myRecs;
        for (std::size_t k = 0; k < numBins; ++k) {
            bool hit = rShare.mFlagBits[k] ^ peerFlag[k];
            if (!hit) continue;
            Rec r;
            r.bin = k;
            r.id = 0;
            std::memcpy(&r.payload, &rShare.mValues(k, 0), sizeof(oc::block));
            myRecs.push_back(r);
        }
        std::size_t maxRecs = numBins * 3;
        std::vector<oc::u8> myHitBufFull(maxRecs * sizeof(Rec), 0);
        std::size_t myRecCount = 0;
        for (std::size_t i = 0; i < myRecs.size() && i < maxRecs; ++i) {
            std::memcpy(myHitBufFull.data() + i * sizeof(Rec), &myRecs[i], sizeof(Rec));
            ++myRecCount;
        }

        numBinsTotal += numBins;

        // 变长收发：Bob 先收后发（与 Alice 相反）
        std::array<oc::u8, 8> peerCountBuf{};
        auto r2 = sock.recv(peerCountBuf);
        coproto::sync_wait(r2);
        std::uint64_t peerCount64 = 0;
        std::memcpy(&peerCount64, peerCountBuf.data(), 8);
        std::size_t peerRecCount = static_cast<std::size_t>(peerCount64);
        if (peerRecCount > maxRecs) peerRecCount = maxRecs;

        std::vector<oc::u8> peerHitBuf(peerRecCount * sizeof(Rec));
        if (peerRecCount > 0) {
            auto r3 = sock.recv(peerHitBuf);
            coproto::sync_wait(r3);
        }
        totalComm += 8 + peerRecCount * sizeof(Rec);

        std::uint64_t myCount64 = myRecCount;
        std::array<oc::u8, 8> myCountBuf{};
        std::memcpy(myCountBuf.data(), &myCount64, 8);
        auto s2 = sock.send(coproto::copy(myCountBuf));
        coproto::sync_wait(s2);
        if (myRecCount > 0) {
            std::vector<oc::u8> myHitBuf(myHitBufFull.data(),
                                         myHitBufFull.data() + myRecCount * sizeof(Rec));
            auto s3 = sock.send(coproto::copy(myHitBuf));
            coproto::sync_wait(s3);
        }
        totalComm += 8 + myRecCount * sizeof(Rec);

        std::vector<Rec> peerRecs;
        for (std::size_t i = 0; i < peerRecCount; ++i) {
            Rec rec;
            std::memcpy(&rec, peerHitBuf.data() + i * sizeof(Rec), sizeof(Rec));
            peerRecs.push_back(rec);
        }

        for (auto& bRec : myRecs) {
            for (auto& aRec : peerRecs) {
                if (bRec.bin != aRec.bin) continue;
                oc::block xorBlock = aRec.payload ^ bRec.payload;
                auto elemBits = blockToBits(xorBlock, params_.length);

                for (const auto& e : Bset) {
                    std::size_t dd = hammingDist(e.bits, elemBits);
                    if (dd <= params_.d) {
                        matches.push_back(PairMatch{0, e.id, dd, round});
                        break;
                    }
                }
            }
        }
    }

    std::sort(matches.begin(), matches.end(),
        [](const PairMatch& x, const PairMatch& y) {
            if (x.a_id != y.a_id) return x.a_id < y.a_id;
            return x.b_id < y.b_id;
        });
    matches.erase(std::unique(matches.begin(), matches.end(),
        [](const PairMatch& x, const PairMatch& y) {
            return x.a_id == y.a_id && x.b_id == y.b_id;
        }), matches.end());

    // 理论通信量
    std::size_t kBits = 40 + 8;
    std::size_t perBinAnds = kBits + 2;  // isZeroCircuit: OR 归约 + 收尾 NOT
    std::size_t gcBytes = numBinsTotal * perBinAnds * 32;
    std::size_t opprfBytes = numBinsTotal * 3 * (kBits/8 + sizeof(oc::block)) * 2;
    std::cerr << "[theory-bob] GC = " << gcBytes << " bytes" << std::endl;
    std::cerr << "[theory-bob] OPPRF ≈ " << opprfBytes << " bytes" << std::endl;
    std::cerr << "[theory-bob] hit-exchange = " << totalComm << " bytes" << std::endl;
    std::cerr << "[theory-bob] total ≈ " << (gcBytes + opprfBytes + totalComm) << " bytes" << std::endl;
    return matches;
}
