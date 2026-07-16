#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Homomorphic Encryption Fuzzy Name Matching -- Educational Demo Script
=============================================================================

This script uses "Lark" as the query name to demonstrate the complete
homomorphic encryption fuzzy matching pipeline step by step.

At each step it prints:
  1. What operation was performed (plain-language explanation)
  2. What the plaintext looks like (numeric vectors / readable text)
  3. What the ciphertext looks like (unreadable encrypted blobs)
  4. The final matching result and similarity score

Core scenario:
  - Party A holds the query name "Lark", wants to check if B's database
    has a similar name, but does NOT want B to know what they searched for.
  - Party B holds a name database, wants to protect their data.

Solution: MinHash (fuzzy hashing) + CKKS homomorphic encryption.
  Similarity is computed on ENCRYPTED data; only Party A decrypts the result.

Usage:
    cd baseline_Integration
    python tests/demo_educational.py
    python tests/demo_educational.py --query "John Smith" --db-limit 100
"""

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path
from random import SystemRandom
from typing import Any

# -- Path setup --
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from config.params import (
    NUM_PERMUTATIONS_CLUSTER,
    NUM_PERMUTATIONS_MATCH,
    RANDOM_MASK_MAX,
    RANDOM_MASK_MIN,
    SHINGLE_SIZE,
    SIMILARITY_THRESHOLD,
    choose_k,
)
from preprocessing.text_cleaner import clean_name
from preprocessing.normalizer import l2_normalize
from minhash.encoder import batch_encode, _generate_shingles, _hash_shingle
from party_b.offline_prep import fit_scaler, build_cluster_matrix
from clustering.kmeans_cosine import run_cosine_kmeans
from ckks.context import create_ckks_context
from ckks.keys import encrypt, serialize_public_context
from ckks.operations import dot_ct_pt, dot_ct_ct, matmul_ct_pt, add_plain
from party_a.local_prep import encode_query_vectors
from party_a.online_querier import decrypt_sim_scores, build_selector, encrypt_selector
from evaluation.dataset_loader import load_dataset


# =============================================================================
# Pretty-print utilities
# =============================================================================

SEP_MAJOR = "=" * 76
SEP_MINOR = "-" * 76


def _banner(title: str) -> None:
    print()
    print(SEP_MAJOR)
    print(f"  {title}")
    print(SEP_MAJOR)


def _step(num: int, title: str, party: str) -> None:
    icon = "[Server]" if party == "B" else "[Client]"
    role = "Database Owner" if party == "B" else "Querier"
    print()
    print(SEP_MINOR)
    print(f"  {icon} Step {num}: {title}")
    print(f"       Executed by: Party {party} ({role})")
    print(SEP_MINOR)


def _sub(title: str) -> None:
    print(f"  > {title}")


def _plain(label: str, value: Any) -> None:
    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            vals = ", ".join(f"{v:.4f}" for v in value[:8])
            tail = ", ..." if len(value) > 8 else ""
            print(f"    [Plaintext] {label}: [{vals}{tail}]  (shape={value.shape})")
        elif value.ndim == 2:
            print(f"    [Plaintext] {label}: shape={value.shape}")
            for i, row in enumerate(value[:2]):
                vals = ", ".join(f"{v:.4f}" for v in row[:5])
                print(f"       row {i}: [{vals}, ...]")
    elif isinstance(value, list):
        if len(value) > 0 and isinstance(value[0], np.ndarray):
            print(f"    [Plaintext] {label}: {len(value)} vectors")
        else:
            vals = ", ".join(str(v)[:40] for v in value[:5])
            tail = ", ..." if len(value) > 5 else ""
            print(f"    [Plaintext] {label}: [{vals}{tail}]")
    elif isinstance(value, str):
        if len(value) > 120:
            print(f"    [Plaintext] {label}: \"{value[:120]}...\"")
        else:
            print(f"    [Plaintext] {label}: \"{value}\"")
    else:
        print(f"    [Plaintext] {label}: {value}")


def _cipher(label: str, ct: Any) -> None:
    if hasattr(ct, "serialize"):
        raw = ct.serialize()
        b64 = base64.b64encode(raw).decode("ascii")
        preview = b64[:80]
        print(f"    [Ciphertext] {label} (CKKS encrypted):")
        print(f"       Type: {type(ct).__name__}")
        print(f"       Size: {len(raw)} bytes (serialized)")
        print(f"       Fragment: {preview}...")
        print(f"       [WARNING] Without the private key, this is unreadable garbage!")
    elif isinstance(ct, bytes):
        b64 = base64.b64encode(ct).decode("ascii")
        preview = b64[:80]
        print(f"    [Ciphertext] {label} (serialized):")
        print(f"       Size: {len(ct)} bytes")
        print(f"       Fragment: {preview}...")
    elif isinstance(ct, list):
        print(f"    [Ciphertext] {label}: {len(ct)} encrypted scalars")
        if len(ct) > 0 and hasattr(ct[0], "serialize"):
            raw = ct[0].serialize()
            print(f"       Each ~{len(raw)} bytes")
            print(f"       [WARNING] All encrypted -- Party B sees only noise!")


def _info(text: str) -> None:
    for line in text.split("\n"):
        print(f"    [Info] {line}")


def _ok(msg: str) -> None:
    print(f"    [OK] {msg}")


# =============================================================================
# Main demo pipeline
# =============================================================================

def demo_pipeline(
    query_name: str = "Lark",
    db_limit: int = 50,
    tau: float = SIMILARITY_THRESHOLD,
) -> None:
    """Run the complete homomorphic encryption fuzzy matching demo."""

    # =========================================================================
    # Introduction
    # =========================================================================

    _banner("HOMOMORPHIC ENCRYPTION FUZZY NAME MATCHING -- Educational Demo")
    print()
    print(f"  Query name (Party A input):  \"{query_name}\"")
    print(f"  Database size (Party B):     {db_limit} name records")
    print(f"  Similarity threshold tau:    {tau}")
    print()
    print("  +-----------------------------------------------------------+")
    print("  |  SCENARIO:                                                |")
    print(f"  |  * Party A (Querier) wants to check if \"{query_name}\"        |")
    print("  |    exists in Party B's database.                          |")
    print("  |  * Party B (Data Owner) has many names but cannot         |")
    print("  |    share them directly with A.                            |")
    print("  |  * Party A also does NOT want B to know the query.        |")
    print("  |  * Solution: Homomorphic Encryption -- compute on         |")
    print("  |    encrypted data, only A sees the final result.          |")
    print("  +-----------------------------------------------------------+")

    # =========================================================================
    # Step 0: Party B Offline Preparation
    # =========================================================================

    _step(0, "Offline Prep: Load DB & Cluster (Party B, done once)", "B")
    _info(
        "Party B prepares its database BEFORE receiving any query:\n"
        "  1) Each name -> MinHash signature (200-number vector)\n"
        "  2) L2-normalize vectors (unit length)\n"
        "  3) StandardScaler (mean=0, std=1 per dimension)\n"
        "  4) K-Means clustering (group similar names into k clusters)\n"
        "  This is OFFLINE -- done once, reused for all queries."
    )

    # Load dataset
    names_a, names_b_all, labels = load_dataset(
        "ncvr_10k", str(_PROJECT_ROOT / "data")
    )
    names_b = names_b_all[:db_limit]
    _sub(f"Loaded {len(names_b)} names, e.g.:")
    for i, name in enumerate(names_b[:5]):
        print(f"      [{i}] \"{name}\"")
    print(f"      ... ({len(names_b)} total)")

    # -- Substep 0a: MinHash encoding --
    print()
    _sub("0a. MinHash encoding -> turn names into number vectors")
    _info(
        "MinHash is a Locality-Sensitive Hashing (LSH) technique.\n"
        "Core idea: the more similar two strings are, the more similar\n"
        "their MinHash signatures will be.\n"
        "Process: trigram shingling -> hash -> 200 random permutations -> min.\n"
        "Result: each name becomes a 200-dimensional number vector."
    )

    signatures_200 = batch_encode(names_b, NUM_PERMUTATIONS_CLUSTER)
    signatures_50 = signatures_200[:, :NUM_PERMUTATIONS_MATCH]

    # Show encoding for the first name
    demo_name = names_b[0]
    print()
    print(f"      Example: how \"{demo_name}\" is encoded:")
    cleaned = clean_name(demo_name)
    print(f"        1) Clean: \"{demo_name}\" -> \"{cleaned}\"")
    shingles = sorted(_generate_shingles(cleaned, SHINGLE_SIZE))
    print(f"        2) Shingle (trigram, size={SHINGLE_SIZE}): {shingles[:8]}{'...' if len(shingles) > 8 else ''}")
    hashes = sorted([_hash_shingle(s) for s in shingles])
    print(f"        3) Hash each shingle to number: {hashes[:5]}...")
    print(f"        4) 200 random permutations, take min -> 200-dim signature")
    _plain(f"\"{demo_name}\" MinHash sig (first 8 dims)", signatures_200[0])

    # -- Substep 0b: L2 normalize --
    print()
    _sub("0b. L2 normalization -> make all vectors unit length")
    _info(
        "After normalization, every vector has length = 1.\n"
        "This means comparing two vectors = comparing their DIRECTION\n"
        "(cosine similarity), unaffected by vector magnitude."
    )

    normalized_200 = l2_normalize(signatures_200)
    normalized_50 = l2_normalize(signatures_50)
    _plain("Normalized vector (first 8 dims)", normalized_200[0])
    print(f"      Vector length check: {np.linalg.norm(normalized_200[0]):.6f} (should be ~1.0)")

    # -- Substep 0c: StandardScaler --
    print()
    _sub("0c. StandardScaler -> center each dimension to mean=0, std=1")
    _info(
        "This makes K-Means clustering more accurate by preventing\n"
        "large-value dimensions from dominating the clustering.\n"
        "Party B shares the scaler parameters (mean, scale) with Party A\n"
        "so A can standardize their query the same way."
    )

    _, standardized_200, scaler_mean, scaler_scale = fit_scaler(normalized_200)
    _plain("scaler_mean (first 8 dims)", scaler_mean)
    _plain("scaler_scale (first 8 dims)", scaler_scale)

    # -- Substep 0d: K-Means clustering --
    print()
    _sub("0d. K-Means clustering -> group similar names together")
    k = choose_k(len(names_b), mode="sqrt")
    centroids, cluster_assignments = run_cosine_kmeans(
        standardized_200, k=k, iterations=20, random_state=42
    )
    cluster_sizes = np.bincount(cluster_assignments, minlength=k)
    print(f"      Cluster count K = {k} (sqrt({len(names_b)}) = {np.sqrt(len(names_b)):.1f})")
    print(f"      Cluster sizes: {cluster_sizes.tolist()}")
    _info(
        "K = sqrt(n) is an engineering heuristic: balances search\n"
        "efficiency and accuracy. Query only compares against K\n"
        "centroids, not all n names."
    )

    # -- Substep 0e: Build cluster matrix --
    print()
    _sub("0e. Build cluster matrix -> arrange cluster vectors into a 3D matrix")
    cluster_matrix, max_size = build_cluster_matrix(
        normalized_50, cluster_assignments, k
    )
    print(f"      Cluster matrix shape: {cluster_matrix.shape}")
    print(f"      Meaning: {k} clusters x max {max_size} names each x 50 dims")
    _info(
        "Each cluster is a 'candidate pool'. The query only searches\n"
        "within the best-matching cluster, avoiding a full database scan."
    )

    # =========================================================================
    # Step 1: Party A - Query Encoding
    # =========================================================================

    _step(1, f"Query Encoding: turn \"{query_name}\" into number vectors", "A")
    _info(
        "Party A encodes the query name using the SAME pipeline as Party B:\n"
        "  Clean -> MinHash -> L2-normalize -> StandardScaler"
    )

    print()
    _sub(f"1a. Clean -> normalize the query format")
    cleaned_query = clean_name(query_name)
    print(f"      Original input: \"{query_name}\"")
    print(f"      After cleaning: \"{cleaned_query}\"")
    _info(
        "Cleaning rules: lowercase, remove non-alpha chars, collapse spaces.\n"
        "This ensures \"Lark\", \"lark\", \" LARK \" all map to the same \"lark\"."
    )

    print()
    _sub("1b. MinHash encoding -> generate digital fingerprint")
    query_shingles = sorted(_generate_shingles(cleaned_query, SHINGLE_SIZE))
    print(f"      Trigram shingles: {query_shingles}")
    _info(
        f"\"{cleaned_query}\" has {len(cleaned_query)} chars -> produces {len(query_shingles)} trigrams.\n"
        "Each trigram is hashed, then 200 permutations take the minimum hash value."
    )

    query_200_std, query_50_norm = encode_query_vectors(
        query_name, scaler_mean, scaler_scale
    )
    print()
    _plain(f"MinHash signature EL=200 (query, first 8 dims)", query_200_std)
    _plain(f"MinHash signature EL=50 (matching, first 8 dims)", query_50_norm)
    _info(
        "EL=200: for K-Means cluster selection (coarse screening).\n"
        "EL=50:  for precise column-wise matching (fine screening).\n"
        "Higher dimension = more discriminative but slower. Two-stage = best of both."
    )

    # =========================================================================
    # Step 2: Party A - CKKS Homomorphic Encryption
    # =========================================================================

    _step(2, "CKKS Encryption: lock the vectors into a 'crypto safe'", "A")
    _info(
        "CKKS is a Homomorphic Encryption scheme. Its magic:\n"
        "  1) Data is encrypted -> completely unreadable to outsiders\n"
        "  2) BUT you can do addition & multiplication on encrypted data!\n"
        "  3) Decrypting the result = same as computing on the original data\n"
        "\n"
        "Analogy: You lock numbers in a safe. Others can do math on the\n"
        "safe's exterior (which mechanically transforms the contents),\n"
        "and only YOU can open it to see the final answer."
    )

    # Create CKKS context
    print()
    _sub("2a. Generate keys -> create CKKS encryption context")
    t0 = time.perf_counter()
    secret_context = create_ckks_context()
    secret_context.generate_relin_keys()
    print(f"      Polynomial modulus (N):  8192")
    print(f"      Scale factor:            2^40 ~= {2**40:,}")
    print(f"      [Timing] Key generation: {(time.perf_counter() - t0)*1000:.0f} ms")
    _info(
        "Generated a KEY PAIR: public key (for encryption & computation)\n"
        "+ private key (for decryption).\n"
        "The public key is shared with Party B (so B can compute on ciphertext).\n"
        "The private key stays ONLY with Party A (so only A can decrypt results)."
    )

    # Serialize public context (NO private key!)
    public_context_bytes = serialize_public_context(secret_context)
    print()
    _sub("2b. Export public context (WITHOUT private key) -> send to Party B")
    print(f"      Public context size: {len(public_context_bytes):,} bytes")
    _info(
        "This public context contains the public key and evaluation keys,\n"
        "but NOT the private key! Party B can compute on ciphertexts but\n"
        "CANNOT decrypt them."
    )

    # Encrypt query vectors
    print()
    _sub("2c. Encrypt query vectors -> plaintext becomes ciphertext")

    encrypted_query_200 = encrypt(query_200_std, secret_context)
    encrypted_query_50 = encrypt(query_50_norm, secret_context)

    _plain("query_200 BEFORE encryption (plaintext, first 8 dims)", query_200_std)
    _cipher("query_200 AFTER encryption (ciphertext)", encrypted_query_200)
    print()
    _plain("query_50 BEFORE encryption (plaintext, first 8 dims)", query_50_norm)
    _cipher("query_50 AFTER encryption (ciphertext)", encrypted_query_50)

    print()
    _info(
        "See! ~200 bytes of plaintext numbers became SEVERAL KB of ciphertext noise.\n"
        "The ciphertext contains 'noise' -- this is CKKS's security foundation.\n"
        "Party A sends ONLY encrypted_query_200 to B now;\n"
        "encrypted_query_50 stays local, sent only in Round 2."
    )

    # =========================================================================
    # Step 3: Party B - Compare to Centroids (on ciphertext!)
    # =========================================================================

    _step(3, "Encrypted Compute 1: query vs cluster centroids (Party B)", "B")
    _info(
        "This is the MAGIC MOMENT of homomorphic encryption!\n"
        "Party B received ONLY ciphertext (unreadable garbage).\n"
        "But using the public key, B can compute:\n"
        "  encrypted_score_i = dot(encrypted_query_200, centroid_i)\n"
        "The result is STILL ENCRYPTED -- B has NO IDEA what the similarity is!"
    )

    print()
    _sub("3a. Compute encrypted dot product with each centroid")
    encrypted_sim_scores = [
        dot_ct_pt(encrypted_query_200, centroid) for centroid in centroids
    ]

    print(f"      Computed {len(encrypted_sim_scores)} encrypted-plaintext dot products")
    print(f"      (one per cluster centroid)")
    _cipher("Encrypted similarity scores", encrypted_sim_scores)
    _info(
        f"ALL {k} scores are encrypted! Party B only knows 'I computed {k} results'\n"
        f"but has NO IDEA which cluster is the best match.\n"
        f"B sends these back to A, who decrypts them with the private key."
    )

    # =========================================================================
    # Step 4: Party A - Decrypt & Select Best Cluster
    # =========================================================================

    _step(4, "Decrypt & Select Best Cluster (Party A)", "A")
    _info(
        f"Party A uses the private key to decrypt the {k} similarity scores,\n"
        f"picks the cluster with the HIGHEST score (argmax),\n"
        f"then tells Party B which cluster to search -- but the selection\n"
        f"is ALSO ENCRYPTED (one-hot vector) so B still doesn't know!"
    )

    # Decrypt similarity scores
    sim_scores = decrypt_sim_scores(encrypted_sim_scores, secret_context)

    print()
    _sub("4a. Decrypt similarity scores")
    for i, score in enumerate(sim_scores):
        bar = "#" * max(1, int(abs(score) * 100))
        print(f"      Cluster {i:2d}: {score:+.6f}  {bar}")
    _info(
        "These are cosine similarities (computed ENTIRELY on encrypted data!).\n"
        "Positive = similar direction (likely match).\n"
        "Negative = opposite direction (not similar).\n"
        "Party A sees all scores; Party B sees NONE of them."
    )

    # Select best cluster
    print()
    _sub("4b. Select best cluster & build encrypted one-hot selector")
    selected_cluster, selector = build_selector(sim_scores, k)
    encrypted_selector = encrypt_selector(selector, secret_context)

    print(f"      Best cluster: #{selected_cluster} (score: {sim_scores[selected_cluster]:.6f})")
    print(f"      Cluster size: ~{cluster_sizes[selected_cluster]} names")
    _plain("One-hot selector (plaintext)", selector)
    _cipher("One-hot selector (encrypted)", encrypted_selector)
    _info(
        f"Position {selected_cluster} of the one-hot is 1.0, all others are 0.0.\n"
        f"But after encryption, Party B sees ONLY garbage!\n"
        f"B will use this encrypted one-hot to 'select' the right cluster,\n"
        f"but B will NEVER know WHICH cluster was selected."
    )

    # =========================================================================
    # Step 5: Party B - Column-wise Matching (on ciphertext!)
    # =========================================================================

    _step(5, "Encrypted Compute 2: Column-wise Matching (Party B)", "B")
    _info(
        "Party B now has:\n"
        "  1) Encrypted one-hot selector (doesn't know which cluster is selected)\n"
        "  2) Encrypted query_50 vector (doesn't know the query content)\n"
        "\n"
        "For each column in the cluster matrix, B performs THREE encrypted ops:\n"
        "  [1] Selector x ClusterMatrix -> encrypted selected name vector\n"
        "  [2] encrypted_query . encrypted_name -> encrypted similarity\n"
        "  [3] Add random mask -mask*tau -> hides raw similarity value\n"
        "\n"
        "ALL THREE operations run on encrypted data. B sees NOTHING in plaintext!"
    )

    print()
    _sub("5a. Column-wise matching (encrypted operations per column)")
    print(f"      Cluster matrix shape: {cluster_matrix.shape}")
    print(f"      Columns to scan: {max_size}")

    _rng = SystemRandom()
    encrypted_results = []
    for col_idx in range(max_size):
        column_j = cluster_matrix[:, col_idx, :]
        mask = _rng.uniform(RANDOM_MASK_MIN, RANDOM_MASK_MAX)

        # Three-step encrypted computation
        masked_column_j = mask * column_j
        encrypted_selected_name = matmul_ct_pt(encrypted_selector, masked_column_j)
        encrypted_cos_score = dot_ct_ct(encrypted_query_50, encrypted_selected_name)
        encrypted_final = add_plain(encrypted_cos_score, -mask * tau)
        encrypted_results.append(encrypted_final)

    print(f"      Done: {len(encrypted_results)} columns processed on ciphertext")
    _cipher("Encrypted column match results", encrypted_results)
    _info(
        "Each column's result = mask * (real_cosine_similarity - tau)\n"
        "The random mask 'mask' (positive) hides the exact similarity from A,\n"
        "but preserves the SIGN: positive -> similarity > threshold (MATCH!).\n"
        "B cannot see which columns are positive (all encrypted)."
    )

    # =========================================================================
    # Step 6: Party A - Decrypt & Final Judgment
    # =========================================================================

    _step(6, "Decrypt & Final Judgment: Find Matches! (Party A)", "A")
    _info(
        "Final step! Party A decrypts each column score with the private key.\n"
        "If a column's score > 0, it means cosine_similarity > tau -> MATCH!\n"
        "With 'early stop' enabled, A stops at the first match to save time."
    )

    print()
    _sub("6a. Decrypt column scores one by one")

    all_scores = []
    match_found = False
    first_positive_col = None

    for col_idx, enc_score in enumerate(encrypted_results):
        values = (
            enc_score.decrypt()
            if hasattr(enc_score, "decrypt")
            else secret_context.decrypt(enc_score)
        )
        plain_score = float(np.asarray(values, dtype=np.float64).reshape(-1)[0])
        all_scores.append(plain_score)

        if plain_score > 1e-6 and not match_found:
            match_found = True
            first_positive_col = col_idx

    # Print all column scores
    print(f"      Threshold tau = {tau}")
    print(f"      {'Col':<6} {'Decrypted Score':>16} {'Verdict':>12}")
    print(f"      {'-'*36}")

    for i, score in enumerate(all_scores):
        status = "[MATCH!]" if score > 1e-6 else "[no match]"
        marker = " <-- HIT!" if i == first_positive_col else ""
        print(f"      {i:<6} {score:>+14.6f}   {status}{marker}")

    print()
    _info(
        "Score meaning: mask * (cosine_similarity - tau)\n"
        "Since mask is always positive, score > 0 if and only if\n"
        "cosine_similarity > tau (the matching threshold).\n"
        "CKKS is approximate, so we use epsilon=1e-6 instead of exact 0."
    )

    # =========================================================================
    # Final Results Summary
    # =========================================================================

    _banner("[SUMMARY] FINAL RESULTS")

    # Find the matched name
    if first_positive_col is not None:
        members = np.where(cluster_assignments == selected_cluster)[0]
        if first_positive_col < len(members):
            db_idx = int(members[first_positive_col])
            matched_name = names_b[db_idx]
        else:
            matched_name = "(index out of range)"
    else:
        matched_name = None

    print()
    print("  +-----------------------------------------------------------+")
    print("  |                                                           |")
    print(f"  |   Query name:      {query_name:<35s}            |")
    print(f"  |   Match result:    {'[MATCH FOUND!]' if match_found else '[no match found]':<30s}                  |")
    if matched_name:
        print(f"  |   Matched to:      {matched_name:<35s}            |")
        best_score = all_scores[first_positive_col] if first_positive_col is not None else float('nan')
        print(f"  |   Column score:    {best_score:+.6f}                               |")
    print(f"  |   Threshold tau:   {tau}                                        |")
    print(f"  |   Columns scanned: {len(all_scores)} (within selected cluster)              |")
    print(f"  |   Selected cluster: #{selected_cluster} (size ~{cluster_sizes[selected_cluster]})                      |")
    print("  |                                                           |")
    print("  +-----------------------------------------------------------+")

    # =========================================================================
    # Privacy Protection Summary
    # =========================================================================

    _banner("[SECURITY] PRIVACY PROTECTION SUMMARY")
    print()
    print("  Security guarantees of this protocol:")
    print()
    print("  [OK] Party B does NOT learn Party A's query name")
    print("     * B receives encrypted query vectors and selector (just noise)")
    print("     * B operates ONLY on ciphertext, never sees any plaintext")
    print("     * B cannot tell which cluster A selected (one-hot is encrypted)")
    print()
    print("  [OK] Party A does NOT learn Party B's full database")
    print("     * A only decrypts similarity scores and the match result")
    print("     * A only sees names in ONE cluster, not all names")
    print("     * A cannot extract raw cosine similarity (masked by B)")
    print()
    print("  [OK] Cosine similarity computed entirely on encrypted data")
    print("     * CKKS supports addition + multiplication on ciphertext")
    print("     * decrypt(compute(encrypt(x))) = compute(x)")
    print()
    print("  [OK] Random mask protects exact similarity values")
    print("     * Each column gets a unique random mask (1.0 ~ 10.0)")
    print("     * A only learns: is similarity > threshold? (yes/no)")
    print("     * Prevents A from inferring database distribution from scores")
    print()
    print(f"  [Key] Tech stack: MinHash (LSH) + CKKS Fully Homomorphic Encryption + K-Means")
    print()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Homomorphic Encryption Fuzzy Matching -- Educational Demo"
    )
    parser.add_argument(
        "--query", type=str, default="Lark",
        help="Query name (default: Lark)"
    )
    parser.add_argument(
        "--db-limit", type=int, default=50,
        help="Database size (default: 50)"
    )
    parser.add_argument(
        "--tau", type=float, default=SIMILARITY_THRESHOLD,
        help=f"Similarity threshold (default: {SIMILARITY_THRESHOLD})"
    )
    args = parser.parse_args()

    demo_pipeline(
        query_name=args.query,
        db_limit=args.db_limit,
        tau=args.tau,
    )
