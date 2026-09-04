from __future__ import annotations

from dataclasses import dataclass

from config.params import NUM_PERMUTATIONS_CLUSTER, NUM_PERMUTATIONS_MATCH


@dataclass(frozen=True)
class MultiAttributeConfig:
    """Configuration for weighted name + DOB matching.

    The final encrypted score is approximately::

        name_weight * sim_name + dob_weight * sim_dob

    because each per-attribute vector is L2-normalized first and each block is
    multiplied by sqrt(weight) before concatenation.
    """

    name_weight: float = 0.70
    dob_weight: float = 0.30
    similarity_threshold: float = 0.80

    # Keep the repository's original MinHash lengths for the name block.
    name_cluster_dim: int = NUM_PERMUTATIONS_CLUSTER
    name_match_dim: int = NUM_PERMUTATIONS_MATCH

    # DOB uses 2 independent one-hot hash blocks of 128 buckets each.
    # Same date => dot product 1. Different dates are normally 0; a one-block
    # collision contributes 0.5 and a full collision probability is 1/16384.
    dob_hash_blocks: int = 2
    dob_buckets_per_block: int = 128
    dob_hash_seed: int = 20260904

    def __post_init__(self) -> None:
        if self.name_weight < 0 or self.dob_weight < 0:
            raise ValueError("attribute weights must be non-negative")
        total = self.name_weight + self.dob_weight
        if abs(total - 1.0) > 1e-12:
            raise ValueError(
                "name_weight + dob_weight must equal 1.0, "
                f"got {total}"
            )
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be in [0, 1]")
        if self.name_cluster_dim <= 0 or self.name_match_dim <= 0:
            raise ValueError("name dimensions must be positive")
        if self.name_match_dim > self.name_cluster_dim:
            raise ValueError("name_match_dim cannot exceed name_cluster_dim")
        if self.dob_hash_blocks <= 0 or self.dob_buckets_per_block <= 1:
            raise ValueError("invalid DOB hash dimensions")

    @property
    def dob_dim(self) -> int:
        return self.dob_hash_blocks * self.dob_buckets_per_block

    @property
    def cluster_dim(self) -> int:
        return self.name_cluster_dim + self.dob_dim

    @property
    def match_dim(self) -> int:
        return self.name_match_dim + self.dob_dim
