from __future__ import annotations

from dataclasses import dataclass

from config.params import (
    DEFAULT_ATTRIBUTE_HASH_SEED,
    DEFAULT_EXACT_BUCKETS_PER_BLOCK,
    DEFAULT_EXACT_HASH_BLOCKS,
    MULTI_ATTRIBUTE_THRESHOLD,
    NAME_ATTRIBUTE,
    NUM_PERMUTATIONS_CLUSTER,
    NUM_PERMUTATIONS_MATCH,
)


@dataclass(frozen=True)
class MultiAttributeConfig:
    """Configuration for weighted name + DOB matching.

    The final encrypted score is approximately::

        name_weight * sim_name + dob_weight * sim_dob

    because each per-attribute vector is L2-normalized first and each block is
    multiplied by sqrt(weight) before concatenation.

    这是 V1 的固定两属性配置。新的多属性代码应直接使用
    :class:`multi_attribute.schema.AttributeSchema`；本类保留是为了让已有的
    demo 与测试继续按 ``cfg=MultiAttributeConfig(...)`` 调用而不必改动。
    """

    name_weight: float = 0.70
    dob_weight: float = 0.30
    similarity_threshold: float = MULTI_ATTRIBUTE_THRESHOLD

    # Keep the repository's original MinHash lengths for the name block.
    name_cluster_dim: int = NUM_PERMUTATIONS_CLUSTER
    name_match_dim: int = NUM_PERMUTATIONS_MATCH

    # DOB uses 2 independent one-hot hash blocks of 128 buckets each.
    # Same date => dot product 1. Different dates are normally 0; a one-block
    # collision contributes 0.5 and a full collision probability is 1/16384.
    dob_hash_blocks: int = DEFAULT_EXACT_HASH_BLOCKS
    dob_buckets_per_block: int = DEFAULT_EXACT_BUCKETS_PER_BLOCK
    dob_hash_seed: int = DEFAULT_ATTRIBUTE_HASH_SEED

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

    def to_schema(self):
        """把本配置表示成通用的 ``AttributeSchema``。

        编码结果与 V1 的实现逐位相同，因此这是新接口的兼容入口。
        """

        # 局部 import 避免 config <-> schema 的模块级循环。
        from .schema import AttributeSchema, AttributeSpec

        return AttributeSchema(
            attributes=(
                AttributeSpec(
                    name=NAME_ATTRIBUTE,
                    kind="fuzzy_text",
                    weight=self.name_weight,
                    params={
                        "cluster_dim": self.name_cluster_dim,
                        "match_dim": self.name_match_dim,
                    },
                ),
                AttributeSpec(
                    name="dob",
                    kind="date",
                    weight=self.dob_weight,
                    params={
                        "blocks": self.dob_hash_blocks,
                        "buckets_per_block": self.dob_buckets_per_block,
                        "seed": self.dob_hash_seed,
                    },
                ),
            ),
            similarity_threshold=self.similarity_threshold,
            name_attribute=NAME_ATTRIBUTE,
            standardize_cluster=True,
        )
