"""Multi-attribute privacy-preserving matching extension.

两个层次的接口：

* ``MatchRecord`` + ``MultiAttributeConfig`` —— V1 的固定 name+DOB 原型。
* ``AttributeSchema`` + ``AttributeRecord`` —— 任意属性集合的通用接口，
  维度与布局由声明推导，协议结构不变。
* ``multi_attribute.dataset`` —— CSV → ``AttributeRecord`` 的数据层，
  把 FEBRL / 合成数据之类的真实文件接进上面那套接口。
"""

from . import kinds  # noqa: F401  (注册内置属性类型)
from .config import MultiAttributeConfig
from .dataset import (
    AttributeQuery,
    apply_labels,
    load_attribute_queries,
    load_attribute_records,
    load_dataset_pair,
    load_febrl_pair,
)
from .encoder import (
    attribute_similarities,
    encode_attribute_matrix,
    encode_record_vectors,
    plaintext_similarity,
)
from .model import AttributeRecord, MatchRecord, record_values
from .registry import (
    AttributeBlock,
    RegistryError,
    encode_attribute,
    get_kind,
    register_encoder,
    registered_kinds,
    unregister_encoder,
)
from .schema import (
    AttributeSchema,
    AttributeSpec,
    SchemaError,
    resolve_schema,
    schema_from_json_file,
)

__all__ = [
    # V1 兼容
    "MatchRecord",
    "MultiAttributeConfig",
    "encode_record_vectors",
    "plaintext_similarity",
    # 通用 schema 接口
    "AttributeRecord",
    "AttributeSchema",
    "AttributeSpec",
    "SchemaError",
    "record_values",
    "resolve_schema",
    "schema_from_json_file",
    # 编码器注册表
    "AttributeBlock",
    "RegistryError",
    "encode_attribute",
    "encode_attribute_matrix",
    "attribute_similarities",
    "get_kind",
    "register_encoder",
    "registered_kinds",
    "unregister_encoder",
    # CSV 数据层
    "AttributeQuery",
    "apply_labels",
    "load_attribute_queries",
    "load_attribute_records",
    "load_dataset_pair",
    "load_febrl_pair",
]
