# specs/ — 需求规约

本目录是项目的单一权威需求库，按领域归档，遵循统一编写规范。

| 文件 | 内容 |
| :--- | :--- |
| `REQUIREMENT_INDEX.json` | 全库需求索引与标识符注册表 |
| `GLOSSARY_GLOBAL.md` | 全局术语表（唯一定义处，其他文档只引用） |
| `matching/matching_fuzzy_match.req.md` | 主 PRD：两方协议、多属性、多语言、批处理 |
| `matching/evidence/` | 验收证据快照（实测 artifact 归档） |

编写约定：需求编号形如 `REQ-<域>.<特性>-<序号>`；状态三态（明确 / 待确认 / 废弃）；不确定内容一律进入待确认清单，不脑补。
