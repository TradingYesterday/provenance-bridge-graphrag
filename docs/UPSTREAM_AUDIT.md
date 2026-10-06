# 上游审计

固定提交：`myz12138/CS-RAG@c84a081c728dd45c284a7165031c853eef4e8f24`。

本仓库不复制其源码。本机 `git clone` 在 SSL 超时后没有得到工作副本。下面是对该提交已公开文件的只读核对，不是论文数字的复现。

## 代码里已经确认的行为

- Phase 1 `process_item` 输出 `query_plan`、`evidence_triples`、`support`、`debug.variable_candidates` 和 `debug.entity_map`。`support.evidence_indices` 指向压缩后的 `evidence_triples` 列表；稳定三元组 ID 是其中的 `kg_triple_id`。
- `rerank_select_topn` 调用本地 cross-encoder，不是 LLM API。`N_eff` 只用于丢掉过于分散的变量候选。本内核提交绑定时不使用 `N_eff` 或 top-1 分数。
- Phase 2 对 unresolved triple 检索句子，但只把段落交给后续生成。它不产生可执行关系绑定，也不以恢复实体为锚点重跑后继图检索。
- resolved 分支的文本记录经常只有 `title` 和 `context`。KG 三元组通常有 `title` 和 `evidence`，没有句子下标。适配器把这种来源标成可能歧义，而不是自动算作支持。
- `retrieve_musique` 先算出 coarse 相似度，随后把 `similarity` 写成 `0.0`。扩展到 MuSiQue 之前要单独处理，不能只在本方法里改。
- `components/phase1/legacy_impl.py` 在导入时设置 HF 镜像并构造 embedding client。单元测试不导入该模块。
- 配置写 `planned_queries/`，仓库目录是 `planner_queries/`。`require_path` 发现配置路径不存在时会指出真实文件，但不会自动改写路径。
- 根目录没有 LICENSE 文件。`kg/config.py` 含有硬编码的 API key 默认值；本仓库不复制该默认值，也不把它写入日志。

## 产物

固定提交里有 2Wiki 原始 JSON、`planner_queries/2wiki_data/query_graph_v8_2wiki.json` 和 `KGs/KG_2wiki` 的 jsonl。Phase 1 / Phase 2 证据 JSON 是运行产物，不在该树里。因此本批没有做真实 20 题的端到端定位。`synthetic_localization_cohort(20)` 和 `tests/test_upstream_adapter.py` 只证明适配器在同构小样上能对上实体 ID、三元组 ID、query triple index，并隔离 gold。这不是 2Wiki 实验。

## 与原始实现的差异

本仓库的执行器是受控内核，不是 “原始 CS-RAG”。差异包括：图候选默认要过来源检查才成为已接纳证明；文本绑定后可以选择重入一跳图检索；不使用 “交集为空就并集”。上游运行封装只打印命令，不执行。
