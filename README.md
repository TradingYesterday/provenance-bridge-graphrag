# provenance-bridge-graphrag

首轮只检验一件事：查询计划和入口正确、某条非末跳关系在图里缺失、支持它的文本还在、恢复出的桥实体仍能在当前图里寻址时，带来源校验的关系绑定能不能把系统重新送回图检索，并取到后继证据。

这是任务书的第一批（任务 1–3）：数据契约、CS-RAG 固定提交的只读适配，以及不调用模型 API 的执行内核。Simple、CoG、Iterative text 和上游完整运行都还没有实现，不能据此声称机制成立。

## 运行

```bash
pip install -e ".[dev]"
python -m pytest
python scripts/run_diagnostics.py --out outputs/diagnostics
python scripts/summarize_results.py outputs/diagnostics/summary.json
python scripts/validate_inputs.py --config configs/2wiki_pilot.yaml
```

诊断集是合成缺边案例，不是 2Wiki 分数。`scripts/run_pilot.py` 会拒绝启动 100–200 题 API 实验。

## 边界

- 模型走环境变量里的 OpenAI 兼容接口。本批没有接通传输层；没有密钥时直接报错，不会编造答案。
- 检索和重排的真实模型不在本批加载。诊断用固定 oracle 候选和程序校验。
- 不复制 CS-RAG 源码。适配器只读 JSON / JSONL。固定提交是 `myz12138/CS-RAG@c84a081c728dd45c284a7165031c853eef4e8f24`。
- 数据、KG、权重和 `.env` 不入库。

更细的约定见 `docs/`。
