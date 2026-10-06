# 数据契约

核心对象在 `bridge_rag.schemas`。缺核心字段会直接失败。未知字段可以留下。`QuestionCase` 拒绝 `answer`、`ground_truth_answer`、`supporting_facts`、`gold_links`、`missing_edge(s)`、`bridge_label`。这些只放在 `GoldRecord`。

## 项与约束

项类型是 `ENTITY`、`VARIABLE` 或 `LITERAL`。实体 ID 必须带 `graph_namespace`。字面量不能当成图实体 ID。

依赖由变量读写的不动点得出，不按数组顺序猜测。两个写入者写同一个变量，或只含变量的环，得到 `PLAN_UNSUPPORTED`。

首轮文本恢复只处理：恰好一个端点已经是当前图里的实体，另一个端点是实体变量，约束不是末跳，并且有后继约束依赖它。两端都未知就等待上游。两端都已知只检查这对端点，不改写绑定。

## 分支与证明

分支状态：`ACTIVE`、`COMPLETE`、`AMBIGUOUS`、`EXHAUSTED`、`INVALIDATED`、`BUDGET_EXHAUSTED`、`PLAN_UNSUPPORTED`。

证明 `kind` 为 `GRAPH` 或 `TEXT`。文本提交不往原图写边。撤销一个绑定会沿读写依赖使派生绑定和证明失效。

验证决策：`SUPPORTED`、`CONFLICT`、`UNKNOWN`、`INVALID_SPAN`、`UNLINKABLE`、`BUDGET_EXHAUSTED`。语义分数不是校准概率。

## 来源

来源 ID 形如 `{data_version}:{question_id}:{doc_index}:{sent_index}`。字符区间针对原始句子，不拼接标题，也不先做归一化。同句重复时用位置区分。只有标题时如果命中多处，标记歧义，不取第一处。

## 图 ID

上游实体 ID、三元组下标删除边后保持不变。屏蔽用 `active` 掩码。`title2triples` 仍保留原下标，活跃视图才去掉被屏蔽的边。

## 缓存键

包含数据版本、问题、约束、当前绑定、模型、参数和 prompt hash。绑定不同就不是同一次检索。
