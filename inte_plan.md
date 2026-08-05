# Character System 整合计划

## 当前状态

当前根目录项目运行时不依赖 `character_system`。

从当前代码可以确认：

- 根目录 CLI 和 shell 入口通过 `run.sh` 调用 `simple_rag.cli`。
- `simple_rag` 只导入自己的模块：`embeddings`、`store`、`roles`、`prompt`、`pipeline`、`documents` 和 `llm`。
- 根目录运行路径没有导入 `character_system`、`PromptCompiler`、`RuntimeContext`、`story_events` 或角色系统 prompt 模板。
- `character_system` 目前通过把自身目录插入 `sys.path` 来导入自己的 `runtime` 包；它还没有被根目录 `pyproject.toml` 正式打包。

因此，根目录 RAG 项目可以在不依赖 `character_system` 的情况下完成索引、检索、RAG prompt 编译和本地模型聊天。

## Character System 提供的能力

`character_system` 是一套独立的 NPC prompt 编译器。它提供：

- `character_system/characters/` 下的结构化 NPC 角色卡；
- `character_system/story/story_events.jsonl` 中的公共剧情事实；
- `character_system/prompts/` 下的 prompt 模板；
- `character_system/runtime/` 下的知识边界过滤、记忆检索和 prompt 编译逻辑；
- 用于校验、prompt 编译、QA 用例编译和角色卡报告生成的脚本。

它的输出结构已经和根目录的模型抽象兼容：

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ]
}
```

两者主要区别在语义层面：

- 根目录 `simple_rag` prompt：面向已索引资料的、带角色口吻的 RAG 问答 prompt；
- `character_system` prompt：面向 NPC 扮演的 prompt，包含剧情截止点、知识边界、人物关系和情节记忆。

## 直接使用前的阻塞点

当前导入进来的 `character_system` 不是完全自包含的。

它的校验逻辑依赖证据原文文件，例如：

- `1453.txt`
- `1454.txt`
- `1455.txt`
- `1456.txt`
- `1457.txt`
- `1458.txt`
- `qa.json`，用于 QA 批量脚本

如果缺少这些文件，默认的 `PromptCompiler(ROOT)` 会在证据校验阶段失败。

数据文件不应该为了适配代码而拆散。推荐保留根目录统一数据结构，例如：

```text
data/
  novel_test/
    1453.txt
    1454.txt
    1455.txt
    1456.txt
    1457.txt
    1458.txt
    qa.json
  world_cup/
    ...
```

角色系统通过 `evidence_root` 配置指向 `data/novel_test`。这样 `character_system` 保持代码和角色卡职责，`data/` 保持统一外部数据职责。

## 整合目标

整合目标是让 `character_system` 成为根目录项目的一等功能，同时不破坏现有 RAG 流程。

目标能力：

- 保持现有命令可用：
  - `./run.sh index`
  - `./run.sh search`
  - `./run.sh prompt`
  - `./run.sh chat`
- 新增角色系统相关命令：
  - 编译单个 NPC prompt；
  - 可选地通过同一套本地 LLM 抽象生成回复；
  - 校验角色系统数据；
  - 从根目录测试命令中运行角色系统测试。
- 复用现有根目录的 `LocalLLM.generate(messages, options)` 接口。
- 支持通过 `evidence_root` 指向统一数据目录，例如 `data/novel_test`。
- 除非明确需要混合模式，否则不要把 RAG 检索 chunk 混进 NPC 的 system prompt。

## 建议架构

### 阶段一：把 Character Runtime 包化

让 `character_system` 可以从根目录项目正常 import。

需要修改：

- 新增 `character_system/__init__.py`。
- 把脚本中的本地 `sys.path` 插入式导入改为包导入。
- 更新 `pyproject.toml` 的 package discovery，使其包含 `character_system*`。
- 保留数据文件、prompt、schema 和示例作为包邻近资源。

期望结果：

```python
from character_system.runtime import PromptCompiler, RuntimeContext
```

可以从仓库根目录正常运行。

### 阶段二：新增根目录 CLI 命令

在 `simple_rag.cli` 中扩展角色系统命令。

建议命令：

```bash
python -m simple_rag.cli character-prompt \
  --character-root character_system \
  --evidence-root data/novel_test \
  --npc lu_jiangxian \
  --query "玄谙究竟是什么？" \
  --cutoff evt-010 \
  --max-chars 4500 \
  --debug
```

```bash
python -m simple_rag.cli character-chat \
  --character-root character_system \
  --evidence-root data/novel_test \
  --npc lu_jiangxian \
  --model /path/to/local/chat-model \
  --query "玄谙究竟是什么？"
```

`character-prompt` 应该输出 `messages` JSON，行为类似当前根目录的 `prompt` 命令。

`character-chat` 应该执行以下流程：

1. 编译角色系统 messages；
2. 把 messages 传给 `LocalLLM.generate`；
3. 打印 assistant 生成结果；
4. 可选地在分隔符后打印 debug 元数据。

### 阶段三：新增 `run.sh` 快捷入口

等 Python CLI 跑通后，再给 `run.sh` 增加快捷命令。

建议命令：

```bash
./run.sh character-prompt "玄谙究竟是什么？"
```

```bash
NPC_ID=lu_jiangxian CUTOFF=evt-010 MODEL_PATH=/path/to/model ./run.sh character-chat "玄谙究竟是什么？"
```

建议环境变量：

- `CHARACTER_ROOT=character_system`
- `CHARACTER_EVIDENCE_ROOT=data/novel_test`
- `NPC_ID=lu_jiangxian`
- `CUTOFF=`
- `CHARACTER_MAX_CHARS=4500`
- `CHARACTER_DEBUG=`

### 阶段四：清理数据路径

移除角色系统对父目录的隐式依赖，但不要拆散根目录统一数据文件夹。

推荐目标结构：

```text
data/
  novel_test/
    1453.txt
    1454.txt
    ...
    qa.json
```

然后更新校验代码，让它接受：

- `character_root`
- `evidence_root`，例如 `data/novel_test`
- 可选的 `qa_path`

编译器不应该再假设 `root.parent` 就是小说原文目录。

### 阶段五：补充测试

新增根目录集成测试。

最低测试范围：

- 现有 `tests/test_rag_pipeline.py` 继续通过；
- `character_system/tests/test_character_system.py` 可以从根目录运行；
- 根目录 CLI `character-prompt` 返回且只返回两条 messages；
- `PromptEchoLLM` 可以基于角色系统 messages 生成回显；
- 缺失证据文件时给出明确校验错误；
- 如果有意支持 `validate_on_load=False`，则覆盖该模式的行为。

根目录测试命令可以继续保持：

```bash
python -m unittest discover -s tests -v
```

也可以在 `run.sh test` 中扩展为同时运行两组测试：

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s character_system/tests -v
```

### 阶段六：可选的 RAG + Character 混合模式

这个阶段应在纯角色系统流程稳定后再做。

可能新增的混合模式：

- `character-rag-prompt`：先检索外部资料，再把选中的事实加入角色用户消息；
- `character-rag-chat`：同上，然后调用本地模型生成。

重要规则：

不要把不可信的 RAG chunk 直接放入 NPC 身份和人格所在的 system 层。

更稳妥的结构：

- system：只放角色系统编译器输出；
- user：放检索资料和用户真实问题；
- debug：记录检索来源和角色编译决策。

## 推荐实现顺序

1. 把证据文件放入统一数据目录，例如 `data/novel_test`，并通过 `evidence_root` 指向它。
2. 把 `character_system` 做成根目录可 import 的包。
3. 新增 `character-prompt` CLI 命令。
4. 新增复用 `LocalLLM` 的 `character-chat` CLI 命令。
5. 增加 `run.sh` 快捷入口。
6. 补充测试。
7. 最后再考虑 RAG + Character 混合命令。

## 初始整合阶段不做的事

- 不重写现有 RAG prompt builder。
- 不立即合并 `configs/roles/*.json` 和 `character_system/characters/*.json`。
- 不在角色系统代码里硬编码 MiniCPM、Qwen 或其他模型的特殊 chat template。
- 不在生产命令中静默绕过证据校验。

## 最终目标

整合完成后，根目录项目应支持两种相互独立的 prompt 模式：

- RAG 模式：基于已索引本地资料，用轻量角色配置回答问题。
- Character 模式：基于结构化角色卡、剧情事件、知识边界和动态状态扮演 NPC。

两种模式共用同一个本地 LLM 生成接口，但保持各自的 prompt 构建逻辑独立。
