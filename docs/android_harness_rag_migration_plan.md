# Android Harness + RAG 迁移计划

目标：把 Simple_Harness 中已经验证过的 Harness、Character System 和 RAG 逻辑迁移到 `MiniCPM-V-demo-Android`，让 Android app 能在本地模型推理链路前完成“角色编译、文档检索、Prompt 组装、来源追踪”，再通过现有 MiniCPM GGUF/JNI 推理接口生成回答。

本文档只描述迁移计划，不假设 Python 运行时会直接进入 Android app。Android 端应复用现有 Kotlin/JNI 推理与模型管理层，把 Simple_Harness 的业务逻辑重写或转译为 Kotlin，并把数据、索引、角色包以可打包资源或 app 私有文件形式接入。

## 1. 当前基线

### 1.1 Simple_Harness 侧

当前可迁移的核心链路是：

```text
data/ 文档
  -> simple_rag.store.VectorIndex
  -> simple_rag.embeddings.HashEmbeddingModel 或 TransformersEmbeddingModel
  -> character_system.runtime.PromptCompiler
  -> simple_rag.character_rag.CharacterRagPromptBuilder
  -> simple_rag.llm.HarnessManagedLLM / GGUFLLM
```

关键模块职责：

- `simple_rag/store.py`：加载文档、分块、保存 `index.json`、按余弦相似度检索 top-k。
- `simple_rag/embeddings.py`：当前离线可用的 `HashEmbeddingModel` 是确定性的 n-gram 哈希向量；可作为 Android 端第一版实现。
- `simple_rag/prompt.py`：普通 RAG prompt 组装，输出 system/user 两条消息。
- `simple_rag/character_rag.py`：Character + RAG 组合逻辑，把角色身份和剧情边界放入 system，把外部检索资料放入 user 的 `<外部检索资料>` 区块。
- `character_system/runtime/prompt_compiler.py`：角色卡、剧情事件、知识边界、动态状态和关系选择的核心编译器。
- `simple_rag/llm.py`：`HarnessManagedLLM` 只负责把 Harness 注册模型解析成 GGUF 文件路径，再委托 `GGUFLLM` 推理；RAG 本身不依赖这个 Python LLM 实现。
- `harness_logic/`：模型注册、模型文件布局、会话、Memory 和 mock/backend 抽象。Simple_Harness 中的最新接入已把模型 artifact 默认放在根目录 `models/<model_id>/<artifact_file>`，而状态/session/memory 默认放在 `harness_logic/data/`。

边界要求：

- 不要把外部 RAG 资料混入角色长期身份。
- 保持检索、prompt builder、sources 输出语义不变。
- 不要把 mock backend 或模型文件存在误认为真实推理验证。

### 1.2 Android app 侧

`MiniCPM-V-demo-Android` 当前已有可接入的 Harness 基础：

```text
MainActivity
  -> com.example.minicpm_v_demo.harness.HarnessFacade
  -> HarnessBackend / LlamaBackendAdapter
  -> LlamaEngine
  -> app/src/main/cpp/* JNI / llama.cpp runtime
```

已存在能力：

- `HarnessModelRegistry.kt`：从 `ModelInfo.AVAILABLE_MODELS` 生成 Harness model spec。
- `LlamaModelStore.kt`：管理当前选中模型、artifact 文件路径和下载完整性。
- `HarnessFacade.kt`：统一暴露模型加载、图像/视频 prefill、文本生成、下载和删除。
- `LlamaBackendAdapter.kt`：把 Harness backend 接口转发到 `LlamaEngine`。
- `MainActivity.kt`：目前用户输入直接调用 `harness.sendUserPrompt(userMsg)`，还没有 RAG/角色 prompt 注入层。

第一版 Android 迁移不需要改 JNI 模型推理核心；主要是在 `MainActivity -> HarnessFacade.sendUserPrompt()` 之间插入 Android 端 RAG/角色编排层。

## 2. 目标架构

Android 端新增一个本地编排层，建议包名：

```text
app/src/main/java/com/example/minicpm_v_demo/harness/rag/
```

目标调用链：

```text
MainActivity
  -> AndroidRagOrchestrator.answer(userInput, mode, characterId, runtimeContext)
       -> DocumentIndex.search()
       -> CharacterPromptCompiler.buildNpcPrompt()
       -> CharacterRagPromptBuilder.build()
       -> ChatTemplateRenderer.render(messages)
  -> HarnessFacade.sendUserPrompt(compiledPromptText)
  -> LlamaEngine / JNI / GGUF
```

第一版可以把 `messages: List<ChatMessage(role, content)>` 渲染成单个字符串传给 `sendUserPrompt()`。中长期应评估 JNI/native 层是否能接收真正的 chat messages 或 system prompt，以减少模板偏差。

## 3. 模块迁移拆分

### 3.1 数据与资产布局

Android app 内建议采用两层数据：

```text
app/src/main/assets/harness/
  characters/
  prompts/
  story/
  rag_indexes/
  rag_documents_manifest.json

context.filesDir/harness/
  indexes/
  documents/
  sessions/
  memory/
  settings.json
```

规则：

- 随 app 发布的角色、剧情事件、默认索引用 `assets/harness/`。
- 用户导入的文档、增量索引、会话和记忆放 `context.filesDir/harness/`。
- 模型 artifact 继续沿用现有 `context.filesDir/models/<model_id>/`，不要混入 RAG 数据目录。
- 初次启动时从 assets 复制默认数据到私有目录，记录 schema version，后续升级做迁移。

需要新增：

- `HarnessDataPaths.kt`：统一返回 `charactersDir`、`storyDir`、`promptsDir`、`indexDir`、`sessionsDir`、`memoryDir`。
- `AssetBootstrapper.kt`：复制 assets 默认数据，校验版本，避免覆盖用户数据。
- `HarnessDataManifest`：记录数据版本、索引版本、embedding 类型、chunk 参数。

### 3.2 RAG 索引与检索

优先迁移 `HashEmbeddingModel + VectorIndex`，因为它不依赖大模型或第三方 Android 推理库，能最快验证端到端流程。

Kotlin 对应类：

```text
EmbeddingModel
HashEmbeddingModel
Chunk
IndexedChunk
SearchResult
VectorIndex
DocumentLoader
DocumentChunker
```

实现要求：

- 保持 Simple_Harness 的默认参数：`dimensions=384`、`ngram_min=2`、`ngram_max=4`、`chunk_size=900`、`overlap=120`。
- `VectorIndex` 支持读取 Simple_Harness 生成的 `index.json`，第一版可先只读不写。
- `SearchResult` 必须保留 `source_path`、`start`、`end`、`score`，用于 UI 来源展示和调试。
- 余弦相似度逻辑保持一致；哈希函数需要与 Python `hashlib.blake2b(digest_size=8)` 对齐。

风险：

- Android 标准库没有直接等价的 BLAKE2b。可选方案：
  1. 引入 Bouncy Castle 或可用的轻量哈希实现。
  2. 第一版离线预构建索引，只在端上对 query 做同算法 embedding。
  3. 临时使用 SHA-256 会破坏与 Simple_Harness 预构建索引的向量一致性，不建议除非索引也全部在端上重建。

验收：

- 使用同一批中文 query，对比 Python 与 Kotlin top-k source 顺序。
- 至少覆盖空 query、短中文、跨文档检索、top_k 非法值。

### 3.3 Character System

将 `character_system/runtime/` 的核心逻辑迁移为 Kotlin：

```text
CharacterPromptCompiler
RuntimeContext
KeywordStoryRetriever
KnowledgeBoundaryFilter
MemoryRetriever
CharacterValidator
```

输入数据保持 JSON/JSONL：

- `characters/<npc_id>.json`
- `story/story_events.jsonl`
- `prompts/roleplay_system.prompt`
- `schemas/*.json` 可在测试中使用，app runtime 可做轻量校验。

迁移原则：

- `PromptCompiler.buildNpcPrompt()` 的输出仍为两条消息：system 和 user。
- system 内保留角色身份、核心人格、剧情 cutoff、动态状态、关系、授权剧情事实、个人经历和对话摘要。
- user 初始只保留玩家原始输入；外部 RAG 资料由 `CharacterRagPromptBuilder` 后续放进 user 的 `<外部检索资料>`。
- `story_cutoff` 默认来自角色卡 `knowledge_scope.story_cutoff`，可由 UI 或剧情状态覆盖。
- `max_character_chars` 仍用于角色 prompt 预算控制。

建议不要第一阶段迁移完整 JSON Schema 校验库。先实现必要字段检查和清晰错误，再在 JVM unit test 中用 fixture 覆盖。

### 3.4 Character + RAG Prompt Builder

迁移 `simple_rag.character_rag.CharacterRagPromptBuilder`，Kotlin 端建议接口：

```kotlin
data class CompiledChatPrompt(
    val messages: List<HarnessChatMessage>,
    val sources: List<RagSource>,
    val characterDebug: Map<String, Any?>,
    val contextText: String
)

class CharacterRagPromptBuilder(
    private val maxContextChars: Int = 5000
)
```

必须保持的 prompt 语义：

- system = 角色 system + `【外部检索资料使用规则】`。
- user = `<外部检索资料>` + 检索资料 + `</外部检索资料>` + `<玩家问题>` + 用户输入 + `</玩家问题>`。
- 没有检索结果时写入 `没有检索到可用资料。`。
- sources 输出包含 source/start/end/score。

同时保留普通 RAG 的 `PromptBuilder`，用于非角色模式：

- system = role rules + 检索资料事实约束。
- user = `<检索资料>` + context + `<用户问题>`。

### 3.5 Chat 模板和模型输入

当前 `HarnessFacade.sendUserPrompt(message: String)` 只接收单个字符串。迁移初期需要新增：

```text
ChatTemplateRenderer.kt
```

职责：

- 把 `List<HarnessChatMessage>` 渲染成 MiniCPM/Qwen/Llama 可接受的单字符串。
- 根据 `HarnessModelSpec.family` 选择默认模板。
- 对 unknown family 使用保守模板：

```text
<|system|>
...

<|user|>
...

<|assistant|>
```

后续优化：

- 如果 native/JNI 层支持独立 system prompt 或 chat message array，应把 `HarnessBackend.sendUserPrompt()` 扩展为 `sendChatMessages(messages, predictLength)`。
- 保持文本模式、视觉模式和视频模式共存：图片/视频 prefill 之后，RAG 编译出的 user prompt 仍作为文本问题发送。

### 3.6 Session 与 Memory

Simple_Harness 的 `harness_logic/session.py` 和 `memory.py` 目前能持久化会话与记忆，但 Memory 尚未注入 Simple_Harness prompt 的主链路。Android 迁移建议分两步：

第一步：

- 实现 `ChatSessionStore`，保存当前会话 turn log。
- 每次生成完成后保存 user/assistant 文本、sources、character_id、story_cutoff、model_id。
- UI 暂不要求历史会话列表，只保证数据结构可用。

第二步：

- 实现 `MemoryStore`，支持 save/list/search/delete。
- 把会话摘要或选中的 memory 接入 `RuntimeContext.conversation_summary`。
- 明确区分 `session_memory` 和 `character_memory`，避免把未授权剧情写进角色长期认知。

## 4. Android UI 接入点

### 4.1 最小可用改动

在 `MainActivity.sendMessage()` 中，当前逻辑是：

```kotlin
harness.sendUserPrompt(userMsg)
```

迁移后变成：

```kotlin
val compiled = ragOrchestrator.compile(
    userInput = userMsg,
    characterId = selectedCharacterId,
    mode = currentRagMode,
    runtimeContext = currentRuntimeContext
)
harness.sendUserPrompt(compiled.renderedPrompt)
```

同时在消息气泡或调试面板中保存：

- `compiled.sources`
- `compiled.characterDebug`
- `compiled.contextText`
- `compiled.messages`

### 4.2 UI 控制

建议新增低侵入控制：

- 设置页或模型页：RAG 开关、角色开关、默认 NPC、top_k、max_context_chars。
- 聊天页：当前角色选择、sources 展开入口。
- Debug 模式：显示最终 prompt、检索来源、角色编译 debug。

第一阶段不要大改聊天 UI。先用默认角色和默认索引跑通链路，再补充交互。

## 5. 分阶段实施计划

### Phase 0：迁移前对齐

交付物：

- 明确 Android 仓库要接入的 Simple_Harness 版本。
- 确认 `character_system/`、`data/novel_test`、默认 `.rag_index/index.json` 是否作为 assets 进入 app。
- 确认首个 Android 目标模型：建议 `minicpm5-0.9b` 文本模型，避免视觉链路干扰。

检查项：

- Simple_Harness 执行 `./run.sh test`。
- Simple_Harness 执行 `./run.sh harness list`，确认模型 ID 与 artifact 名称。
- Android 执行现有 Gradle unit test 或至少 assemble debug。

### Phase 1：数据打包与读取

交付物：

- `assets/harness/` 数据布局。
- `HarnessDataPaths.kt`。
- `AssetBootstrapper.kt`。
- 数据 manifest 和版本检查。

验收：

- app 首次启动后，默认角色、剧情事件、prompt template、RAG index 能出现在 `context.filesDir/harness/`。
- 重启 app 不覆盖用户侧数据。

### Phase 2：Kotlin RAG 检索

交付物：

- `HashEmbeddingModel.kt`。
- `VectorIndex.kt`。
- `DocumentChunker.kt`。
- `RagRepository.kt` 或 `RagSearchService.kt`。

验收：

- Kotlin unit test 读取 Simple_Harness 生成的 `index.json`。
- 与 Python 对照 query 的 top-k sources 一致或误差可解释。
- 检索结果能输出 source/start/end/score。

### Phase 3：角色编译器迁移

交付物：

- `CharacterPromptCompiler.kt`。
- `RuntimeContext.kt`。
- `KeywordStoryRetriever.kt`。
- `KnowledgeBoundaryFilter.kt`。
- `MemoryRetriever.kt`。
- fixtures 对照测试。

验收：

- 对 `xuan_an`、`lu_jiangxian` 的固定输入，Kotlin 输出的 system prompt 包含与 Python 相同的关键 section。
- story cutoff、生效关系、授权事实、dropped item 逻辑可通过测试验证。
- 未知 NPC、未知 cutoff、空输入给出明确错误。

### Phase 4：Prompt 组装与模型接入

交付物：

- `CharacterRagPromptBuilder.kt`。
- `PromptBuilder.kt`。
- `ChatTemplateRenderer.kt`。
- `AndroidRagOrchestrator.kt`。
- `HarnessFacade` 新增可选的编译后 prompt 入口，或在 `MainActivity` 内先行调用 orchestrator。

验收：

- Debug 输出中能看到最终 system/user messages。
- 真实发送给 `LlamaEngine` 的字符串可被记录或显示。
- 无模型时可用 fake backend 测试编译和渲染；有模型时用 `minicpm5-0.9b` 做真实端上推理。

### Phase 5：UI 和来源展示

交付物：

- 聊天页 RAG/角色开关。
- sources 展示入口。
- Debug prompt 展示入口。
- 错误提示：无索引、无角色、索引版本不兼容、模型未下载。

验收：

- 普通聊天、RAG 聊天、角色 RAG 聊天三种模式可切换。
- 模型未下载时仍能完成 prompt 编译调试，不触发推理。
- sources 不阻塞流式生成。

### Phase 6：Session/Memory 接入

交付物：

- `ChatSessionStore.kt`。
- `MemoryStore.kt`。
- `conversation_summary` 接入 `RuntimeContext`。
- 会话恢复策略。

验收：

- 生成完成后保存 turn、sources、角色、模型和时间。
- 重启 app 后可恢复最近会话。
- Memory 注入必须有测试证明不会越过角色知识边界。

## 6. 测试策略

### 6.1 Simple_Harness 源侧测试

在迁移前固定一批 golden cases：

```text
tests/android_migration_cases/
  rag_queries.json
  character_prompt_cases.json
  character_rag_cases.json
```

每个 case 保存：

- 输入 query。
- npc_id。
- story_cutoff。
- top_k。
- sources。
- system/user prompt 关键片段。
- debug 中 selected/dropped/retrieval_decisions 的关键字段。

### 6.2 Android JVM 单元测试

建议新增：

```text
app/src/test/java/com/example/minicpm_v_demo/harness/rag/
```

覆盖：

- Hash embedding 确定性。
- Vector index schema 读取。
- top-k 排序。
- Character prompt 编译。
- Character + RAG prompt 组装。
- Chat template 渲染。
- 错误路径。

### 6.3 Android instrumentation / 手动验收

覆盖：

- 首次启动 asset bootstrap。
- 模型未下载时的提示。
- 模型已下载时端上生成。
- 图像/视频模式下 RAG 文本 prompt 与 prefill 不冲突。
- 低内存或长 prompt 下的截断策略。

## 7. 关键风险与决策

### 7.1 不建议直接运行 Python

Android 中嵌入 Python 会增加包体、启动成本、ABI 复杂度和调试难度，也难以和现有 Kotlin/JNI 推理生命周期对齐。建议将逻辑迁移为 Kotlin。

### 7.2 Embedding 一致性

如果继续使用 Simple_Harness 的预构建 `index.json`，Android query embedding 必须与 Python `HashEmbeddingModel` 一致。否则 top-k 会漂移。

决策建议：

- 第一版使用同算法 Kotlin hash embedding。
- 第二版再评估小型端侧 embedding 模型或服务器/桌面预构建索引。

### 7.3 Prompt 长度与上下文窗口

MiniCPM5 文本模型当前 registry runtime hint 是 `context_size=4096`。角色 system、外部检索资料和用户输入叠加后很容易超长。

需要实现：

- `max_character_chars`。
- `max_context_chars`。
- 模型级 context budget。
- 最终 rendered prompt 字符数/估算 token 数日志。

### 7.4 Chat 模板偏差

Simple_Harness 的 Python GGUF 走 `llama_cpp.create_chat_completion(messages=...)`。Android 当前 `sendUserPrompt(String)` 很可能只按现有 native 上下文追加用户文本。直接把 system/user 渲染成字符串可能与 Python 输出存在差异。

迁移第一版接受这个差异，但必须：

- Debug 展示最终 rendered prompt。
- 为不同 model family 保留模板分支。
- 后续评估 native 层 chat message 支持。

### 7.5 视觉模型共存

RAG 是文本上下文增强，MiniCPM-V 的图像/视频能力通过 `prefillImage()` / `prefillVideoFrames()` 进入 native context。迁移时不要改视觉 prefill 的时序，只在发送文本问题前替换为编译后的 prompt。

## 8. 推荐文件清单

Android 新增文件：

```text
app/src/main/java/com/example/minicpm_v_demo/harness/data/HarnessDataPaths.kt
app/src/main/java/com/example/minicpm_v_demo/harness/data/AssetBootstrapper.kt
app/src/main/java/com/example/minicpm_v_demo/harness/data/HarnessDataManifest.kt

app/src/main/java/com/example/minicpm_v_demo/harness/rag/EmbeddingModel.kt
app/src/main/java/com/example/minicpm_v_demo/harness/rag/HashEmbeddingModel.kt
app/src/main/java/com/example/minicpm_v_demo/harness/rag/VectorIndex.kt
app/src/main/java/com/example/minicpm_v_demo/harness/rag/DocumentChunker.kt
app/src/main/java/com/example/minicpm_v_demo/harness/rag/RagSearchService.kt
app/src/main/java/com/example/minicpm_v_demo/harness/rag/PromptBuilder.kt
app/src/main/java/com/example/minicpm_v_demo/harness/rag/CharacterRagPromptBuilder.kt
app/src/main/java/com/example/minicpm_v_demo/harness/rag/AndroidRagOrchestrator.kt

app/src/main/java/com/example/minicpm_v_demo/harness/character/RuntimeContext.kt
app/src/main/java/com/example/minicpm_v_demo/harness/character/CharacterPromptCompiler.kt
app/src/main/java/com/example/minicpm_v_demo/harness/character/KeywordStoryRetriever.kt
app/src/main/java/com/example/minicpm_v_demo/harness/character/KnowledgeBoundaryFilter.kt
app/src/main/java/com/example/minicpm_v_demo/harness/character/MemoryRetriever.kt

app/src/main/java/com/example/minicpm_v_demo/harness/chat/HarnessChatMessage.kt
app/src/main/java/com/example/minicpm_v_demo/harness/chat/ChatTemplateRenderer.kt
```

Android 修改文件：

```text
app/src/main/java/com/example/minicpm_v_demo/MainActivity.kt
app/src/main/java/com/example/minicpm_v_demo/harness/HarnessFacade.kt
app/src/main/java/com/example/minicpm_v_demo/harness/HarnessBackend.kt
app/src/main/java/com/example/minicpm_v_demo/harness/LlamaBackendAdapter.kt
app/src/main/java/com/example/minicpm_v_demo/harness/HarnessModelSpec.kt
app/build.gradle.kts
```

Simple_Harness 侧新增迁移辅助：

```text
tests/android_migration_cases/
scripts/export_android_migration_fixture.py
```

## 9. 验收标准

迁移完成后，至少满足：

- Android app 可以加载默认角色和默认 RAG index。
- 输入同一问题时，Android 端能输出与 Simple_Harness 同结构的 final messages。
- Android 端显示或记录 sources。
- 文本模型 `minicpm5-0.9b` 已下载时，角色 RAG prompt 能进入真实本地 GGUF 推理。
- 无模型时，仍可调试 prompt、sources 和 character_debug。
- 视觉模型仍能正常图片/视频 prefill，RAG 只替换文本 prompt，不破坏现有媒体流程。
- JVM tests 覆盖 RAG 检索、角色编译、prompt 组装和 chat 模板。

## 10. 建议执行顺序

1. 在 Simple_Harness 固化 golden fixtures。
2. 在 Android 建立 `assets/harness/` 和数据 bootstrap。
3. 迁移 HashEmbedding + VectorIndex，只做检索。
4. 迁移 CharacterPromptCompiler，只做 prompt 编译。
5. 迁移 CharacterRagPromptBuilder 和 ChatTemplateRenderer。
6. 在 `MainActivity` 接入 orchestrator，默认使用文本模型验证。
7. 加 sources/debug UI。
8. 接入 session 和 memory。
9. 再处理用户文档导入、端上建索引和高级 embedding。

第一轮最小闭环应控制在 Phase 1 到 Phase 4：只要 Android 能从默认数据检索、编译角色 RAG prompt，并把最终 prompt 送入现有 `HarnessFacade.sendUserPrompt()`，就可以开始真实端上推理验证。
