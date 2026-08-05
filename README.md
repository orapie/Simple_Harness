# Simple Harness RAG

这是一个本地优先的最小 RAG 项目，面向端侧或本机大模型实验：

- 递归读取 `data/` 里的资料，支持 `.txt`、`.md`、`.json`、`.jsonl`、`.csv`，安装可选依赖后支持 `.pdf` 和 `.docx`。
- 将资料切分为 chunk，建立本地向量索引。
- 检索相关 chunk，并把搜索结果组装进 prompt。
- 通过角色 JSON 控制大模型的身份、语气、行为规则和知识边界。
- 可接入 `character_system`，把小说角色卡、剧情边界和 `data/` 中其他资料合并成 Character + RAG prompt。
- 可接入任意 Hugging Face `transformers` 兼容的本地模型路径。

推荐数据结构：

```text
data/
├── novel_test/    # character_system 使用的小说证据文件和 qa.json
└── world_cup/     # 普通 RAG / Character + RAG 可检索的其他资料
```

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

如需读取 PDF 或 Word：

```bash
python -m pip install -e ".[documents]"
```

## 建索引

默认索引 `data/`，输出到 `.rag_index/index.json`。不传 embedding 模型时使用内置哈希向量，便于离线快速验证；传入本地 embedding 模型路径时使用 `transformers` 生成向量。

简易脚本：

```bash
./run.sh index
```

```bash
python -m simple_rag.cli index --data data --index .rag_index/index.json
```

使用本地 embedding 模型：

```bash
python -m simple_rag.cli index \
  --data data \
  --index .rag_index/index.json \
  --embedding-model /path/to/local/embedding-model \
  --device mps
```

## 检索

```bash
./run.sh search "西班牙夺冠后有哪些争议？"
```

```bash
python -m simple_rag.cli search \
  --index .rag_index/index.json \
  --query "西班牙夺冠后有哪些争议？" \
  --top-k 5
```

## 只编译 Prompt

```bash
ROLE_PATH=configs/roles/xuan_an.json ./run.sh prompt "请总结世界杯决赛后的关键新闻。"
```

```bash
python -m simple_rag.cli prompt \
  --index .rag_index/index.json \
  --role configs/roles/default.json \
  --query "请总结世界杯决赛后的关键新闻。"
```

## 接入本地大模型聊天

`--model` 指向本地 `transformers` 兼容模型目录。未指定 `--model` 时，命令使用 `PromptEchoLLM` 回显最终 messages，方便检查检索和 prompt 注入效果；真正生成回答需要设置本地模型路径。

```bash
MODEL_PATH=/path/to/local/chat-model DEVICE=mps ./run.sh chat "请用角色口吻解释资料里的世界杯商业化争议。"
```

```bash
python -m simple_rag.cli chat \
  --index .rag_index/index.json \
  --role configs/roles/default.json \
  --model /path/to/local/chat-model \
  --device mps \
  --query "请用角色口吻解释资料里的世界杯商业化争议。"
```

常用生成参数：

```bash
python -m simple_rag.cli chat \
  --index .rag_index/index.json \
  --role configs/roles/xuan_an.json \
  --model /path/to/local/chat-model \
  --max-new-tokens 512 \
  --temperature 0.7 \
  --top-p 0.9 \
  --query "玄谙会如何评价这些资料？"
```

## 单独运行 Character System

`character_system` 负责把小说证据、角色卡、剧情事件、知识边界和玩家输入编译成可送给大模型的两条 messages。它默认会自动读取 `data/novel_test`，不需要额外传 `--evidence-root`。

校验角色卡和剧情数据：

```bash
python3 character_system/scripts/validate_character_data.py
```

正常输出：

```text
OK: 2 NPC cards, 20 shared events, all references valid
```

只编译 NPC prompt：

```bash
python3 character_system/scripts/compile_prompt.py \
  --npc lu_jiangxian \
  --input "玄谙究竟是什么？" \
  --cutoff evt-010 \
  --debug
```

输出中真正要送给模型的是 `messages`；`debug` 只用于本地检查知识边界和检索决策。

## 结合 Character System 的 RAG

根目录支持把 `character_system` 编译出的 NPC prompt 与 `data/` 中其他资料的 RAG 检索结果合并。角色身份、人格、剧情截止点和知识边界仍由 `character_system` 控制；外部检索资料会作为本轮资料放入 `user` 消息，不会写入角色长期记忆。

先对统一 `data/` 目录建索引：

```bash
./run.sh index
```

只编译可送入模型的 Character + RAG messages：

```bash
NPC_ID=lu_jiangxian CUTOFF=evt-010 ./run.sh character-rag-prompt "用陆江仙的口吻解释世界杯商业化争议。"
```

成功时输出 JSON，核心结构如下：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你正在扮演故事中的一个角色..."
    },
    {
      "role": "user",
      "content": "<外部检索资料>..."
    }
  ],
  "sources": [
    {
      "source": ".../data/world_cup/...",
      "start": 0,
      "end": 890,
      "score": 0.170943
    }
  ]
}
```

判断组装成功的标准：

- `messages[0].content` 里包含 NPC 身份信息，例如 `我是陆江仙`；
- `messages[1].content` 里包含 `<外部检索资料>` 和 `<玩家问题>`；
- `sources` 指向 `data/` 中被检索到的资料文件；
- 如果加 `--debug`，还会看到 `character_debug`，用于检查角色系统选中了哪些剧情事实、过滤了哪些未来事件。

等价 Python 命令：

```bash
python -m simple_rag.cli character-rag-prompt \
  --index .rag_index/index.json \
  --character-root character_system \
  --npc lu_jiangxian \
  --cutoff evt-010 \
  --query "用陆江仙的口吻解释世界杯商业化争议。"
```

接入本地大模型生成：

```bash
NPC_ID=lu_jiangxian \
CUTOFF=evt-010 \
MODEL_PATH=/path/to/local/chat-model \
DEVICE=mps \
./run.sh character-rag-chat "用陆江仙的口吻解释世界杯商业化争议。"
```

不设置 `MODEL_PATH` 时，`character-rag-chat` 会回显最终 messages 和 sources；设置 `MODEL_PATH` 后才会调用本地模型生成 assistant 回复。

如果小说证据文件不在默认的 `data/novel_test`，可以显式指定：

```bash
EVIDENCE_ROOT=/path/to/novel_test ./run.sh character-rag-prompt "问题"
```

常用角色参数：

```bash
NPC_ID=lu_jiangxian
NPC_ID=xuan_an
CUTOFF=evt-010
CUTOFF=evt-018
```

## 角色配置

角色文件是普通 JSON：

```json
{
  "id": "default",
  "name": "资料助理",
  "persona": "谨慎、直接、优先依据检索资料回答。",
  "style": ["用中文回答", "先给结论，再给依据"],
  "rules": ["不能把检索资料之外的信息说成事实"],
  "refusal": "资料不足时直接说明不足。"
}
```

新增角色时，把 JSON 放到 `configs/roles/`，再通过 `--role` 指定即可。

## 运行测试

```bash
./run.sh test
```

该命令会同时验证：

- 普通 RAG pipeline；
- Character + RAG pipeline；
- `character_system` 的角色卡、剧情事件和 prompt 编译边界。
