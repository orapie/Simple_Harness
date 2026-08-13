# Simple Harness RAG

这是一个本地优先的最小 RAG 项目，面向端侧或本机大模型实验：

- 递归读取 `data/` 里的资料，支持 `.txt`、`.md`、`.json`、`.jsonl`、`.csv`，安装可选依赖后支持 `.pdf` 和 `.docx`。
- 将资料切分为 chunk，建立本地向量索引。
- 检索相关 chunk，并把搜索结果组装进 prompt。
- 通过角色 JSON 控制大模型的身份、语气、行为规则和知识边界。
- 可接入 `character_system`，把小说角色卡、剧情边界和 `data/` 中其他资料合并成 Character + RAG prompt。
- 可接入任意 Hugging Face `transformers` 兼容的本地模型路径。
- 已接入从 MiniCPM Android 提取的 `harness_logic`，可在本项目内运行模型注册、artifact 状态、下载计划、mock backend、角色 prompt、session 和 memory 编排。

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

如需接入 GGUF 格式本地大模型：

```bash
python -m pip install -e ".[gguf]"
```

Apple Silicon 上如果需要 Metal 加速，建议按 `llama-cpp-python` 当前平台说明安装带 Metal 的构建；本项目侧通过 `GGUF_N_GPU_LAYERS` 控制 offload 层数。

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

未指定 `MODEL_PATH` / `--model` 时，命令使用 `PromptEchoLLM` 回显最终 messages，方便检查检索和 prompt 注入效果；真正生成回答需要设置本地模型路径。

当前支持两种本地模型后端：

- Transformers：`MODEL_PATH` 指向 Hugging Face `transformers` 兼容模型目录。
- GGUF：`MODEL_PATH` 指向 `.gguf` 文件，默认会自动选择 GGUF 后端；也可以显式设置 `MODEL_BACKEND=gguf`。

### Transformers 模型目录

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

### GGUF 模型文件

GGUF 后端使用 `llama-cpp-python`。先安装：

```bash
python -m pip install -e ".[gguf]"
```

如果本地还没有模型，可以先从 Hugging Face 下载一个 GGUF 文件。安装下载工具：

```bash
python -m pip install -U huggingface_hub
```

公开模型通常可以直接下载；遇到 gated/private 模型时，先登录：

```bash
hf auth login
```

在 Hugging Face 搜索模型时，优先找带 `GGUF` 的仓库，并选择单个 `.gguf` 文件下载。首次本地测试建议从 `Q4_K_M.gguf` 量化文件开始；它通常比 `Q8_0.gguf` 更省内存，也比更低位量化更稳。下载命令格式如下：

```bash
mkdir -p models
hf download <repo-id> <model-file.gguf> --local-dir models/<model-name>
```

示例：

```bash
hf download bartowski/Llama-3.2-1B-Instruct-GGUF \
  Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --local-dir models/llama-3.2-1b-instruct
```

下载完成后，`MODEL_PATH` 指向这个 `.gguf` 文件：

```bash
MODEL_PATH=models/llama-3.2-1b-instruct/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

`models/` 和 `*.gguf` 已在 `.gitignore` 中忽略，模型文件会留在本地，不会被提交到仓库。

`models/` 是当前项目统一的本地模型文件池：

- 普通 RAG 可以直接用 `MODEL_PATH` 指向任意本地模型文件或 Transformers 模型目录。
- Harness 会按自己的模型注册表读取 `models/<model_id>/<artifact_file>`，例如 `models/llama-3.2-1b-instruct/Llama-3.2-1B-Instruct-Q4_K_M.gguf`。

也就是说，两者不需要两套模型目录；区别在于普通 RAG 直接吃路径，Harness 按模型 ID 和 artifact 文件名管理路径。

普通 RAG 接 GGUF：

```bash
MODEL_PATH=/path/to/model.gguf \
MODEL_BACKEND=gguf \
GGUF_N_CTX=4096 \
GGUF_N_GPU_LAYERS=35 \
./run.sh chat "解释资料里的世界杯商业化争议"
```

Character + RAG 接 GGUF，例如 MiniCPM 一类 GGUF 模型：

```bash
NPC_ID=lu_jiangxian \
CUTOFF=evt-010 \
MODEL_PATH=/path/to/minicpm-model.gguf \
MODEL_BACKEND=gguf \
GGUF_N_CTX=4096 \
GGUF_N_GPU_LAYERS=35 \
./run.sh character-rag-chat "用陆江仙的口吻解释世界杯商业化争议。"
```

等价 Python 命令：

```bash
python -m simple_rag.cli character-rag-chat \
  --index .rag_index/index.json \
  --npc lu_jiangxian \
  --cutoff evt-010 \
  --model /path/to/minicpm-model.gguf \
  --model-backend gguf \
  --gguf-n-ctx 4096 \
  --gguf-n-gpu-layers 35 \
  --query "用陆江仙的口吻解释世界杯商业化争议。"
```

GGUF 常用参数：

```bash
MODEL_BACKEND=gguf          # 可省略；.gguf 后缀会自动识别
GGUF_N_CTX=4096             # 上下文长度
GGUF_N_GPU_LAYERS=35        # GPU/Metal offload 层数；CPU 运行可设 0
GGUF_CHAT_FORMAT=           # 特殊模型需要时指定 llama-cpp-python 的 chat_format
MAX_NEW_TOKENS=512
TEMPERATURE=0.7
TOP_P=0.9
```

如果模型 GGUF 文件内带有 chat template，通常不需要设置 `GGUF_CHAT_FORMAT`。如果某个 MiniCPM GGUF 需要特定 chat format，再按该模型发布说明设置。

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

## MiniCPM Harness Logic

`harness_logic/` 是从 MiniCPM Android 项目迁入的 Python Harness 编排层。它现在作为 Simple Harness 的本地包运行，默认运行状态放在 `harness_logic/data/`，默认模型 artifact 读取仓库根目录的 `models/`，并复用本项目根目录的 `character_system/` 和 `data/novel_test`。

### 当前 models 目录示例

当前仓库根目录的 `models/` 里有这些本地文件：

```text
models/
├── MiniCPM5-1B-Claude-Opus-Fable5-Thinking-Q8_0.gguf
└── llama-3.2-1b-instruct/
    └── Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

它们的使用方式不同：

| 文件 | 当前用途 | 说明 |
| --- | --- | --- |
| `models/MiniCPM5-1B-Claude-Opus-Fable5-Thinking-Q8_0.gguf` | 普通 RAG 直接用 `MODEL_PATH` | 文件在 `models/` 根下，当前不匹配 Harness 注册表要求的 `models/<model_id>/<artifact_file>` 结构。 |
| `models/llama-3.2-1b-instruct/Llama-3.2-1B-Instruct-Q4_K_M.gguf` | Harness 结构化路径示例 | 路径匹配 Harness 的 `llama-3.2-1b-instruct` 注册项；如果它只是占位文件，需要替换成真实 GGUF 后才能真实推理。 |

完整使用顺序建议如下：

1. 安装依赖。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[gguf]"
```

2. 建立 RAG 索引。

```bash
./run.sh index
```

3. 先检查 RAG 检索结果。

```bash
./run.sh search "世界杯决赛后有什么商业争议？"
```

4. 检查最终会送给模型的 RAG prompt。

```bash
ROLE_PATH=configs/roles/xuan_an.json \
./run.sh prompt "请用角色口吻总结资料里的争议。"
```

5. 用当前 MiniCPM5 GGUF 文件直接跑普通 RAG。

```bash
MODEL_PATH=models/MiniCPM5-1B-Claude-Opus-Fable5-Thinking-Q8_0.gguf \
MODEL_BACKEND=gguf \
GGUF_N_CTX=4096 \
GGUF_N_GPU_LAYERS=35 \
./run.sh chat "解释资料里的世界杯商业化争议"
```

6. 用同一个 MiniCPM5 文件跑 Character + RAG。

```bash
MODEL_PATH=models/MiniCPM5-1B-Claude-Opus-Fable5-Thinking-Q8_0.gguf \
MODEL_BACKEND=gguf \
NPC_ID=lu_jiangxian \
CUTOFF=evt-010 \
GGUF_N_CTX=4096 \
GGUF_N_GPU_LAYERS=35 \
./run.sh character-rag-chat "用陆江仙的口吻解释世界杯商业化争议。"
```

7. 查看 Harness 注册模型和当前 artifact 状态。

```bash
./run.sh harness list
./run.sh harness status
```

8. 使用 Harness 管理的模型路径跑 RAG。

如果要通过 Harness 读取 `llama-3.2-1b-instruct`，文件必须存在于：

```text
models/llama-3.2-1b-instruct/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

确认是真实 GGUF 文件后运行：

```bash
MODEL_BACKEND=harness \
HARNESS_MODEL_ID=llama-3.2-1b-instruct \
GGUF_N_CTX=4096 \
GGUF_N_GPU_LAYERS=35 \
./run.sh chat "解释资料里的世界杯商业化争议"
```

9. 使用 Harness 管理的模型路径跑 Character + RAG。

```bash
MODEL_BACKEND=harness \
HARNESS_MODEL_ID=llama-3.2-1b-instruct \
NPC_ID=lu_jiangxian \
CUTOFF=evt-010 \
GGUF_N_CTX=4096 \
GGUF_N_GPU_LAYERS=35 \
./run.sh character-rag-chat "用陆江仙的口吻解释世界杯商业化争议。"
```

10. 如果想让当前 MiniCPM5 文件也走 Harness 管理，需要把它整理成 Harness 注册表对应的目录和文件名。当前注册表里 MiniCPM5 的模型 ID 和文件名是：

```text
models/minicpm5-0.9b/MiniCPM5-1B-Q4_K_M.gguf
```

可以复制或移动成本地 Harness 结构：

```bash
mkdir -p models/minicpm5-0.9b
cp models/MiniCPM5-1B-Claude-Opus-Fable5-Thinking-Q8_0.gguf \
  models/minicpm5-0.9b/MiniCPM5-1B-Q4_K_M.gguf
```

然后用 Harness 后端运行：

```bash
MODEL_BACKEND=harness \
HARNESS_MODEL_ID=minicpm5-0.9b \
GGUF_N_CTX=4096 \
GGUF_N_GPU_LAYERS=35 \
./run.sh chat "解释资料里的世界杯商业化争议"
```

注意：这一步只是让 Harness 能按注册表找到本地文件；文件本身是否完全适配该注册项，还取决于模型实际格式、chat template 和量化版本。

查看注册模型：

```bash
./run.sh harness list
```

查看当前选中模型状态：

```bash
./run.sh harness status
```

编译 Harness 角色 prompt：

```bash
./run.sh harness character-prompt \
  --character lu_jiangxian \
  --input "玄谙究竟是什么？" \
  --cutoff evt-010
```

运行 mock 角色对话：

```bash
./run.sh harness character-chat \
  --backend mock \
  --model llama-3.2-1b-instruct \
  --character xuan_an \
  --input "你为什么停止拼合七枚鉴身碎片？" \
  --cutoff evt-018 \
  --touch-demo-files
```

也可以直接使用 Python 入口：

```bash
python -m harness_logic list
python -m harness_logic character-pack validate
```

当前 Harness backend 仍是 mock backend；它能验证模型注册、文件状态、Prompt 编译、Session 和 Memory 链路，但不代表真实 LLM 推理已经接通。

### RAG 使用 Harness 管理的本地 GGUF

普通 RAG 和 Character + RAG 的检索逻辑保持不变；当设置 `MODEL_BACKEND=harness` 时，生成阶段会从 `harness_logic` 的模型注册表和运行目录中解析本地 GGUF 文件，然后复用本项目的 GGUF 后端执行真实推理。

先查看模型 ID 和 artifact 文件名：

```bash
./run.sh harness list
./run.sh harness status
```

把 GGUF 放到 Harness 期望的位置，例如：

```text
models/llama-3.2-1b-instruct/Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

也可以先看下载来源：

```bash
./run.sh harness select llama-3.2-1b-instruct
./run.sh harness download-plan
```

普通 RAG 接 Harness 管理的本地 GGUF：

```bash
MODEL_BACKEND=harness \
HARNESS_MODEL_ID=llama-3.2-1b-instruct \
GGUF_N_CTX=4096 \
GGUF_N_GPU_LAYERS=35 \
./run.sh chat "解释资料里的世界杯商业化争议"
```

Character + RAG 接 Harness 管理的本地 GGUF：

```bash
MODEL_BACKEND=harness \
HARNESS_MODEL_ID=llama-3.2-1b-instruct \
NPC_ID=lu_jiangxian \
CUTOFF=evt-010 \
./run.sh character-rag-chat "用陆江仙的口吻解释世界杯商业化争议。"
```

等价 Python 命令：

```bash
python -m simple_rag.cli chat \
  --index .rag_index/index.json \
  --query "解释资料里的世界杯商业化争议" \
  --model-backend harness \
  --harness-root harness_logic/data \
  --harness-models-root models \
  --harness-model-id llama-3.2-1b-instruct
```

如果对应 GGUF 文件不存在，命令会直接报出缺失的本地 artifact 路径；这时先按 `download-plan` 提示下载或手动放入该路径。

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

使用 GGUF / MiniCPM GGUF 时：

```bash
NPC_ID=lu_jiangxian \
CUTOFF=evt-010 \
MODEL_PATH=/path/to/minicpm-model.gguf \
MODEL_BACKEND=gguf \
GGUF_N_CTX=4096 \
GGUF_N_GPU_LAYERS=35 \
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
