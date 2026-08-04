# Simple Harness RAG

这是一个本地优先的最小 RAG 项目，面向端侧或本机大模型实验：

- 递归读取 `data/` 里的资料，支持 `.txt`、`.md`、`.json`、`.jsonl`、`.csv`，安装可选依赖后支持 `.pdf` 和 `.docx`。
- 将资料切分为 chunk，建立本地向量索引。
- 检索相关 chunk，并把搜索结果组装进 prompt。
- 通过角色 JSON 控制大模型的身份、语气、行为规则和知识边界。
- 可接入任意 Hugging Face `transformers` 兼容的本地模型路径。

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

`--model` 指向本地 `transformers` 兼容模型目录。未指定 `--model` 时，命令只返回已组装的 prompt，方便检查检索和角色注入效果。

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
python -m unittest discover -s tests -v
```
