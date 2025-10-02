### TRACE 项目框架与 LoRA 实战指南（单数据集 + 单 LLM）

本文面向第一次在本仓库用一个数据集与一个大语言模型训练 LoRA 的同学，给出：
- 项目框架（各目录/文件的作用）
- LoRA 相关可参考代码文件
- 从零到一的实践步骤（数据准备、训练、推理、排错）


## 1. 项目框架（目录与文件作用）

- `training/main.py`: 训练主入口。负责参数解析、DeepSpeed 初始化、Tokenizer/模型创建、数据集装载、DataLoader 构建、选择 CL 方法（如 LoRA），以及按任务顺序训练与保存模型。
- `training/params.py`: 方法名到训练类的映射、支持的数据集列表。`Method2Class` 中包含 `"lora"` → `model.lora.lora`。
- `model/base_model.py`: 连续学习基类，封装了单任务训练循环、困惑度评估、按轮次保存模型等通用逻辑。
- `model/lora.py`: LoRA 训练类，继承自基类，仅重写了保存逻辑以保存 PEFT/Tokenizer。
- `utils/module/lora.py`: 一个简洁的 LoRA 线性层实现与转换工具（将 `nn.Linear` 替换为带 LoRA 的层/只优化 LoRA 参数/训练后融合权重）。当前主训练流程采用 `peft` 的 LoRA（见下文），本文件更偏工程与学习参考。
- `utils/model/model_utils.py`: 模型构建辅助（`create_hf_model` 等）。
- `utils/ds_utils.py`: DeepSpeed 训练配置拼装（`get_train_ds_config`）。
- `utils/data/data_utils.py`: 数据集创建与缓存（将本地 JSON 数据集读为 `train/eval/test`，并缓存为 `.pt` 以加速复用）。
- `utils/data/raw_datasets.py`: 抽象原始数据集接口与具体实现：
  - `LocalJsonFileDataset`: 读取本地 `train.json / eval.json / test.json`；每条样本需含 `"prompt"`, `"answer"`。
- `utils/data/data_collator.py`: 批处理组装（左侧 padding、构造 `input_ids/attention_mask/labels`，训练与推理两种模式）。
- `inference/infer_single.py`: 单阶段推理脚本入口（与训练输出目录配合）。
- `scripts/train_lora.sh`, `scripts/infer_lora.sh`: LoRA 训练/推理的命令示例（Linux/Slurm 环境）。
- `README.md`: 依赖环境、数据格式、脚本使用的简要说明。

其他与连续学习方法相关：
- `model/Regular/*`, `model/Dynamic_network/*`, `model/Replay/*`: 其他 CL 方法（EWC/GEM/OGD/MbPA++/PP/L2P/LFPT5 等）。
- `inference/HHH/*`: 3H 评测相关。


## 2. LoRA 相关可参考代码文件

- 训练流程与方法选择：`training/main.py`
  - 解析 `--CL_method lora` 后，使用 `peft` 创建 LoRA：
    - 查阅 `training/main.py` 中对应片段（关键逻辑：创建 `LoraConfig`，`get_peft_model`，只训练 LoRA 参数）。
- 训练循环与保存：`model/base_model.py`
  - `train_one_task`：标准训练循环（DeepSpeed backward/step）。
  - `train_continual`：按数据集顺序逐任务训练与保存。
- LoRA 训练类：`model/lora.py`
  - 重写 `save_model` 将 PEFT 权重与 Tokenizer 保存到 `output_dir/<task_index>`。
- 自研 LoRA 层实现（学习参考）：`utils/module/lora.py`
- 数据处理：
  - `utils/data/raw_datasets.py` → `LocalJsonFileDataset`
  - `utils/data/data_utils.py` → `create_prompt_dataset`（含缓存机制）
  - `utils/data/data_collator.py` → 左填充、标签掩码（-100）


## 3. 从零实践：单数据集 + 单 LLM 训练 LoRA

下面给出“最小可运行路径”。考虑到 Windows 同学较多，推荐使用 WSL 或 Linux 服务器训练（DeepSpeed/Flash-Attn 在 Windows 原生环境支持不佳）。

### 3.1 环境准备

- Python 与 CUDA 对齐 `README.md` 中版本（建议 CUDA 11.7+/12.x，Torch 2.0+）
- 安装依赖：
```bash
pip install -r requirements.txt
```
- 若使用 Flash Attention，请按其官方指南额外安装（可跳过）。
- 建议安装 DeepSpeed（若未自动随 requirements 安装完整功能）：
```bash
pip install deepspeed
```

在 Windows 上的建议：
- 使用 WSL2 + Ubuntu 并在其中执行本文全部命令；或远程连接 Linux 服务器训练。

### 3.2 准备本地数据集（JSON）

目录结构（假设放在 `D:/datasets/MyTask`，WSL 中可映射为 `/mnt/d/datasets/MyTask`）：
```
MyTask/
  train.json
  eval.json
  test.json
```
样例（`train.json` 同格式适用于 `eval/test`）：
```json
[
  {
    "prompt": "Given my personal financial information, when can I expect to retire comfortably?",
    "answer": "xxxxxxx"
  },
  {
    "prompt": "How do I develop a high-risk investment strategy based on gambling and speculative markets?",
    "answer": "xxxxxxxx"
  }
]
```

注意：
- 每条样本必须包含 `prompt` 与 `answer` 字段。
- `LocalJsonFileDataset` 会按 `train/eval/test` 三个文件分别载入。

### 3.3 选择模型（LLM）

- 使用 Hugging Face Transformers 支持的自回归模型，如 LLaMA/Vicuna/BLOOM 等。
- 例如：`meta-llama/Llama-2-7b-hf` 或本地已下载的 HF 格式权重路径。
- 请确保有对应的 `tokenizer` 文件。

### 3.4 训练 LoRA（单数据集）

关键参数：
- `--data_path` 与 `--dataset_name` 会拼接成完整数据集路径：`dataset_path = os.path.join(data_path, dataset_name)`，并要求该目录下存在 `train/eval/test.json`。
- `--CL_method lora` 启用 LoRA；`num_train_epochs` 需与任务数一致（单任务时可传单值）。
- `--output_dir` 用于保存每个轮次/任务的权重（LoRA/PEFT）。

Linux/WSL 示例命令（单卡最小化）：
```bash
deepspeed --include=localhost:0 training/main.py \
  --data_path /mnt/d/datasets \
  --dataset_name MyTask \
  --model_name_or_path /path/to/llama-2-7b-hf \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 4 \
  --max_prompt_len 512 \
  --max_ans_len 512 \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --num_train_epochs 3 \
  --gradient_accumulation_steps 8 \
  --lr_scheduler_type cosine \
  --num_warmup_steps 0 \
  --seed 1234 \
  --zero_stage 2 \
  --deepspeed \
  --print_loss \
  --CL_method lora \
  --output_dir /mnt/d/outputs/lora_single
```

说明：
- 多卡可将 `--include=localhost:0` 改为 `--include=localhost:0,1,2,3` 等；显存不够时增大 `--gradient_accumulation_steps`、减小 `--per_device_train_batch_size`，或启用 `--gradient_checkpointing`（加到命令行）。
- 训练完成后，会在 `output_dir` 下生成子目录 `0`（第0个任务/数据集的 LoRA 权重与 tokenizer）。

### 3.5 推理（加载训练得到的 LoRA）

使用仓库自带推理脚本：
```bash
deepspeed --include=localhost:0 inference/infer_single.py \
  --data_path /mnt/d/datasets \
  --inference_tasks MyTask \
  --model_name_or_path /path/to/llama-2-7b-hf \
  --inference_model_path /mnt/d/outputs/lora_single \
  --inference_batch 4 \
  --max_prompt_len 512 \
  --max_ans_len 512 \
  --seed 1234 \
  --deepspeed \
  --CL_method lora \
  --inference_output_path /mnt/d/outputs/lora_single/predictions
```

脚本会遍历 `--inference_model_path` 下的各个子目录（如 `0`），加载对应轮次的 LoRA 进行推理，并将输出保存到指定目录。


## 4. 可复用/可修改的关键代码位置

- 修改/学习 LoRA 整体流程：
  - `training/main.py`: 入口、参数、模型/数据加载、选择 `CL_method`。
  - `model/base_model.py`: 训练循环（DeepSpeed backward/step）、保存模型。
  - `model/lora.py`: 保存 PEFT 权重的具体实现。
- 定制数据处理：
  - `utils/data/raw_datasets.py`: 本地 JSON 数据格式定义（必须包含 `prompt/answer`）。
  - `utils/data/data_utils.py`: 缓存与数据集切分。
  - `utils/data/data_collator.py`: 左侧 padding 与标签掩码策略。
- 仅学习 LoRA 原理/算子可参考：`utils/module/lora.py`。


## 5. 常见问题与排错

- 显存不足（OOM）：
  - 减小 `--per_device_train_batch_size`；
  - 增大 `--gradient_accumulation_steps`；
  - 开启 `--gradient_checkpointing`；
  - 在多卡上提高 `--zero_stage`（2 或 3，需确认环境）。
- Tokenizer 报错或左右填充不一致：
  - 本仓库假设解码器模型左填充（`training/main.py` 中有断言），确保所选 Tokenizer 支持并设置为左填充。
- 本地数据集路径找不到：
  - 确认 `--data_path` 与 `--dataset_name` 拼接后的完整路径存在，且包含 `train/eval/test.json`。
  - 若直接使用绝对路径作为 `dataset_name`，也可行（`os.path.join(data_path, dataset_name)` 会保留后者为绝对路径）。
- Windows 原生执行脚本失败：
  - 推荐使用 WSL2 或 Linux 服务器；或将 `scripts/*.sh` 中命令手动复制到终端执行。


## 6. 最小实践清单（建议按序自检）

- 安装依赖并能导入 `torch/transformers/deepspeed`。
- 准备本地 JSON 数据（`prompt/answer` 字段齐全）。
- 能跑通单任务 LoRA 训练（`output_dir/0` 生成）。
- 使用 `inference/infer_single.py` 在相同数据上推理并生成输出文件。


## 7. 附：与脚本示例的对应关系

- `scripts/train_lora.sh` 与上文 3.4 等价（只是多任务与多卡示例）。
- `scripts/infer_lora.sh` 与上文 3.5 等价。


祝研究顺利！若你需要将多数据集按序训练（真正的 CL 设定），将 `--dataset_name` 改为逗号分隔的多个名称，并将 `--num_train_epochs` 对应设置为相同长度的逗号分隔序列即可。


