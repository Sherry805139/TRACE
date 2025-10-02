#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
# 说明：本文件是训练主入口，整体流程为：解析参数 → 初始化分布式/DeepSpeed → 构建Tokenizer与模型 →
# 根据CL_method选择方法（如LoRA）→ 加载与缓存数据集 → 构建DataLoader → 按任务顺序训练并保存模型。
import sys
sys.dont_write_bytecode = True
# 不生成 .pyc 字节码文件，便于调试与避免多进程/容器环境下的写入冲突

import argparse
import os
import math
import sys
from tqdm import tqdm
# tqdm 用于显示进度条，便于观察训练进度

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    get_constant_schedule_with_warmup
)
# Transformers 提供模型/Tokenizer与优化器调度等通用工具

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.utils import safe_get_full_grad
# DeepSpeed 提供高效分布式训练与优化器（CPUAdam/FusedAdam）


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
# 将项目根目录加入 sys.path，确保可以导入仓库内的模块（utils/*、model/* 等）
from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from utils.model.model_utils import create_hf_model

# add flash attention
from utils.flash_attention.llama_flash_att import replace_llama_attn_with_flash_attn
from utils.flash_attention.bloom_flash_att import replace_bloom_attn_with_flash_attn

replace_llama_attn_with_flash_attn()
replace_bloom_attn_with_flash_attn()
# 将LLaMA/BLOOM注意力替换为flash-attn实现，可降低显存占用并提高速度（可选）

# my_peft中修改了lora相关的逻辑
from model.Replay.LFPT5 import getInitialPrompt
from model.Dynamic_network.PP import PP, convert_PP_model
from model.Dynamic_network.L2P import convert_L2P_model


from params import Method2Class, AllDatasetName
# 方法名到训练类的映射，以及默认数据集名称列表


# TODO, check support for OPT and llama


def parse_args():
    # 解析命令行参数，定义训练/数据/分布式/日志等配置
    def list_of_strings(arg):
        return arg.split(',')
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        type=str,
                        default='Dahoas/rm-static',
                        help='Path to the training dataset, a single data path.')
    parser.add_argument('--dataset_name',
                        type=list_of_strings,
                        default='all',
                        help='Dataset to be used.')
    # dataset_name 支持用逗号分隔多个任务；会与 data_path 组合成完整数据目录
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_prompt_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--max_ans_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=list_of_strings,
                        default=None,
                        help="Total number of training epochs to perform.")
    # 支持按任务指定多个轮数（逗号分隔），单任务可给单一数值
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="A seed for reproducible training.")
    # local_rank 一般表示当前进程在当前节点的编号，global_rank 表示当前进程在所有进程中的编号
    # local_rank 为 -1 时，表示不使用分布式训练。这个值一般由 pytorch/deepspeed 自动设置，用户不用管
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    # store_true 表示如果命令行中有这个参数，则 args.disable_dropout 为 True, 否则默认为 False
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    # ZeRO 优化等级（0/1/2/3）：更高等级可降低显存，但对环境要求更高
    
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step1_tensorboard")
    ## Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        help='Prints loss at each step.')
    # added by wangxiao
    parser.add_argument('--CL_method',
                default=None,
                help='continual learning method used')
    parser = deepspeed.add_config_arguments(parser)
    # 让 DeepSpeed 注入其配置相关参数（如 --deepspeed 等）
    # 解析后得到一个对象，属性访问如 args.learning_rate
    args = parser.parse_args()


    return args


def main():
    args = parse_args()
    # 主入口：初始化、构建模型/数据、选择方法并开始训练

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()
    # 分布式场景下初始化通信后端（DeepSpeed 封装）

    args.global_rank = torch.distributed.get_rank()
    # 记录当前进程的全局rank（单卡通常为0）

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="v2_sft")
    # set batch size
    # 设置每个 GPU 的微批次大小，也就是单个 GPU 一次处理的数据样本数量
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    # 这行计算的是全局有效批次大小，由三部分相乘得到：
    # args.per_device_train_batch_size：每个 GPU 的微批次大小
    # torch.distributed.get_world_size()：总进程数（通常等于总 GPU 数量）
    # args.gradient_accumulation_steps：梯度累积步数（将多个微批次的梯度累积后再更新参数）
    # 举例来说：如果每个 GPU 一次处理 2 个样本（per_device），使用 4 个 GPU（world_size=4），梯度累积 2 步，那么总批次大小就是 2×4×2=16。
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps
    # 根据总进程数与梯度累积计算全局batch size，供 DeepSpeed 做并行策略

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    # Barrier to make sure all process are ready to train
    torch.distributed.barrier()
    # 设定随机种子确保可复现；分布式同步一次，确保各rank就绪

    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    # default the LLM is decoder only model, so padding side is left
    assert tokenizer.padding_side == 'left'
    assert tokenizer.truncation_side == "left"
    # 假设解码器模型采用左填充与左截断，与 DataCollator 的实现保持一致

    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config=ds_config,
                            disable_dropout=args.disable_dropout
                            )
    # 创建HF模型，按需禁用dropout以增强稳定性
    
    # some CL methods can be realized by peft
    if args.CL_method == "LFPT5":
        from utils.my_peft import get_peft_model, PromptTuningInit, PromptTuningConfig, LoraConfig, TaskType

        initial_prompt = getInitialPrompt(tokenizer, prompt_token_number=300)
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=300,
            prompt_tuning_init_text=initial_prompt,
            tokenizer_name_or_path=args.model_name_or_path,
        )
        model = get_peft_model(model, peft_config)

    if args.CL_method == "O-LoRA":
        from utils.my_peft import get_peft_model, PromptTuningInit, PromptTuningConfig, LoraConfig, TaskType

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        for name, param in model.named_parameters():
            if name.find("loranew_") != -1:
                param.requires_grad = True
            elif name.find("lora_") != -1:
                param.requires_grad = False
                
    if args.CL_method == "OGD":
        from peft import get_peft_model, LoraConfig, TaskType
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        for name, param in model.named_parameters():
            if name.find("lora") != -1:
                param.requires_grad = True

    if args.CL_method == "lora":
        from peft import get_peft_model, LoraConfig, TaskType
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        for name, param in model.named_parameters():
            if name.find("lora") != -1:
                param.requires_grad = True
        # 经典LoRA：仅训练lora_*相关低秩参数，其余权重冻结
    
    train_task_list = {}
    eval_task_list = {}
    test_task_list = {}


    if args.dataset_name[0] == "all":
        Datasets = AllDatasetName
    else:
        Datasets = args.dataset_name
    for dataset in Datasets:
        # 遍历每个任务/数据集，依次创建DataLoader，并在后续按顺序训练
        dataset_path = os.path.join(args.data_path,dataset)
        # Prepare the data
        train_dataset, eval_dataset, test_dataset = create_prompt_dataset(
            args.local_rank,
            dataset_path,
            args.data_output_path,
            args.seed
        )

        # DataLoaders creation:
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
            eval_sampler = SequentialSampler(eval_dataset)
            test_sampler = SequentialSampler(test_dataset)

        else:
            train_sampler = DistributedSampler(train_dataset)
            eval_sampler = DistributedSampler(eval_dataset)
            test_sampler = DistributedSampler(test_dataset)
        # 单机场景：随机/顺序采样；分布式场景：使用分布式采样器确保样本划分不重复

        # 数据预处理的 "打包器"，负责将零散的样本整理成模型可接受的批次数据，主要处理：
        # 对文本进行 padding（补全），使同批次样本长度一致
        # 可能包含截断过长文本的操作
        # 处理标签（label）与输入的对齐等

        # 训练模式下构造labels并做-100掩码
        data_collator = DataCollator(
            tokenizer,
            padding="longest",
            max_prompt_len=args.max_prompt_len,
            max_ans_len=args.max_ans_len,
            pad_to_multiple_of=8,
            inference=False
        )
        # 推理模式下不构造labels，仅输入prompt
        inf_data_collator = DataCollator(
            tokenizer,
            model=model,
            padding="longest",
            max_prompt_len=args.max_prompt_len,
            max_ans_len=args.max_ans_len,
            pad_to_multiple_of=8,
            inference=True
        )
                

        train_dataloader = DataLoader(train_dataset,
                                    collate_fn=data_collator,
                                    sampler=train_sampler,
                                    batch_size=args.per_device_train_batch_size)
        eval_dataloader = DataLoader(eval_dataset,
                                    collate_fn=data_collator,
                                    sampler=eval_sampler,
                                    batch_size=args.per_device_eval_batch_size)
        test_dataloader = DataLoader(test_dataset,
                            collate_fn=inf_data_collator,
                            sampler=test_sampler,
                            batch_size=args.per_device_eval_batch_size)
        train_task_list[dataset] = train_dataloader
        eval_task_list[dataset] = eval_dataloader
        test_task_list[dataset] = test_dataloader

        # dataset对象包含input_ids, attention_mask, labels, sources, gts等字段


    def evaluation(model, eval_dataloader):
        # 验证集困惑度评估（当前默认注释掉调用）
        model.eval()
        losses = 0
        for step, batch in enumerate(eval_dataloader):
            # implementation, batch = {k: v.to(device) for k, v in batch.items()}
            del batch['sources']
            batch = to_device(batch, device)
            with torch.no_grad():
                # TODO, check output
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
        losses = losses / (step + 1)
        try:
            perplexity = torch.exp(losses)
        except OverflowError:
            perplexity = float("inf")
        try:
            perplexity = get_all_reduce_mean(perplexity).item()
        except:
            pass
        return perplexity

    def get_optimizer(model):
        # 构建优化器与学习率调度器（DeepSpeed会接管backward/step）
        # Split weights in two groups, one with weight decay and the other not.
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            model, args.weight_decay)

        AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
        optimizer = AdamOptimizer(optimizer_grouped_parameters,
                                lr=args.learning_rate,
                                betas=(0.9, 0.95))
        
        total_train_dataloader_len = sum(len(train_task_list[task]) for task in list(train_task_list.keys()))
        num_update_steps_per_epoch = math.ceil(
            total_train_dataloader_len / args.gradient_accumulation_steps)
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps
        )
        
        return optimizer, lr_scheduler
    
    if args.CL_method=="PP" or args.CL_method=="L2P":
        # PP/L2P等方法需要获取嵌入层形状，用于注入可学习提示
        if "opt" in args.model_name_or_path.lower():
            embed_tokens_shape = model.model.decoder.embed_tokens.weight.shape
            embed_tokens = model.model.decoder.embed_tokens
            
            args.embed_tokens_dim = embed_tokens_shape[1]
            args.embed_tokens_length = embed_tokens_shape[0]
            args.embed_tokens = embed_tokens
        elif "llama" in args.model_name_or_path.lower():
            embed_tokens_shape = model.model.embed_tokens.weight.shape
            embed_tokens = model.model.embed_tokens
            
            args.embed_tokens_dim = embed_tokens_shape[1]
            args.embed_tokens_length = embed_tokens_shape[0]
            args.embed_tokens = embed_tokens
            
        if args.CL_method=="PP":
            args.prefix_len = 20
            args.task_length = len(train_task_list)
            model = convert_PP_model(model, args)
            
        elif args.CL_method=="L2P":
            args.pool_size = 10
            args.prompt_length = 5
            args.prompt_init = "uniform"
            model = convert_L2P_model(model, args)
            for name, params in model.named_parameters():
                if "prompt" not in name:
                    params.requires_grad=False
        # 除提示参数外其余梯度关闭，仅优化提示参数
                    
    optimizer, lr_scheduler = get_optimizer(model)
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)
    # 交由DeepSpeed托管模型/优化器/调度器，返回Engine封装（提供engine.backward/engine.step）

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    # 启用梯度检查点以节省显存（代价是更多前向计算）

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    # print_rank_0(
    #     f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
    #     args.global_rank)
    # perplexity = evaluation(model, eval_dataloader)
    # print_rank_0(f"ppl: {perplexity}", args.global_rank)

    # Initialize the global progress bar

    if args.CL_method in Method2Class.keys():
        CL_Trainer = Method2Class[args.CL_method](model, tokenizer, optimizer, train_task_list, eval_task_list, test_task_list, args)
        CL_Trainer.train_continual()
    # 根据映射构造具体训练器（如 lora → model.lora.lora），执行按任务顺序的训练与保存


if __name__ == "__main__":
    main()
    # 脚本入口
