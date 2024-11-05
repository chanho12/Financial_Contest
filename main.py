import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import deepspeed
from transformers import ( 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    set_seed, 
    BitsAndBytesConfig, 
    AutoConfig, 
    SchedulerType,
    get_scheduler
    )

from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

import wandb
import random
import numpy as np
import pandas as pd
import argparse
import os
import math
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    PeftConfig,
    prepare_model_for_kbit_training,
)
    
def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        default=['Dahoas/rm-static'],
                        help='Path to the training dataset. Accepted format:'
                        '1) a single data path, 2) multiple datasets in the'
                        'form: dataset1-path dataset2-path ...')
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
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.1,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
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
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')

    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",   
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--train_data_path', type=str, required=True, help="training path")
    parser.add_argument('--valid_data_path', type=str, required=True, help="validation path")
    parser.add_argument('--num_train_samples', type=int, required=True, help="number of rows in training datas.")
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser = deepspeed.add_config_arguments(parser)
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    # Validate settings
    if args.gradient_checkpointing and args.lora_dim > 0:
        assert (
            not args.only_optimize_lora
        ), "--gradient_checkpointing and --only_optimizer_lora cannot be enabled at the same time."

    return args
def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    lora_lr=5e-4,
    no_decay_name_list=[
        "bias",
        "layer_norm.weight",
        "layernorm.weight",
        "norm.weight",
        "ln_f.weight",
    ],
    lora_name_list=["lora_right_weight", "lora_left_weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                    not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad
                    and not any(nd in n.lower() for nd in lora_name_list)
                )
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                    not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad
                    and any(nd in n.lower() for nd in lora_name_list)
                )
            ],
            "weight_decay": weight_decay,
            "lr": lora_lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                    any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad
                )
            ],
            "weight_decay": 0.0,
        },
    ]

    non_empty_groups = []
    for group in optimizer_grouped_parameters:
        if group["params"]:
            non_empty_groups.append(group)
    return non_empty_groups


# 커스텀 데이터셋 정의
class CustomDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        
        # 효율적인 배치 토크나이즈를 위해 배치 크기를 설정
        batch_size = 1000  # 필요에 따라 조정
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            # 토크나이즈 (패딩 및 트렁케이션 없이)
            encodings = self.tokenizer(batch_texts, 
                                       padding=False, 
                                       truncation=False, 
                                       return_length=True)
            # 길이가 max_length 이상인 텍스트만 필터링
            for text, length in zip(batch_texts, encodings['length']):
                if length <= max_length:  # 최대 길이 초과 샘플 필터링
                    self.texts.append(text)
        
        print(f"Filtered dataset size: {len(self.texts)}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def create_model(model_name_or_path):
    model_config = AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    model_config.dropout = 0.0

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, config=model_config, trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, fast_tokenizer=True)

    return model, tokenizer

    
def get_lora_model(model_path):
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    # `pad_token` 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True  # 필요한 경우 추가
    )

    model.resize_token_embeddings(len(tokenizer))

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer

def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}

def evaluation(model, eval_dataloader, device):
    model.eval()
    total_loss = 0
    total_steps = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            total_steps += 1
    avg_loss = total_loss / total_steps
    perplexity = torch.exp(torch.tensor(avg_loss))
    return perplexity.item(), avg_loss

def main():

    args = parse_args()

    model_path = args.model_name_or_path

    ############################################# 분산 학습 작업

    ### single node
    if args.local_rank == -1:
            device = torch.device("cuda")
    else: ## multiple node
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()


    GLOBAL_BATCH_SIZE = 32
    MICRO_BATCH_SIZE = 4

    ds_config = {
        "train_batch_size": args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "steps_per_print": 10,
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
            # Offload 설정 추가
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            } if args.offload else None,
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            } if args.offload else None
        },
        "bf16": {"enabled": True},
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "optimizer": {
            "type": "DeepSpeedCPUAdam",
            "params": {
                "lr": args.learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-08,
                "weight_decay": args.weight_decay
            }
        },
        "scheduler": {
            "type": args.lr_scheduler_type.value,
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.learning_rate,
                "warmup_num_steps": args.num_warmup_steps,
            }
        }
    }

    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps
    
    torch.distributed.barrier() 
    
    ######################################## 분산 작업 끝.
    

    model, tokenizer = get_lora_model(model_path)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_disable()


    prompt = "You should answer the following question in a multi-hop way: "

    # 훈련 데이터 로드 및 프롬프트 적용
    # 훈련 데이터 로드 및 프롬프트 적용
    train_data = pd.read_csv(args.train_data_path)
    train_data['text'] = prompt + train_data['question'] + tokenizer.eos_token + train_data['response']
    print(f"Train set: {len(train_data)} samples")

    # 검증 데이터 로드 및 프롬프트 적용
    val_data = pd.read_csv(args.valid_data_path)
    val_data['text'] = prompt + val_data['question'] + tokenizer.eos_token + val_data['response']
    print(f"Validation set: {len(val_data)} samples")
    train_texts = [str(text) for text in train_data['text'].tolist()]
    train_texts = [str(text) for text in train_data['text'].tolist()]

    train_dataset = CustomDataset(train_texts, tokenizer, max_length=args.max_seq_len)
    val_dataset = CustomDataset(train_texts, tokenizer, max_length=args.max_seq_len)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, sampler=train_sampler)
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, batch_size=args.per_device_train_batch_size, sampler=val_sampler)
    
    num_update_steps_per_epoch = math.ceil(
        (len(train_data) * 1.125) / torch.distributed.get_world_size() / args.gradient_accumulation_steps)

    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay)
    
    optimizer = DeepSpeedCPUAdam(optimizer_grouped_parameters,
                                lr=args.learning_rate,
                                betas=(0.9, 0.95))

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    # 모델을 GPU로 이동
    model, optimizer, _, lr_scheduler = deepspeed.initialize(args=args,
                                                         model=model,
                                                         model_parameters=model.parameters(),
                                                         lr_scheduler=lr_scheduler,
                                                         optimizer=optimizer,
                                                         config=ds_config)

    wandb.init(project="Shared Memory")

    global_step = 0
    total_steps = 0
    for epoch in range(args.num_train_epochs):
        print(f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}")
        model.train()
        mean_loss = 0
        train_sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            if args.global_rank == 0:
                wandb.log({'step': loss.item()})  # 수정: loss를 로그할 때 .item()을 사용
            model.backward(loss)
            model.step()
            mean_loss += loss.item()
            total_steps = global_step + step

            # if total_steps > 0 and total_steps % args.eval_interval == 0:
            #     print('Evaluating model ...')
            #     print(f"***** Evaluating perplexity, Epoch {epoch+1}/{args.num_train_epochs} *****")
            #     perplexity, dev_loss = evaluation(model, val_dataloader, device)
            #     print(f"Perplexity: {perplexity}, Loss: {dev_loss}")
            #     if args.global_rank == 0:
            #         wandb.log({'perplexity':perplexity, 'loss' : dev_loss })

        #     if total_steps > 0 and total_steps % args.save_interval == 0 and args.output_dir is not None:
        #         print(f'Saving model ... at step {total_steps}')
        #         save_path = os.path.join(args.output_dir, f'checkpoint-{total_steps}')
        #         model.save_pretrained(save_path)
        # global_step = total_steps

    # 최종 모델 저장
    if args.output_dir is not None:
        print('Saving final model...')
        if hasattr(model, 'module'):
            model_to_save = model.module
        else:
            model_to_save = model

        # LoRA 가중치를 병합합니다.
        model_to_save = model_to_save.merge_and_unload()

        # 모델과 토크나이저를 저장합니다.
        if args.global_rank == 0:
            print('Saving final model in PyTorch bin format...')
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)



if __name__ == '__main__':
    main()