import os

import torch
from datasets import load_dataset
import warnings
from transformers import AutoTokenizer, HfArgumentParser, TrainingArguments
from trl import (
    DataCollatorForCompletionOnlyLM,
    ModelConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
)

warnings.filterwarnings("ignore", message=".*Could not find response key.*")

def construct_device_map(use_deepspeed: bool):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    if use_deepspeed:
        device_map = None
    else:
        device_map = "auto"
        if ddp:
            device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    return device_map


def clean_content(res):
    return (" ".join([token for token in res.split(" ") if token])).strip()


def conversations_formatting_function(tokenizer: AutoTokenizer):
    def format_dataset(examples):
        output_texts = []
        for i in range(len(examples["user_prompt"])):
            msgs = [
                {
                    "role": "system",
                    "content": "You are a financial assistant. Always provide accurate and reliable information to the best of your abilities",
                },
                {"role": "user", "content": clean_content(examples["user_prompt"][i])},
                {"role": "assistant", "content": clean_content(examples["resp"][i])},
            ]
            output_texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
        return output_texts

    return format_dataset


parser = HfArgumentParser((TrainingArguments, ModelConfig))
training_args, model_config = parser.parse_args_into_dataclasses()
use_deepspeed = training_args.deepspeed not in [None, ""]
training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)


################
# Model & Tokenizer
################
torch_dtype = (
    model_config.torch_dtype
    if model_config.torch_dtype in ["auto", None]
    else getattr(torch, model_config.torch_dtype)
)
quantization_config = get_quantization_config(model_config)
model_kwargs = dict(
    revision=model_config.model_revision,
    trust_remote_code=model_config.trust_remote_code,
    attn_implementation=model_config.attn_implementation,
    torch_dtype=torch_dtype,
    use_cache=False if training_args.gradient_checkpointing else True,
    device_map=construct_device_map(use_deepspeed)
    if quantization_config is not None
    else None,
    quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_config.model_name_or_path, use_fast=True
)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"


################
# Dataset
################
raw_datasets = load_dataset("next-tat/tat-llm-instructions")
train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["validation"]


################
# Training
################
trainer = SFTTrainer(
    model=model_config.model_name_or_path,
    model_init_kwargs=model_kwargs,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=conversations_formatting_function(tokenizer),
    data_collator=DataCollatorForCompletionOnlyLM(
        "[/INST]", tokenizer=tokenizer, pad_to_multiple_of=8
    ),
    max_seq_length=4096,
    tokenizer=tokenizer,
    peft_config=get_peft_config(model_config),
)
trainer.train()
trainer.save_model(training_args.output_dir)
