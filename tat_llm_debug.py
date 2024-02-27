import argparse


import deepspeed
from datasets import load_dataset
from transformers import LlamaForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


def get_optimizer_grouped_parameters(
    model,
    weight_decay,
    lora_lr=5e-4,
    no_decay_name_list=[
        "bias", "layer_norm.weight", "layernorm.weight", "norm.weight",
        "ln_f.weight"
    ],
    lora_name_list=["lora_right_weight", "lora_left_weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n.lower()
                                                    for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n.lower() for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n.lower()
                                                for nd in lora_name_list))
            ],
            "weight_decay":
            weight_decay,
            "lr":
            lora_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n.lower()
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]

    non_empty_groups = []
    for group in optimizer_grouped_parameters:
        if group["params"]:
            non_empty_groups.append(group)
    return non_empty_groups


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy", type=str, default="dummy",
                        help="dummy")

    return parser


arg_parser = get_argument_parser()
arg_parser = deepspeed.add_config_arguments(arg_parser)
args = arg_parser.parse_args()
deepspeed.init_distributed()


from transformers.deepspeed import HfDeepSpeedConfig
from deepspeed.ops.adam import FusedAdam
import json
import pathlib

ds_config = json.loads(pathlib.Path('./ds_config_debug.json').read_text())

dschf = HfDeepSpeedConfig(ds_config)

model = LlamaForCausalLM.from_pretrained('/data/.nn_framework_data/projects/da/models/llm/base/hf_llama2_7b', load_in_8bit=True)

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['q_proj']
)

model = get_peft_model(model, peft_config)


optimizer_grouped_parameters = get_optimizer_grouped_parameters(
    model, 0, 1e-3
)
optimizer = FusedAdam(optimizer_grouped_parameters, lr=1e-3, betas=(0.9, 0.95))

model_engine, optimizer, _, _ = deepspeed.initialize(args=args,
                                             model=model,
                                             optimizer=optimizer,
                                             config=ds_config,
                                             dist_init_required=True)

tokenizer = AutoTokenizer.from_pretrained('/data/.nn_framework_data/projects/da/models/llm/base/hf_llama2_7b')
batch = tokenizer("I ate an apple", return_tensors="pt").to('cuda')
batch['labels'] = batch['input_ids']

model_engine.train()

optimizer.quantize_nontrainable_params()

#forward() method
loss = model_engine(**batch).loss

#runs backpropagation
model_engine.backward(loss)

#weight update
model_engine.step()

