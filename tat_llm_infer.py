import argparse
import json
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import List

import torch
from datasets import load_dataset
from peft.tuners.tuners_utils import BaseTunerLayer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def count_token(tokenizer, input_text, bos=False):
    extra = 1 if bos else 0
    return len(tokenizer.encode(input_text, add_special_tokens=False)) + extra


tokenizer = AutoTokenizer.from_pretrained("next-tat/tat-llm-7b-fft", use_fast=True)
# <s> inclusive
sorrounding = tokenizer.apply_chat_template(
    [
        {
            "role": "system",
            "content": "You are a financial assistant. Always provide accurate and reliable information to the best of your abilities",
        },
        {"role": "user", "content": ""},
    ],
    tokenize=False,
)
sorrounding = sorrounding.replace("<</SYS>>", "<</SYS>>\n\n")
sorrounding_len = count_token(tokenizer, sorrounding)
least_resp_len = 500
max_prompt_len = 4096 - least_resp_len - sorrounding_len


@dataclass
class Text:
    items: List[str]
    merged: str
    header: str = "### Text\n\n"
    bottom: str = "\n\n\n\n"

    @classmethod
    def create(cls, merged_text):
        text_items = [
            f"{item.strip()}\n" for item in merged_text.split("\n") if item.strip()
        ][1:]
        return cls(items=text_items, merged=merged_text)

    def __str__(self):
        return self.merged

    def set_items(self, new_items):
        self.items = new_items
        self.merged = f'{self.header}{"".join(i for i in new_items)}{self.bottom}'


def truncate_finqa_or_tatqa(d):
    merged_text = d["user_prompt"]

    intro_m = re.search(r".+(?=### Instruction)", merged_text, re.DOTALL)
    inst_m = re.search(r"### Instruction.+(?=### Table)", merged_text, re.DOTALL)
    table_m = re.search(r"### Table.+(?=### Text)", merged_text, re.DOTALL)
    text_m = re.search(r"### Text.+(?=### Question)", merged_text, re.DOTALL)
    ques_m = re.search(r"### Question.+$", merged_text, re.DOTALL)

    kwargs = {}
    for section_name, section_m in zip(
        ["introduction", "instruction", "table", "question"],
        [intro_m, inst_m, table_m, ques_m],
    ):
        section_text = merged_text[section_m.start() : section_m.end()]
        kwargs[section_name] = section_text

    text = merged_text[text_m.start() : text_m.end()]
    kwargs["text"] = Text.create(text)

    ori_content_len = count_token(
        tokenizer,
        f'{kwargs["introduction"]}{kwargs["instruction"]}{kwargs["table"]}{kwargs["text"]}{kwargs["question"]}',
    )

    if ori_content_len > max_prompt_len:
        exceed = ori_content_len - max_prompt_len
        target_text_len = count_token(tokenizer, str(kwargs["text"])) - exceed
        least_text_len = (
            count_token(tokenizer, kwargs["text"].header)
            + count_token(tokenizer, kwargs["text"].items[0])
            + count_token(tokenizer, kwargs["text"].bottom)
        )
        if target_text_len >= least_text_len:  # truncatable
            selected_text_items = [kwargs["text"].items[0]]
            selected_text_len = least_text_len
            for text_item in kwargs["text"].items[1:]:
                if target_text_len >= selected_text_len + count_token(
                    tokenizer, text_item
                ):
                    selected_text_items.append(text_item)
                    selected_text_len += count_token(tokenizer, text_item)
                else:
                    kwargs["text"].set_items(selected_text_items)
                    d[
                        "transformed_user_prompt"
                    ] = f'{kwargs["introduction"]}{kwargs["instruction"]}{kwargs["table"]}{kwargs["text"]}{kwargs["question"]}'
        else:
            d["transformed_user_prompt"] = None
    else:
        d[
            "transformed_user_prompt"
        ] = f'{kwargs["introduction"]}{kwargs["instruction"]}{kwargs["table"]}{kwargs["text"]}{kwargs["question"]}'

    return d


def truncate_tatdqa(d):
    merged_text = d["user_prompt"]
    intro_m = re.search(r".+(?=### Instruction)", merged_text, re.DOTALL)
    inst_m = re.search(r"### Instruction.+(?=### Document)", merged_text, re.DOTALL)
    document_m = re.search(r"### Document.+(?=### Question)", merged_text, re.DOTALL)
    ques_m = re.search(r"### Question.+$", merged_text, re.DOTALL)

    kwargs = {}
    for section_name, section_m in zip(
        ["introduction", "instruction", "document", "question"],
        [intro_m, inst_m, document_m, ques_m],
    ):
        section_text = merged_text[section_m.start() : section_m.end()]
        kwargs[section_name] = section_text

    ori_content_len = count_token(
        tokenizer,
        f'{kwargs["introduction"]}{kwargs["instruction"]}{kwargs["document"]}{kwargs["question"]}',
    )

    if ori_content_len > max_prompt_len:
        exceed = ori_content_len - max_prompt_len
        doc_input_ids = tokenizer.encode(kwargs["document"], add_special_tokens=False)
        doc_current_len = len(doc_input_ids)
        target_doc_len = doc_current_len - exceed
        doc_end_len = count_token(tokenizer, "\n" * 5)
        least_doc_len = int(doc_current_len * 0.5) + doc_end_len
        if target_doc_len >= least_doc_len:  # truncatable
            kwargs["document"] = (
                tokenizer.decode(doc_input_ids[: target_doc_len - doc_end_len])
                + "\n" * 5
            )
            print(d["id"])
            d[
                "transformed_user_prompt"
            ] = f'{kwargs["introduction"]}{kwargs["instruction"]}{kwargs["document"]}{kwargs["question"]}'
        else:
            d["transformed_user_prompt"] = None
    else:
        d[
            "transformed_user_prompt"
        ] = f'{kwargs["introduction"]}{kwargs["instruction"]}{kwargs["document"]}{kwargs["question"]}'
    return d


def infer(model, tokenizer, input_prompt, max_new_tokens):
    input_ids = tokenizer(input_prompt, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to("cuda")
    prompt_len = len(input_ids[0])

    generation_config = GenerationConfig(
        pad_token_id=0,
        num_beams=1,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
        )
        s = generation_output[0][prompt_len:]
        output = tokenizer.decode(s)
        return output


def conversations_formatting_function(tokenizer: AutoTokenizer):
    def format_dataset(example):
        msgs = [
            {
                "role": "system",
                "content": "You are a financial assistant. Always provide accurate and reliable information to the best of your abilities",
            },
            {"role": "user", "content": example["transformed_user_prompt"]},
        ]
        example["transformed_user_prompt"] = tokenizer.apply_chat_template(
            msgs, tokenize=False
        )
        return example

    return format_dataset


def main(model_name_or_path, data_type, output_path, lora_name_or_path):
    output_path = pathlib.Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if lora_name_or_path is not None:
        model.load_adapter(lora_name_or_path)
        for module in model.modules():
            if isinstance(module, BaseTunerLayer):
                module.merge()
    model.eval()

    if data_type == "all":
        data_list = ["finqa", "tatqa", "tatdqa"]
    else:
        data_list = [data_type]

    for one_type in data_list:
        print(f"Start to make inference on {one_type}")
        raw_datasets = load_dataset(
            "json",
            data_files={
                "test": f"./data/sft/stepwise/{one_type}/{one_type}_dataset_test.json"
            },
        )
        test_dataset = raw_datasets["test"]

        o = (
            test_dataset.map(
                truncate_finqa_or_tatqa if one_type != "tatdqa" else truncate_tatdqa
            )
            .filter(lambda example: example["transformed_user_prompt"] is not None)
            .map(conversations_formatting_function(tokenizer))
        )

        predictions = []
        for idx in tqdm(range(len(o))):
            sample = o[idx]
            input_prompt = sample["transformed_user_prompt"]
            output = infer(
                model,
                tokenizer,
                input_prompt,
                max_new_tokens=4096 - count_token(tokenizer, input_prompt),
            )
            predictions.append(
                {
                    "id": sample["id"],
                    "user_prompt": sample["user_prompt"],
                    "prediction": output,
                }
            )

        with open(output_path / f"{one_type}_pred.json", "w") as saver:
            json.dump(predictions, saver)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference of TAT-LLM")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=False,
        default="next-tat/tat-llm-7b-fft",
        help="Specify the model name or path. It can be either a HuggingFace repository or a local folder where the model is downloaded. If you want to try the LoRa weights, set this value to the original llama2 model path instead of the FFT tat-llm model. In this case, you also need to provide the --lora_name_or_path flag.",
    )
    parser.add_argument(
        "--lora_name_or_path",
        type=str,
        required=False,
        default=None,
        help="If provided, it will be combined with the --model_name_or_path to load tat-llm-lora models.",
    )
    parser.add_argument(
        "--test_data_type", type=str, required=True, help="finqa, tatqa, tatdqa or all"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="output folder to save inference results",
    )

    args = parser.parse_args()

    param_str = "\n".join(["%20s = %s" % (k, v) for k, v in sorted(vars(args).items())])
    print(
        "usage: %s\n%20s   %s\n%s\n%s\n"
        % (" ".join(sys.argv), "ARG", "VALUE", "_" * 50, param_str)
    )

    model_name_or_path = args.model_name_or_path
    lora_name_or_path = args.lora_name_or_path
    test_data_type = args.test_data_type
    output_path = args.output_path

    main(
        model_name_or_path=model_name_or_path,
        data_type=test_data_type,
        output_path=output_path,
        lora_name_or_path=lora_name_or_path,
    )
