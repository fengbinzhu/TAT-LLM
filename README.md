TAT-LLM: A Specialized Language Model for Discrete Reasoning over Tabular and Textual Data
====================

We present TAT-LLM, a language model specialized in answering questions over financial Tabular and Textual Data.

| Model | Size | FINQA | TAT-QA | TAT-DQA |
| ---   | ---  | ---   | ---   | ---    |
| GPT-3.5-Turbo | - | 58.00 | 59.47 | 52.74 |
| GPT-4 | - | 63.91 | 71.92 | 64.46 |
| [TAT-LLM-7B-LORA](https://huggingface.co/next-tat/tat-llm-7b-lora) | 7B | 65.13 | 76.49 | 71.38 |
| [TAT-LLM-7B-FFT](https://huggingface.co/next-tat/tat-llm-7b-fft) | 7B | 69.75 | 76.91 | 72.64 |
| [TAT-LLM-13B-LORA](https://huggingface.co/next-tat/tat-llm-13b-lora) | 13B | 71.93 | 77.51 | 72.22 |
| [TAT-LLM-13B-FFT](https://huggingface.co/next-tat/tat-llm-13b-fft) | 13B | 72.97 | 78.41 | 73.18 |
| [TAT-LLM-70B-LORA](https://huggingface.co/next-tat/tat-llm-70b-lora) | 70B | **76.81** | 81.42 | 76.55 |
| [TAT-LLM-70B-FFT](https://huggingface.co/next-tat/tat-llm-70b-fft) | 70B | 76.11 | **82.20** | **76.97** |

Refer to our [TAT-LLM Paper](https://arxiv.org/abs/2401.13223) for more information.

### Requirements

To create an environment with [MiniConda](https://docs.conda.io/en/latest/miniconda.html) and activate it.

```bash
conda create -n tat-llm python=3.9
conda activate tat-llm
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirement.txt
```

### Dataset

The TAT-LLM model was trained using data from the folder `data/sft`, and its predictions are stored in the folder `data/prediction`. The training data can also be accessed on [🤗Hugging Face](https://huggingface.co/datasets/next-tat/tat-llm-instructions).

### Train

Parameter-Efficent finetuning Llama-2-7b on 1 X A100 80GB GPU

```bash
# Make sure you have access to llama2 so that lora layers can be created successfully

python tat_llm_train.py \
    --output_dir {output_folder_to_save_model_checkpoints_and_tensorboard_runs} \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --report_to tensorboard \
    --group_by_length \
    --learning_rate 3e-4 \
    --warmup_ratio 0.03 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 10 \
    --logging_steps 1 \
    --num_train_epochs 3 \
    --max_steps -1 \
    --gradient_checkpointing \
    --load_in_8bit \
    --use_peft \
    --lora_target_modules q_proj k_proj v_proj o_proj \
    --lora_alpha 16 \
    --log_level info \
    --evaluation_strategy steps \
    --save_strategy steps \
    --eval_steps 406 \
    --save_steps 406
```

<details> 
<summary>Parameter-Efficent finetuning Llama-2-13b/70b on 8 X A100 80GB GPU</summary>

```bash
# Make sure you have access to llama2 so that lora layers can be created successfully

torchrun --rdzv-backend c10d \
  --rdzv-endpoint localhost:7788 \
  --nnodes 1 \
  --nproc_per_node 8 \
  tat_llm_train.py \
  --output_dir {output_folder_to_save_model_checkpoints_and_tensorboard_runs} \
  --model_name_or_path "meta-llama/Llama-2-13b-hf" \
  --report_to tensorboard \
  --group_by_length \
  --learning_rate 3e-4 \
  --warmup_ratio 0.03 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 5 \
  --logging_steps 1 \
  --num_train_epochs 3 \
  --max_steps -1 \
  --gradient_checkpointing \
  --use_peft \
  --lora_target_modules q_proj k_proj v_proj o_proj \
  --lora_alpha 16 \
  --log_level info \
  --evaluation_strategy steps \
  --save_strategy steps \
  --eval_steps 406 \
  --save_steps 406 \
  --bf16 \
  --deepspeed ds_config_lora.json
```


</details>

<details>
<summary>Full-Parameter finetuning Llama2-7b/13b/70b on 8 X A100 80GB GPU</summary>


```bash
torchrun --rdzv-backend c10d \
  --rdzv-endpoint localhost:7788 \
  --nnodes 1 \
  --nproc_per_node 8 \
  tat_llm_train.py \
  --output_dir {output_folder_to_save_model_checkpoints_and_tensorboard_runs} \
  --model_name_or_path "meta-llama/Llama-2-13b-hf" \
  --report_to tensorboard \
  --group_by_length \
  --learning_rate 3e-6 \
  --warmup_ratio 0.03 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 5 \
  --logging_steps 1 \
  --num_train_epochs 3 \
  --max_steps -1 \
  --gradient_checkpointing \
  --bf16 \
  --deepspeed ds_config_fft.json
```
    
</details>


### Inference

If you want to make inference with FFT model e.g., `tat-llm-7b-fft`

```bash
# test_data_type should be one of ['finqa', 'tatqa', 'tatdqa', 'all']

python tat_llm_infer.py \
    --model_name_or_path "next-tat/tat-llm-7b-fft" \
    --test_data_type {test_data_type} \
    --output_path {output_folder_to_save_prediction_files}
```

If you want to make inference with LoRa model e.g., `tat-llm-7b-lora`


```bash
# Make sure you have access to llama2 so that lora weights can be merged successfully

python tat_llm_infer.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf"
    --lora_name_or_path "next-tat/tat-llm-7b-lora" \
    --test_data_type {test_data_type} \
    --output_path {output_folder_to_save_prediction_files}
```

Remarks:
- Depending on the GPU utilized and the packages installed, the final prediction results may exhibit slight variations
- Regards with gpu resources, tat-llm-7b/13b model requires 1 X A100 80GB GPU while tat-llm-70b model requires 2 X A100 80GB GPU


### Evaluation

To evaluate the prediction of the LLMs


```bash
python tat_llm_eval.py --dataset_name={dataset_name} --model_name={model_name} --model_type={model_type}
```

 `dataset_name`
- finqa
- tatqa
- tatdqa

`model_name`
- tat-llm-7b
- tat-llm-13b
- tat-llm-70b

`model_type`
- fft
- lora

### Citation
Please kindly consider citing our work if you are using this code repo in your work, thank you.
```
@misc{zhu2024tatllm,
      title={TAT-LLM: A Specialized Language Model for Discrete Reasoning over Tabular and Textual Data},
      author={Fengbin Zhu and Ziyang Liu and Fuli Feng and Chao Wang and Moxin Li and Tat-Seng Chua},
      year={2024},
      eprint={2401.13223},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
