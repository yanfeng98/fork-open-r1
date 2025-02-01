"""
# Full training
CUDA_VISIBLE_DEVICES=0 python sft.py \
    --model_name_or_path Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --dataset_name alpaca_gpt4 \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --max_steps -1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --max_seq_length 512 \
    --bf16 True \
    --logging_steps 25 \
    --output_dir Qwen2-0.5B-SFT \
    # --eval_strategy steps \
    # --eval_steps 100 \
    # --packing

# LoRA
CUDA_VISIBLE_DEVICES=0 python sft.py \
    --model_name_or_path Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --dataset_name alpaca_gpt4 \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --max_steps -1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --max_seq_length 512 \
    --bf16 True \
    --logging_steps 25 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir Qwen2-0.5B-SFT
    # --eval_strategy steps \
    # --eval_steps 100 \
    # --packing \

# deepspeed
accelerate launch --config_file=accelerate_configs/deepspeed_zero{1,2,3}.yaml --num_processes {NUM_GPUS} path_to_your_script.py --all_arguments_of_the_script

accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml --num_processes 8 sft.py \
    --model_name_or_path Qwen/Qwen2.5-Coder-7B-Instruct \
    --dataset_name alpaca_gpt4 \
    --learning_rate 2.0e-5 \
    --num_train_epochs 3 \
    --max_steps -1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --max_seq_length 512 \
    --bf16 True \
    --logging_steps 25 \
    --output_dir Qwen2.5-Coder-7B-Instruct-SFT \
    # --eval_strategy steps \
    # --eval_steps 100 \
    # --packing
"""

from datasets import load_dataset
from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    SFTConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

def formatting_prompts_func(examples):
    output_text = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input_text = examples["input"][i]
        response = examples["output"][i]

        if len(input_text) >= 1:
            text = f'''
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
### Instruction:
{instruction}
            
### Input:
{input_text}
            
### Response:
{response}
'''
        else:
            text = f'''
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
### Instruction:
{instruction}
            
### Response:
{response}
'''
        output_text.append(text.strip())

    return output_text


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    # dataset = load_dataset(script_args.dataset_name)
    dataset = load_dataset("json", name=script_args.dataset_name, data_files="./alpaca_gpt4_data_zh.json")
    print(dataset)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        # eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        formatting_func=formatting_prompts_func,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
