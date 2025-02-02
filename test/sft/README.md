## SFT

To run SFT on a dataset distilled from DeepSeek-R1 with reasoning traces such as [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k), run:

### full

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch test/sft/sft.py --config test/sft/recipes/config_full.yaml
```

### lora

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch test/sft/sft.py --config test/sft/recipes/config_lora.yaml
```

### deepspeed

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file test/sft/accelerate_configs/deepspeed_zero3.yaml --num_processes 1 test/sft/sft.py --config test/sft/recipes/config_ds.yaml
```