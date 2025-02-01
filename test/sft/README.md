## SFT

To run SFT on a dataset distilled from DeepSeek-R1 with reasoning traces such as [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k), run:

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch test/sft/sft.py --config test/sft/recipes/config_full.yaml
```

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch test/sft/sft.py --config test/sft/recipes/config_lora.yaml
```

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py --config recipes/qwen/Qwen2.5-1.5B-Instruct/sft/config_full.yaml
```