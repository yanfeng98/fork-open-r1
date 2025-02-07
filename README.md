# Open R1

**Table of Contents**  
- [Open R1](#open-r1)
  - [Overview](#overview)
    - [Plan of attack](#plan-of-attack)
  - [Installation](#installation)
  - [Training models](#training-models)
    - [SFT](#sft)
    - [GRPO](#grpo)
  - [Evaluating models](#evaluating-models)
  - [Reproducing Deepseek's evaluation results](#reproducing-deepseeks-evaluation-results)
    - [MATH-500](#math-500)
    - [GPQA Diamond](#gpqa-diamond)
  - [Data generation](#data-generation)
    - [Generate data from a smol distilled R1 model](#generate-data-from-a-smol-distilled-r1-model)
    - [Generate data from DeepSeek-R1](#generate-data-from-deepseek-r1)
  - [Contributing](#contributing)

## Overview

- `src/open_r1`: contains the scripts to train and evaluate models as well as generate synthetic data:
    - `grpo.py`: trains a model with GRPO on a given dataset.
    - `sft.py`: performs a simple SFT of a model on a dataset.
    - `evaluate.py`: evaluates a model on the R1 benchmarks.
    - `generate.py`: generates synthetic data from a model using [Distilabel](https://github.com/argilla-io/distilabel).
- `Makefile`: contains easy-to-run commands for each step in the R1 pipeline leveraging the scripts above.

### Plan of attack

We will use the DeepSeek-R1 [tech report](https://github.com/deepseek-ai/DeepSeek-R1) as a guide, which can roughly be broken down into three main steps:

* Step 1: replicate the R1-Distill models by distilling a high-quality corpus from DeepSeek-R1.
* Step 2: replicate the pure RL pipeline that DeepSeek used to create R1-Zero. This will likely involve curating new, large-scale datasets for math, reasoning, and code.
* Step 3: show we can go from base model to RL-tuned via multi-stage training.

<center>
    <img src="assets/plan-of-attack.png" width="500">
</center>


## Installation

**Note: Libraries rely on CUDA 12.4. Double check your system if you get segmentation faults.**

To run the code in this project, first, create a Python virtual environment using e.g. `uv`.
To install `uv`, follow the [UV Installation Guide](https://docs.astral.sh/uv/getting-started/installation/).


```shell
uv venv openr1 --python 3.11 && source openr1/bin/activate && uv pip install --upgrade pip --link-mode=copy
```

Next, install vLLM:

```shell
uv pip install vllm==0.7.1 --link-mode=copy
```

This will also install PyTorch `v2.5.1` and it is **very important** to use this version since the vLLM binaries are compiled for it. You can then install the remaining dependencies for your specific use case via `pip install -e .[LIST OF MODES]`. For most contributors, we recommend:

```shell
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]" --link-mode=copy
```

Next, log into your Hugging Face and Weights and Biases accounts as follows:

```shell
huggingface-cli login
wandb login
```

Finally, check whether your system has Git LFS installed so that you can load and push models/datasets to the Hugging Face Hub:

```shell
git-lfs --version
```

If it isn't installed, run:

```shell
sudo apt-get install git-lfs
```

## Training models

We support training models with either DDP or DeepSpeed (ZeRO-2 and ZeRO-3). To switch between methods, simply change the path to the `accelerate` YAML config in `configs`.

### SFT

To run SFT on a dataset distilled from DeepSeek-R1 with reasoning traces such as [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k), run:

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py --config recipes/qwen/Qwen2.5-1.5B-Instruct/sft/config_full.yaml
```

To launch a Slurm job, run:

```shell
sbatch --output=/path/to/logs/%x-%j.out --err=/path/to/logs/%x-%j.err slurm/sft.slurm {model} {dataset} {accelerator}
```

Here `{model}` and `{dataset}` refer to the model and dataset IDs on the Hugging Face Hub, while `{accelerator}` refers to the choice of an 🤗 Accelerate config file in configs. 

### GRPO

To train via the GRPO trainer, we use one GPU to run vLLM for faster generation and the remaining GPUs for training. For example, one a node with 8 GPUs, use the `recipes/accelerate_configs/zero3.yaml` config and then overwrite `num_processes` to run on 7 devices:

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml --num_processes=7 src/open_r1/grpo.py --config recipes/qwen/Qwen2.5-1.5B-Instruct/grpo/confg_full.yaml
```

We provide a minimal reproducible experiment using GRPO for mathematical reasoning, referencing the approach from [SimpleRL-Reason](https://hkust-nlp.notion.site/simplerl-reason) which uses a 7B model trained on 8K examples. Running this on 8 H100 80G GPU takes about 3 hours:

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml --num_processes=7 src/open_r1/grpo.py --config recipes/deepseek/DeepSeek-R1-Distill-Qwen-7B/grpo/config_base_math_smalllr.yaml
```

Our final [model](Dongwei/Qwen-2.5-7B_Base_Math_smalllr), while using different learning rates, loss functions and reward structures, achieves 69.4% accuracy on Math_500, demonstrating a 17%+ improvement over the base model.

To launch a Slurm job, run:

```shell
sbatch --output=/path/to/logs/%x-%j.out --err=/path/to/logs/%x-%j.err slurm/grpo.slurm {model} {dataset} {accelerator}
```

You can find more model configurations in the [recipes](./recipes).

## Evaluating models

We use `lighteval` to evaluate models, with custom tasks defined in `src/open_r1/evaluate.py`. For models which fit on a single GPU, run:

```shell
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilisation=0.8"
OUTPUT_DIR=data/evals/$MODEL

# AIME 2024
TASK=aime24
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# MATH-500
TASK=math_500
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# GPQA Diamond
TASK=gpqa:diamond
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 
```

> [!IMPORTANT]
> You must set `max_model_length=32768` in the `vllm` command to align with the `generation_size` we define per eval. Without this, `lighteval` will throw an error.

To increase throughput across multiple GPUs, use _data parallel_ as follows:

```shell
NUM_GPUS=8
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.8"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 
```

For large models which require sharding across GPUs, use _tensor parallel_ and run:

```shell
NUM_GPUS=8
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.8"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

export VLLM_WORKER_MULTIPROC_METHOD=spawn
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 
```

You can also launch an evaluation with `make evaluate`, specifying the model, task, and optionally the parallelism technique and number of GPUs.

To evaluate on a single GPU:

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24
```

To use Data Parallelism:

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24 PARALLEL=data NUM_GPUS=8
```

To use Tensor Parallelism:

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24 PARALLEL=tensor NUM_GPUS=8
```

## Reproducing Deepseek's evaluation results

> [!NOTE]
> The DeepSeek-R1 paper uses sampling with a temperature of 0.6, a top-p value of 0.95, and 64 responses per query to estimate `pass@1`. Below, we report the results from greedy decoding, which likely explains the small 1-3σ discrepancies between our results and theirs.

### MATH-500

We are able to reproduce Deepseek's reported results on the MATH-500 benchmark within ~1-3 standard deviations:

| Model                         | MATH-500 (🤗 LightEval) | MATH-500 (DeepSeek Reported) |
|:------------------------------|:-----------------------:|:----------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |          81.2           |             83.9             |
| DeepSeek-R1-Distill-Qwen-7B   |          91.8           |             92.8             |
| DeepSeek-R1-Distill-Qwen-14B  |          94.2           |             93.9             |
| DeepSeek-R1-Distill-Qwen-32B  |          95.0           |             94.3             |
| DeepSeek-R1-Distill-Llama-8B  |          85.4           |             89.1             |
| DeepSeek-R1-Distill-Llama-70B |          93.4           |             94.5             |

To reproduce these results use the following command:

```shell
NUM_GPUS=1 # Set to 8 for 32B and 70B models
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilisation=0.8,tensor_parallel_size=$NUM_GPUS"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|math_500|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

Alternatively, you can launch Slurm jobs as follows:

```shell
python scripts/run_benchmarks.py --model-id={model_id}  --benchmarks math_500
```

### GPQA Diamond

We are able to reproduce Deepseek's reported results on the GPQA Diamond benchmark within ~1-3 standard deviations:

| Model                         | GPQA Diamond (🤗 LightEval) | GPQA Diamond (DeepSeek Reported) |
|:------------------------------|:---------------------------:|:--------------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |            33.3             |               33.8               |
| DeepSeek-R1-Distill-Qwen-7B   |            48.4             |               49.1               |
| DeepSeek-R1-Distill-Qwen-14B  |            55.6             |               59.1               |
| DeepSeek-R1-Distill-Qwen-32B  |            58.6             |               62.1               |
| DeepSeek-R1-Distill-Llama-8B  |            51.0             |               49.0               |
| DeepSeek-R1-Distill-Llama-70B |            65.2             |               65.2               |

To reproduce these results use the following command:

```shell
NUM_GPUS=1 # Set to 8 for 32B and 70B models
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilisation=0.8,tensor_parallel_size=$NUM_GPUS"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|gpqa:diamond|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

```shell
python scripts/run_benchmarks.py --model-id={model_id}  --benchmarks gpqa
```

## Data generation

### Generate data from a smol distilled R1 model

The following example can be run in 1xH100. 
First install the following dependencies:

```shell
uv pip install "distilabel[vllm]>=1.5.2"
```

Now save the following snippet into a file named `pipeline.py` and run it with `python pipeline.py`. It will generate 4 outputs for each of the 10 examples (change the username for the repository to your org/user name):

```python
from datasets import load_dataset
from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration


prompt_template = """\
You will be given a problem. Please reason step by step, and put your final answer within \boxed{}:
{{ instruction }}"""

dataset = load_dataset("AI-MO/NuminaMath-TIR", split="train").select(range(10))

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Exchange with another smol distilled r1

with Pipeline(
    name="distill-qwen-7b-r1",
    description="A pipeline to generate data from a distilled r1 model",
) as pipeline:

    llm = vLLM(
        model=model_id,
        tokenizer=model_id,
        extra_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 8192,
        },
        generation_kwargs={
            "temperature": 0.6,
            "max_new_tokens": 8192,
        },
    )
    prompt_column = "problem"
    text_generation = TextGeneration(
        llm=llm, 
        template=prompt_template,
        num_generations=4,
        input_mappings={"instruction": prompt_column} if prompt_column is not None else {}
    )


if __name__ == "__main__":
    distiset = pipeline.run(dataset=dataset)
    distiset.push_to_hub(repo_id="username/numina-deepseek-r1-qwen-7b")
```

Take a look at the sample dataset at [HuggingFaceH4/numina-deepseek-r1-qwen-7b](https://huggingface.co/datasets/HuggingFaceH4/numina-deepseek-r1-qwen-7b).


### Generate data from DeepSeek-R1

To run the bigger DeepSeek-R1, we used 2 nodes, each with 8×H100 GPUs using the slurm file present in this repo at `slurm/generate.slurm`. First, install the dependencies:

(for now we need to install the vllm dev wheel that [fixes the R1 cuda graph capture](https://github.com/vllm-project/vllm/commits/221d388cc5a836fa189305785ed7e887cea8b510/csrc/moe/moe_align_sum_kernels.cu))
```shell
pip install https://wheels.vllm.ai/221d388cc5a836fa189305785ed7e887cea8b510/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu121

uv pip install "distilabel[vllm,ray,openai]>=1.5.2"
```

And then run the following command:

```shell
sbatch slurm/generate.slurm \
    --hf-dataset AI-MO/NuminaMath-TIR \
    --temperature 0.6 \
    --prompt-column problem \
    --model deepseek-ai/DeepSeek-R1 \
    --hf-output-dataset username/r1-dataset
```

> [!NOTE]  
> While the job is running, you can setup an SSH tunnel through the cluster login node to access the Ray dashboard from your computer running `ssh -L 8265:ray_ip_head_node:8265 <login_node>`, then browsing `http://localhost:8265`

## Contributing

Contributions are welcome. Please refer to https://github.com/huggingface/open-r1/issues/23.
