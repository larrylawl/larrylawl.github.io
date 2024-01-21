---
title: 'Engineering Tricks'
date: 2022-12-27
permalink: /posts/2022/12/engineering-tricks/
tags:
  - advice
---
Compilation of tricks I find useful.


## LLMs Training
### Memory efficient attention
- [Flash Attention 2](https://github.com/Dao-AILab/flash-attention) which has a HF integration to llama 2 by fastchat [here](https://github.com/lm-sys/FastChat/blob/main/fastchat/train/llama2_flash_attn_monkey_patch.py)
- [Xformers] which also has a HF integration to llama2 by fastchat [here](https://github.com/lm-sys/FastChat/blob/main/fastchat/train/llama_xformers_attn_monkey_patch.py)

### Fused Kernels
- `matmul + bias`, `matmul + bias + gelu`, `cross-entropy loss`, `rotary embedding`, `fused dropout + residual + layer norm`, implemented by FA2 [here](https://github.com/Dao-AILab/flash-attention/tree/main/training)
- FusedAdam, implemented by torch natively or via deepspeed [here](https://deepspeed.readthedocs.io/en/latest/optimizers.html)
- Fused SwiGLU from xformers.

### Parallelism
- **Difference between 3D parallelism and ZeRO-3**, explained [here](https://github.com/microsoft/DeepSpeed/discussions/1911~).

### Misc
- **[Importance of CUDA versions](https://stackoverflow.com/questions/9727688/how-to-get-the-cuda-version)**. Ensure that cuda driver version (output from `nvidia-smi`) supports cuda toolkit version (output from `nvcc --version`). More info [here](https://docs.nvidia.com/deploy/cuda-compatibility/)
- **If LLM is pre-trained in bf16, finetune it with bf16 (not fp16).** See more [h[here](https://huggingface.co/docs/transformers/v4.13.0/en/performance#bf16)
- **Packing with attention segment masking.** See this twitter thread [here](https://twitter.com/agihippo/status/1645798187339505666). Credits to the twitters in thread.
- **4 bit optimiser PEFT training**. See more [here](https://huggingface.co/blog/4bit-transformers-bitsandbytes).

## LLM Loading
- Deepspeed: need to use `deepspeed.zero.init`
- HF: `model.from_pretrained(low_cpu_mem_usage=True)`

## LLM Inference
- Use [vLLM](https://github.com/vllm-project/vllm) - paged attention and 2x speed up compared to vanilla transformers on internal benchmark
- FP16 inference.
- **8 bit inferece**. 

```
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_OR_PATH,
                                             device_map="auto",
                                             load_in_8bit=True,)
model.eval();
```

> To support 8 bit inference for your own models, follow this [discussion](https://github.com/huggingface/transformers/issues/22488). 

- **Generate text excluding prompt.**

```
def remove_input_ids_from_output_ids(input_ids, output_ids, tokenizer):
    """ Remove input_ids from output_ids. Applicable only for causalLM models, which output input_ids in outputs."""
    input_ids_lens = torch.tensor([
    tokenized.ne(tokenizer.pad_token_id).sum().item() for tokenized in input_ids])
    padding_lens = torch.tensor([(tokenized == tokenizer.pad_token_id).sum().item() for tokenized in input_ids])
    total_lens = input_ids_lens + padding_lens
    outputs = [op[total_lens[i]:] for i, op in enumerate(output_ids)]
    return outputs
```

## Software Engineering
- **[Set up passwordless ssh](https://linuxize.com/post/how-to-setup-passwordless-ssh-login/).** If you still have issues, it's likely file permissions. Set permissions according to [this](https://superuser.com/questions/215504/permissions-on-private-key-in-ssh-folder).
- **[Project Setup Boilerplate](https://goodresearch.dev/setup.html)**. Finally doing away with relative imports!
- **[Gitignore Boilerplate](https://github.com/github/gitignore/blob/main/Python.gitignore)**. Initialise gitignore from here.
- **Commands to create venv**

```
python3.9 -m venv name-of-venv-folder
# python3.9 -m venv name-of-venv-folder --system-site-packages  
source name-of-venv-folder/bin/activate
```

- **Find kernel for jupyter notebook.** (general jupyter notebook, not jupyter notebook + VScode) 

```
# activate env
source env/bin/activate

# ensures jupyter and python is in same environment. See https://stackoverflow.com/questions/48193822/import-of-package-works-in-ipython-shell-but-not-in-jupyter-notebook
pip install notebook --ignore-installed  

# these needs to be same
which python3
which jupyter

# use jupyter notebook in venv. See
ipython kernel install --user --name=venv

# to see env, simply refresh
```

For Jupyter notebook + VScode, ensure that you've installed the following extensions: jupyter, python.

## HuggingFace
- **Sharing datasets**. Guides on how to share my own HF datasets.
  - [Sharing](https://huggingface.co/docs/datasets/share)
  - [Create a dataset loading script](https://huggingface.co/docs/datasets/dataset_script#create-a-dataset-loading-script)
  - [Create a dataset card](https://huggingface.co/docs/datasets/dataset_card) 

## Docker/Singularity
- **Maping between Nvidia container version and CUDA Toolkit and Pytorch version**: Link [here](https://docs.nvidia.com/deeplearning/frameworks/pdf/PyTorch-Release-Notes.pdf)
- **To run docker for multinode training, set up network bridge between containers.** See [here](https://docs.sylabs.io/guides/3.1/user-guide/networking.html).
