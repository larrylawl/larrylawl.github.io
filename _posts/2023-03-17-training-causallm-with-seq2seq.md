---
title: 'Simple solution to training causal LMs with seq2seq objective'
date: 2022-12-27
permalink: /posts/2022/12/training-causallm-with-seq2seq/
tags:
  - research
---

## Motivation
How do we train causal language models (e.g. Alpaca, LLaMA, gpt-neox-20b...) with seq2seq objective? This goal is important because we want to instruction-tune our causal LMs, especially since Alpaca is the best open model at time of writing.

## Approach
The naive solution is to concatenate the source and target strings. However, the main issue here is that the loss is incurred in the next-word-prediction of the source strings. 

To circumvent this, [Alpaca](https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py) simply ignored the loss in the source strings. Concretely:

```
def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]  # concatenate source and target strings
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX  # the source string's loss is ignored with IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

```

Note how the source string's loss is ignored with `IGNORE_INDEX`

## Implications

**Seq2Seq prompting.**

In concatenating the source and target strings, it may not be obvious to the model how to differentiate the source from target strings. I suspect that Alpaca/self-instruct circumvented this by making the differentiation clear via prompts:

```
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
```

Notice how `### Instruction:` tells the model where the source string is while `### Response:` tells the model where the target string is.

**Increased GPU Memory usage**. To my understanding, the `input` and `labels` will now both be the concatenated source and target strings. In contrast for seq2seq models, the `input` will only be the source strings while the `labels` will only be the target strings. Thus this neat trick incurs additional GPU memory.

**Packing is more intuitive with causal LM.** Packing is the act of packing training examples together. In causal LM, we can pack via

```
(source->target)[IGNORE_INDEX](source->target)[IGNORE_INDEX]...(source->target)[IGNORE_INDEX])
```

Notice how the target string immediately comes after the source. In contrast, packing for seq2seq LM will look like

```
Input: (source)[IGNORE_INDEX](source)[IGNORE_INDEX]...(source)[IGNORE_INDEX]
Target: (target)[IGNORE_INDEX](target)[IGNORE_INDEX]...(target)[IGNORE_INDEX]
```

To me, it's not intuitive that the model can match the ith target to the ith source string. 

## Credits
Cheers to Alpaca, LlaMMA, and OS for finally solving this engineering puzzle for me! Do LMK if any parts don't make sense to you - I'm still learning myself.