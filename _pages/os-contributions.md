---
layout: archive
title: "Open Source Contributions"
permalink: /os-contributions/
author_profile: true
---

{% include base_path %}

While I work in defence, I love open source (OS)! I try to contribute to OS in my small ways. Here's a list of my contributions:
- [Fixed resuming checkpointing bug in `lit_gpt` by `Lightning-AI`](https://github.com/Lightning-AI/lit-gpt/pull/661). Previously resuming checkpoint may not take the latest checkpoint.
- [Fixed dataloader bug in `TinyLlama`, a popular pretraining codebase](https://github.com/jzhang38/TinyLlama/issues/67). The bug resulted in >35% of data being duplicated in the dataloader.
- [Accelerate support for GLM](https://huggingface.co/THUDM/glm-10b-chinese/discussions/2). Enable 8-bit inference and CPU offloading for GLM (SOTA Chinese LLM).
- [Alpaca Indon dataset](https://huggingface.co/datasets/larrylawl/alpaca-cleaned-indon). Translated `alpaca` dataset to Indonesian using NLLB.
- [豆瓣读书 dataset](https://huggingface.co/datasets/larrylawl/douban-dushu). Dataset containing chinese book reviews from 豆瓣读书.
