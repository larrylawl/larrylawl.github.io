---
title: 'Simple tricks for Retrieval Augmented Generation (RAG) systems'
date: 2023-05-04
permalink: /posts/2023/03/rag-tricks/
tags:
  - research
---

## Motivation
Retrieval Augmented Generation (RAG) is a framework for retrieving facts from an external knowledge base to ground large language models (LLMs) on the most accurate, up-to-date information. RAG is increasingly popular in industry as it's simple to implement yet powerful. Here I'll share some tricks in improving RAG systems. 

## Use structured text over PDF text
RAG systems typically preprocess the data as chunks of text, embed them, then store them in the search index. If the data is in PDF format, we need to additionally convert the PDF to text. However, this preprocessing can be noisy. If the data is in structured format, we can use a parser to get a much cleaner text. For example for `docx`, we can use `pandoc` to conver them to clean text.

Here's the quantitative results for RAG on (my) PDF documents:

| Metric            | Single Hop   | Table Related   | Multi Hop   | Weighted Avg   |
| ----------------- | ------------ | --------------- | ----------- | --------------- |
| Cosine Similarity | 0.7993       | 0.8121          | 0.8012      | 0.8036          |
| Rouge1 F1         | 0.3371       | 0.3356          | 0.3121      | 0.332           |
| Rouge1 Recall     | 0.5847       | 0.5608          | 0.4423      | 0.5506          |
| Retrieval F1      | 0.6042       | 0.7             | 0.5667      | 0.6186          |
| Retrieval Recall  | 0.8125       | 1               | 0.5556      | 0.7692          |

Here's the results for RAG on `pandoc` documents:

| Metric            | Single Hop   | Table Related   | Multi Hop   | Weighted Avg   |
| ----------------- | ------------ | --------------- | ----------- | --------------- |
| Cosine Similarity | 0.7829       | 0.845           | 0.8228      | 0.8098          |
| Rouge1 F1         | 0.3449       | 0.5225          | 0.4619      | 0.4223          |
| Rouge1 Recall     | 0.4984       | 0.7046          | 0.5372      | 0.5701          |
| Retrieval F1      | 0.6042       | 0.8000          | 0.5000      | 0.6170          |
| Retrieval Recall  | 0.8125       | 1.0000          | 0.5000      | 0.7436          |


## Prepending document title to your chunks
Every document has a document title (eg file name) and is chunked. One way to improve retrieval is to prepend the document title to each chunk. 

Here's the quantitative results without prepending:

| Metric                                | Single Hop   | Table Related   | Multi Hop   | Weighted Avg   |
| ------------------------------------- | ------------ | --------------- | ----------- | --------------- |
| Cosine Similarity (Instructor-XL)    | 0.8114       | 0.8559          | 0.8667      | 0.8371          |
| Rouge1 F1                            | 0.3292       | 0.4859          | 0.4266      | 0.4013          |
| Rouge1 Recall                        | 0.5829       | 0.6335          | 0.6721      | 0.617           |
| Retrieval F1                         | 0.7111       | 0.8182          | 0.4167      | 0.6364          |
| Retrieval Recall                     | 0.8667       | 0.9091          | 0.4167      | 0.7179          |

Here's the results with prepending:

| Metric                                | Single Hop   | Table Related   | Multi Hop   | Weighted Avg   |
| ------------------------------------- | ------------ | --------------- | ----------- | --------------- |
| Cosine Similarity (Instructor-XL)    | 0.8381       | 0.8701          | 0.8135      | 0.8445          |
| Rouge1 F1                            | 0.3967       | 0.4848          | 0.3151      | 0.4117          |
| Rouge1 Recall                        | 0.6671       | 0.6806          | 0.5623      | 0.6521          |
| Retrieval F1                         | 0.8000       | 0.9091          | 0.5000      | 0.7209          |
| Retrieval Recall                     | 0.9333       | 1.0000          | 0.5000      | 0.7947          |

Credits to Shin Youn for this tip

## Converting flat tables to JSON format
For flat tables (i.e. mxn table, with only one value per cell) which are long, the entries near the bottom of the table may be too far away from the headers. One simple trick is to convert these flat tables to JSON format, such that the key is the table header and the value is the cell value.

Credits to Jun How and Qian Hui for this tip.