---
type: paper
node_id: paper:komanduri2025_vlm
title: "CausalVLBench: Benchmarking Visual Causal Reasoning in Large Vision-Language Models"
origin_skill: research-wiki
created_at: 2026-04-21T21:45:31Z
updated_at: 2026-04-21T21:48:31Z
tags: ["benchmarks", "causal-reasoning", "llm", "reasoning", "vlm"]
authors: ["Aneesh Komanduri", "Karuna Bhaila", "Xintao Wu"]
year: 2025
venue: arXiv
external_ids: {"arxiv": "2506.11034"}
relevance: related
---

# One-line thesis

Large language models (LLMs) have shown remarkable ability in various language tasks, especially with their emergent in-context learning capability.

## Problem / Gap

Despite increasing interest in the utility of LLMs in causal reasoning tasks such as causal discovery and counterfactual reasoning, there has been relatively little work showcasing the abilities of LVLMs on visual causal reasoning tasks.

## Method

We take this opportunity to formally introduce a comprehensive causal reasoning benchmark for multi-modal in-context learning from LVLMs.

## Key Results

- Large language models (LLMs) have shown remarkable ability in various language tasks, especially with their emergent in-context learning capability.
- Extending LLMs to incorporate visual inputs, large vision-language models (LVLMs) have shown impressive performance in tasks such as recognition and visual question answering (VQA).

## Assumptions

- Initialized from PDF metadata and first-page abstract extraction; refine after full-paper reading.

## Limitations / Failure Modes

- This wiki initialization did not extract full-paper limitations sections automatically.

## Reusable Ingredients

- Relation-sensitive evaluation protocol or benchmark construction ideas
- Reasoning-focused diagnostics for compositional failures

## Open Questions

- How directly does this paper transfer to parser-free relational CLIP training and evaluation?

## Claims

- Large language models (LLMs) have shown remarkable ability in various language tasks, especially with their emergent in-context learning capability.

## Connections

- out: `addresses_gap` -> `gap:G2`

## Relevance to This Project

Related prior art for the RelBottleneck-CLIP research direction.
