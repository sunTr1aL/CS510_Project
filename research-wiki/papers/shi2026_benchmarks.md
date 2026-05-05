---
type: paper
node_id: paper:shi2026_benchmarks
title: "Vision Transformers Need More Than Registers"
origin_skill: research-wiki
created_at: 2026-04-21T21:45:31Z
updated_at: 2026-04-21T21:48:31Z
tags: ["benchmarks"]
authors: ["Cheng Shi", "Yizhou Yu", "Sibei Yang"]
year: 2026
venue: arXiv
external_ids: {"arxiv": "2602.22394"}
relevance: peripheral
---

# One-line thesis

Vision Transformers (ViTs), when pre-trained on large-scale data, provide general-purpose representations for diverse downstream tasks.

## Problem / Gap

However, artifacts in ViTs are widely observed across different supervision paradigms and downstream tasks.

## Method

However, artifacts in ViTs are widely observed across different supervision paradigms and downstream tasks.

## Key Results

- However, artifacts in ViTs are widely observed across different supervision paradigms and downstream tasks.
- Our solution selectively integrates patch features into the CLS token, reducing the influence of background-dominated shortcuts and consistently improving performance across 12 benchmarks under label-, text-, and self-supervision.

## Assumptions

- Initialized from PDF metadata and first-page abstract extraction; refine after full-paper reading.

## Limitations / Failure Modes

- This wiki initialization did not extract full-paper limitations sections automatically.

## Reusable Ingredients

- Relation-sensitive evaluation protocol or benchmark construction ideas

## Open Questions

- How directly does this paper transfer to parser-free relational CLIP training and evaluation?

## Claims

- Vision Transformers (ViTs), when pre-trained on large-scale data, provide general-purpose representations for diverse downstream tasks.

## Connections

- out: `addresses_gap` -> `gap:G2`

## Relevance to This Project

Peripheral prior art for the RelBottleneck-CLIP research direction.
