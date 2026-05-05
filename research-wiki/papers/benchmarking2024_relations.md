---
type: paper
node_id: paper:benchmarking2024_relations
title: "Benchmarking Multi-Image Understanding in Vision and Language Models: Perception, Knowledge, Reasoning, and Multi-Hop Reasoning"
origin_skill: research-wiki
created_at: 2026-04-21T21:45:31Z
updated_at: 2026-04-21T21:48:31Z
tags: ["benchmarks", "llm", "reasoning", "relations"]
authors: ["Unknown"]
year: 2024
venue: arXiv
external_ids: {"arxiv": "2406.12742"}
relevance: core
---

# One-line thesis

The advancement of large language models (LLMs) has significantly broadened the scope of applications in natural language processing, with multi-modal LLMs extending these capabilities to integrate and interpret visual data.

## Problem / Gap

However, existing benchmarks for visual language models (VLMs) predominantly focus on singleimage inputs, neglecting the crucial aspect of multi-image understanding.

## Method

In this paper, we introduce a Multi-Image Relational Benchmark MIRB, designed to evaluate VLMs’ ability to compare, analyze, and reason across multiple images.

## Key Results

- However, existing benchmarks for visual language models (VLMs) predominantly focus on singleimage inputs, neglecting the crucial aspect of multi-image understanding.
- In this paper, we introduce a Multi-Image Relational Benchmark MIRB, designed to evaluate VLMs’ ability to compare, analyze, and reason across multiple images.

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

- The advancement of large language models (LLMs) has significantly broadened the scope of applications in natural language processing, with multi-modal LLMs extending these capabilities to integrate and interpret visual data.

## Connections

- out: `addresses_gap` -> `gap:G2`
- out: `addresses_gap` -> `gap:G3`

## Relevance to This Project

Core prior art for the RelBottleneck-CLIP research direction.
