---
type: paper
node_id: paper:compositional2024_relations
title: "Compositional Chain-of-Thought Prompting for Large Multimodal Models"
origin_skill: research-wiki
created_at: 2026-04-21T21:45:31Z
updated_at: 2026-04-21T21:48:31Z
tags: ["benchmarks", "compositionality", "llm", "reasoning", "relations", "scene-graphs", "vlm"]
authors: ["Unknown"]
year: 2024
venue: arXiv
external_ids: {"arxiv": "2311.17076"}
relevance: core
---

# One-line thesis

The combination of strong visual backbones and Large Language Model (LLM) reasoning has led to Large Multimodal Models (LMMs) becoming the current standard for a wide range of vision and language (VL) tasks.

## Problem / Gap

However, recent research has shown that even the most advanced LMMs still struggle to capture aspects of compositional visual reasoning, such as attributes and relationships between objects.

## Method

To overcome this, inspired by chain-of-thought methods, we propose Compositional Chain-of-Thought (CCoT), a novel zero-shot Chain-of-Thought prompting method that utilizes SG representations in order to extract compositional knowledge from an LMM.

## Key Results

- However, recent research has shown that even the most advanced LMMs still struggle to capture aspects of compositional visual reasoning, such as attributes and relationships between objects.
- Through extensive experiments, we find that the proposed CCoT approach not only improves LMM performance on several vision and language (VL) compositional benchmarks but also improves the performance of several popular LMMs on general multimodal benchmarks, without the need for fine-tuning or annotated ground-truth SGs.

## Assumptions

- Initialized from PDF metadata and first-page abstract extraction; refine after full-paper reading.

## Limitations / Failure Modes

- This wiki initialization did not extract full-paper limitations sections automatically.

## Reusable Ingredients

- Relation-sensitive evaluation protocol or benchmark construction ideas
- Structural supervision or latent-structure design reference
- Reasoning-focused diagnostics for compositional failures

## Open Questions

- How directly does this paper transfer to parser-free relational CLIP training and evaluation?

## Claims

- The combination of strong visual backbones and Large Language Model (LLM) reasoning has led to Large Multimodal Models (LMMs) becoming the current standard for a wide range of vision and language (VL) tasks.

## Connections

- out: `addresses_gap` -> `gap:G1`
- out: `addresses_gap` -> `gap:G2`
- out: `addresses_gap` -> `gap:G3`

## Relevance to This Project

Core prior art for the RelBottleneck-CLIP research direction.
