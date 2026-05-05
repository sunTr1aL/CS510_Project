---
type: paper
node_id: paper:uknow2024_vlm
title: "UKnow: A Unified Knowledge Protocol with Multimodal Knowledge Graph Datasets for Reasoning and Vision-Language Pre-Training"
origin_skill: research-wiki
created_at: 2026-04-21T21:45:31Z
updated_at: 2026-04-21T21:48:31Z
tags: ["benchmarks", "reasoning", "scene-graphs", "vlm"]
authors: ["Unknown"]
year: 2024
venue: arXiv
external_ids: {"arxiv": "2302.06891"}
relevance: core
---

# One-line thesis

This work presents a unified knowledge protocol, called UKnow, which facilitates knowledge-based studies from the perspective of data.

## Problem / Gap

Particularly focusing on visual and linguistic modalities, we categorize data knowledge into five unit types, namely, in-image, in-text, cross-image, cross-text, and image-text, and set up an efficient pipeline to help construct the multimodal knowledge graph from any data collection.

## Method

Particularly focusing on visual and linguistic modalities, we categorize data knowledge into five unit types, namely, in-image, in-text, cross-image, cross-text, and image-text, and set up an efficient pipeline to help construct the multimodal knowledge graph from any data collection.

## Key Results

- Experiments on 4 benchmarks demonstrate the potential of UKnow in supporting common-sense reasoning and boosting vision-language pre-training with a single dataset, benefiting from its unified form of knowledge organization.

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

- This work presents a unified knowledge protocol, called UKnow, which facilitates knowledge-based studies from the perspective of data.

## Connections

- out: `addresses_gap` -> `gap:G1`
- out: `addresses_gap` -> `gap:G2`
- out: `addresses_gap` -> `gap:G3`

## Relevance to This Project

Core prior art for the RelBottleneck-CLIP research direction.
