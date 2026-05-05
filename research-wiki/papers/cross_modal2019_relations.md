---
type: paper
node_id: paper:cross_modal2019_relations
title: "Cross-modal Scene Graph Matching for Relationship-aware Image-Text Retrieval"
origin_skill: research-wiki
created_at: 2026-04-21T21:45:31Z
updated_at: 2026-04-21T21:48:31Z
tags: ["relations", "retrieval", "scene-graphs", "vlm"]
authors: ["Unknown"]
year: 2019
venue: arXiv
external_ids: {"arxiv": "1910.05134"}
relevance: core
---

# One-line thesis

Image-text retrieval of natural scenes has been a popular research topic.

## Problem / Gap

Since image and text are heterogeneous cross-modal data, one of the key challenges is how to learn comprehensive yet unified representations to express the multi-modal data.

## Method

In the light of recent success of scene graph in many CV and NLP tasks for describing complex natural scenes, we propose to represent image and text with two kinds of scene graphs: visual scene graph (VSG) and textual scene graph (TSG), each of which is exploited to jointly characterize objects and relationships in the corresponding modality.

## Key Results

- We achieve state-ofthe-art results on Flickr30k and MSCOCO, which verifies the advantages of our graph matching based approach for image-text retrieval.

## Assumptions

- Initialized from PDF metadata and first-page abstract extraction; refine after full-paper reading.

## Limitations / Failure Modes

- This wiki initialization did not extract full-paper limitations sections automatically.

## Reusable Ingredients

- Baseline or ablation point for CLIP-style alignment
- Structural supervision or latent-structure design reference

## Open Questions

- How directly does this paper transfer to parser-free relational CLIP training and evaluation?

## Claims

- Image-text retrieval of natural scenes has been a popular research topic.

## Connections

- out: `addresses_gap` -> `gap:G1`
- out: `addresses_gap` -> `gap:G3`

## Relevance to This Project

Core prior art for the RelBottleneck-CLIP research direction.
