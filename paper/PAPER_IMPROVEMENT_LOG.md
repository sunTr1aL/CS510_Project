# Paper Improvement Log

Input: `NARRATIVE_REPORT.md`  
Venue style: ACM Proceedings / SIGCONF anonymous review  
Date: 2026-05-05  
Constraint: ignore upper page limit; keep at least 8 pages of main body before references/appendix.

## Baseline After Expansion

- Artifact: `main_round0_expanded.pdf`
- Build: 10 pages total; references begin on page 9.
- Main body length target: satisfied with pages 1--8 before references.
- Verification: no unresolved citations or references; overfull boxes are below 10pt.

## Round 1

- Gemini job: `17d4d983abc847ad908422edb64d7381`
- Gemini thread: `b8c6b81a6b9145fd85abc5eedd7b7815`
- Score: 8.5/10
- Artifact: `main_round1.pdf`
- Changes:
  - Added an introduction disclaimer that the 1k COCO guardrail is a catastrophic-collapse signal, not a certificate of full retrieval parity.
  - Softened the main tradeoff phrasing from a failure-mode claim to a Pareto-style tradeoff claim.
  - Standardized R033/R040 wording to "no-channel residual".
  - Marked R028 with a star in Figure 1.
  - Replaced the residual-safety wording with a nearest-neighbor preservation caveat.
  - Reframed ROP/principal-subspace projection and asymmetric Flickr retrieval behavior as future-work hypotheses.
  - Elevated seed-fragile COCO retention in the conclusion.

## Round 2

- Gemini job: `74e873ede03045eeb982591c38d0909e`
- Score: 9.2/10
- Artifact: `main_round2.pdf`
- Changes:
  - Clarified that relation accuracy is the simple average across evaluated ARO and SugarCrepe sub-suites.
  - Contextualized R022 as an early diagnostic run omitted from the main tables.
  - Added text linking the Figure 1 star marker to selected R028.

## Round 3

- Gemini job: `69695b77a8714af49ce30893542232cb`
- Score: 9.5/10
- Artifact: `main_round3.pdf`
- Changes: none; reviewer declared the paper submission ready.

## Round 4

- Gemini job: `337b95400c344ff3967ef054a2d142e6`
- Score: 10/10
- Artifact: `main_round4.pdf`
- Changes: none; reviewer declared no edits required.

## Residual Build Notes

- The ACM class warning about `printacmref=false` remains intentionally because this is an anonymous project-style draft without a printed ACM reference strip.
- BibTeX warns about missing publisher/address/pages for a few entries; citation metadata is intentionally conservative where exact venue metadata is unavailable.
- Local TeX uses the project-local `paper/texmf` tree and disables microtype expansion because the minimal TeX setup lacks the full scalable font stack.

---

## Fresh Citation-Grounded Improvement Loop (2026-05-05)

User instruction: use more references based on papers included in `literature/` and `research-wiki/`; keep page limit ignored; preserve at least 8 pages before references/appendix.

Pre-loop citation pass:
- Added structured image-text matching and scene/knowledge-graph references from local literature and research-wiki.
- Added benchmark/counterfactual reasoning references for MIRB, CausalVLBench, causal-order probes, and compositional chain-of-thought prompting.
- Added PEFT/null-space/principal-subspace preservation references.
- Recompiled to 11 pages total; references begin after the 9-page main body.
- Preserved `main_round0_original.pdf` as the post-citation baseline. The previous baseline PDF is preserved as `main_round0_original_pre_20260505_fresh_loop.pdf`.

### Score Progression

| Round | Score | Verdict | Key Changes |
|-------|-------|---------|-------------|
| Round 0 | n/a | n/a | Citation-grounded baseline after expanding local references |
| Round 1 | 7/10 | Almost | Added bottleneck-vs-rank framing, validation-gate protocol, Winoground image-pair contrast explanation, and additional local references |
| Round 2 | 8/10 | Ready / Yes | Clarified fixed gate, appendix self-containedness, Figure 1 seed-dispersion pointer, and residual-gradient citation |

## Fresh Round 1 Review & Fixes

- Gemini job: `61b2b1bd591a4415b1414af5b4d42700`
- Gemini thread: `8aa3ecd05bf24509a14a8b272e93fca7`
- Artifact after fixes: `main_round1.pdf`

<details>
<summary>Gemini Review (Fresh Round 1)</summary>

This senior reviewer assessment evaluates the paper **"Anchored Relational Residual Adaptation for CLIP"** based on the provided LaTeX source and project context.

### 1. Overall Score: 7/10 (Accept)
The paper is a strong candidate for an ACM-style venue. It distinguishes itself through exceptional empirical honesty, a well-defined and constrained problem scope ("guardrail" preservation), and a clear-eyed analysis of its own limitations (Winoground failures, seed fragility). While it lacks comparison to standard PEFT baselines like LoRA, it provides a rigorous internal ablation of its proposed architecture and objective.

### 2. Summary
The paper introduces Relational Residual Bottlenecking (RRB), a method for adding relational sensitivity to frozen CLIP encoders using a bounded, multi-channel residual path. It demonstrates that structured counterfactual supervision, when combined with feature-space anchoring, improves relation-benchmark accuracy while avoiding the catastrophic retrieval collapse seen in replacement-based or unconstrained residual methods. The work is notably conservative, framing its contributions as a characterization of a specific relation-vs-preservation tradeoff rather than a general solution to compositional reasoning.

### 3. Strengths
- Methodological Transparency: The inclusion of negative results (Winoground, Flickr30k tax, diffuse channel responsibility) and the "useful negative result" regarding the targeted slot loss significantly increase the paper's credibility and scientific value.
- Rigorous Ablation Suite: The comparison between RRB, hard replacement bottlenecks (R011/R014), and no-channel residual controls (R033/R040) effectively isolates the multi-channel interface as a key preservation mechanism.
- Conceptual Clarity: The distinction between "architectural bottlenecks" and "semantic slots" is handled with technical precision, avoiding the common pitfall of over-claiming interpretability without evidence.
- Structural Integrity: The paper is well-organized, with a failure taxonomy and seed-stability analysis that go beyond simple leaderboard reporting to offer genuine diagnostic insight.
- Framing: The "guardrail" terminology and the focus on maintaining the pretrained embedding path as "dominant" provide a refreshing and practical lens for multimodal adaptation.

### 4. Weaknesses
- MAJOR: Absence of External PEFT Baselines. While acknowledged in Section 5.4, the lack of a head-to-head comparison with LoRA or simple prompt tuning makes it difficult to assess if RRB's "Pareto-style tradeoff" is unique to its architecture or a general property of low-rank/residual adaptation on this data.
- MAJOR: Seed Fragility on Retrieval. The paper reports that COCO retention is seed-fragile. While this is analyzed honestly, the paper doesn't propose a robust selection rule or a "safe" hyperparameter regime that guarantees guardrail passage, leaving the method's reliability in question.
- MINOR: Theoretical "Why" for the Bottleneck. The paper empirically shows the bottleneck helps preservation (vs. no-channel controls), but the theoretical intuition (e.g., in terms of gradient alignment or spectral properties of the residual) remains underdeveloped.
- MINOR: Winoground Interpretation. The paper treats the Winoground failure as an "overgeneralization" warning. However, more analysis of why the counterfactual training signal fails to translate to the paired-image setting (e.g., lack of inter-image contrast during training) would strengthen the discussion.

### 5. Actionable Fixes (LaTeX Only)
For the PEFT Weakness (Section 5.4 & Intro):
- Fix: Explicitly frame RRB as a "bottleneck-first" study. Add a sentence in the Introduction or Related Work clarifying that RRB is designed to test the architectural constraint of the multi-channel interface, which is distinct from the rank constraint of LoRA. This shifts the focus from "beating" PEFT to "understanding the role of bottlenecks in residual preservation."

For the Seed Fragility (Section 4.4 & 5.1):
- Fix: Propose a "Validation Gate" protocol in Section 3.6. Instead of just a "selection rule," describe a procedure where multiple seeds are trained and only those passing the COCO/ImageNet gate are considered for deployment, framing the variance not as a failure of the method but as a necessary filtering step for safe multimodal adaptation.

For the Winoground Failure (Section 4.5):
- Fix: Add a brief sentence in Section 5.2 (Generalization Gap) noting that the current training objective (Eq. 2) only contrasts a single image with two captions. Mention as a specific future direction that extending the margin loss to "Image-Pair Contrast" (mimicking the Winoground structure) might be necessary for broader binding.

### 6. Missing References
The paper has integrated many local citations, but could further strengthen its "structured" and "communication" narrative using the following from the project's literature:
- Semantic Communication: Dery et al. (2026), "CACHE-TO-CACHE: DIRECT SEMANTIC COMMUNICATION BETWEEN LARGE LANGUAGE MODELS" is highly relevant to the "multi-channel bottleneck" as a form of latent communication/summarization.
- Graph Reasoning: Shi et al. (2026), "Cross-modal multi-relational graph reasoning" could be cited in Section 2 (Structured matching) to further bolster the motivation for graph-like structure.
- VLM Fundamentals: Darcet et al. (2023/2024), "Vision Transformers Need More Than Registers" is relevant to Section 3.2 when discussing token features and sequence summarization.
- Latent Alignment: Fu et al. (2026), "Latent Space Communication via K-V Cache Alignment" provides a complementary perspective on aligning latent spaces without full fine-tuning.

### 7. Verdict: Almost
The paper is excellently written and scientifically robust in its internal analysis. It is "Almost" ready; if it can strengthen the "Missing Baselines" section by adding the "bottleneck-vs-rank" framing and integrate the remaining high-value local literature on semantic communication/latent alignment, it will be a very strong "Accept." The author's decision to prioritize honesty over "SOTA chasing" is the paper's greatest asset and should be preserved.

</details>

### Fresh Round 1 Fixes Implemented

1. Added Introduction and Related Work language framing RRB as a bottleneck-first architectural study rather than a PEFT-dominance claim.
2. Added a validation-gate protocol to Section 3.6 and linked it to seed pass-rate interpretation in Section 4.4.
3. Added a Section 5.2 explanation that the current loss contrasts one image with edited captions, whereas Winoground requires two-image/two-caption assignment; added image-pair contrast as future work.
4. Added local/research-wiki references: CMRGR, ViT token aggregation artifacts, latent KV-cache alignment, and Cache-to-Cache.
5. Recompiled with BibTeX and saved `main_round1.pdf`.

## Fresh Round 2 Review & Fixes

- Gemini job: `3696b7b3b1b948bd80e5f13b0ace9fad`
- Gemini thread: `8aa3ecd05bf24509a14a8b272e93fca7`
- Final artifact: `main_round2.pdf`

<details>
<summary>Gemini Review (Fresh Round 2)</summary>

This re-assessment evaluates the updated paper after the Round 2 LaTeX-only revisions.

### 1. Overall Score: 8/10 (Accept)
The paper has matured from a promising technical report into a high-quality, scientifically rigorous submission. By shifting the framing from a "SOTA-chasing" PEFT method to a "bottleneck-first" architectural study, the authors have effectively neutralized the lack of external baselines. The honesty regarding seed fragility and the Winoground gap is now a core strength of the paper's narrative.

### 2. Summary
The paper presents Relational Residual Bottlenecking (RRB), an architectural intervention for CLIP that adds relational sensitivity via a multi-channel residual path. The work characterizes a specific "guardrail" tradeoff, demonstrating that while relational gains are repeatable across seeds, retrieval preservation requires a procedural validation gate. The authors provide a detailed failure taxonomy and empirical evidence that structured counterfactuals, rather than targeted slot losses, are the primary driver of the observed improvements.

### 3. Strengths
- Scientific Honesty (Highest Signal): The transformation of the "seed fragility" weakness into a "Validation Gate" protocol (Section 3.6) is excellent. It moves the paper from "my method might work" to "here is the procedure required to make this method safe for deployment."
- Framing and Scope: The explicit positioning in the Introduction as a "bottleneck-first study" (contrasting architectural vs. rank constraints) provides the necessary theoretical distance from LoRA/PEFT baselines.
- Integration of Local Literature: The new citations (Song et al., Shi et al., Dery et al., Fu et al.) are well-integrated. Specifically, citing Shi et al. (2026) regarding ViT token artifacts (Section 3.2) grounds the bottleneck design in modern VLM theory.
- Logical Consistency: The "useful anti-claim" regarding the targeted loss (Section 4.3) is a strong example of internal ablation. Removing the unnecessary complexity of the targeted loss makes the final method more elegant and easier to reproduce.
- External Validity Framing: The Winoground/Flickr30k analysis (Section 5.2) correctly identifies the "one-image-contrast" vs. "image-pair-contrast" gap, turning a negative result into a clear roadmap for future research.

### 4. Weaknesses
- MINOR: Figure 1 Legend/Clarity. While the text is strong, Figure 1 still shows the "Selected RRB" as a single point. Given the discussion on seed fragility, a small "cloud" or error bars on that specific point (similar to the detail in Figure 2) would better represent the "tradeoff region" mentioned in the text.
- MINOR: Notation on Scaling. Section 3.1 introduces alpha as a fixed residual gate. While the text mentions alpha=0.05 is simple, a brief footnote or sentence on why a learnable gate was avoided (e.g., to prevent the residual from "taking over" the embedding space during early training) would strengthen the "dominance" argument.
- MINOR: Appendix References. The Appendix (Section A) refers to local file paths (e.g., refine-logs/...). While useful for the current project, these should be generalized or removed if the paper is destined for a public venue, as reviewers will not have access to the local NAS.

### 5. Actionable Fixes (LaTeX Only)
For the Appendix (Section A):
- Fix: Replace the specific local file paths with descriptive text (e.g., "Detailed diagnostic logs and per-category failure reports for all runs are available in the supplementary material.") to ensure the paper remains self-contained for external reviewers.

For the Scaling Argument (Section 3.1):
- Fix: Add a short sentence: "A fixed gate alpha is preferred over a learned parameter to ensure the architectural prior of the 'dominant pretrained path' is maintained throughout the training process, preventing the residual from disproportionately influencing the gradient flow in early epochs."

### 6. Missing References
The bibliography is now very strong. One final optional addition to Section 5.1 (Residual Size) could be:
- Metric Drift: Luo et al. (2026), "KeepLoRA: Continual Learning with Residual Gradient Adaptation" could be cited when discussing how residuals can move examples in directions that damage nearest-neighbor structure, as they also deal with gradient alignment in residual paths.

### 7. Verdict: Ready for submission? Yes
The paper is now a complete, honest, and technically sound contribution. The "Validation Gate" protocol effectively addresses the reliability concerns, and the bottleneck-vs-rank framing clarifies the paper's contribution to the broader PEFT landscape. It is ready for submission to an ACM-style venue.

</details>

### Fresh Round 2 Fixes Implemented

1. Added a Figure 1 caption note that the selected point is a point estimate and seed dispersion is summarized in Figure 2.
2. Clarified why the fixed residual gate is preferred over a learned gate.
3. Cited KeepLoRA in the residual-size discussion.
4. Replaced appendix-local paths with supplementary-artifact language suitable for external reviewers.
5. Recompiled and saved `main_round2.pdf`.

## Fresh Loop Final Verification

- Final PDF: `main.pdf` and `main_round2.pdf`
- Total pages: 11
- Main body before references/appendix: references begin after page 9, satisfying the requested minimum of 8 pages.
- Undefined citations/references: 0
- Overfull hboxes over 10pt: 0
- Remaining overfull hboxes: 3.75403pt, 4.77711pt, and 8.81711pt.
