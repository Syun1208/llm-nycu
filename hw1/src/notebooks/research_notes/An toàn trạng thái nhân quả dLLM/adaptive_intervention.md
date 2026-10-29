# Adaptive / Conditional Inference-Time Intervention for LLM Safety, and Transfer to Diffusion LLMs and Diffusion Models

Scope: 2023–2026 (as of 2026-10-19). A* venues are marked **[A*]**. Items marked **(verified)** were checked on a primary page (arXiv abstract, proceedings, or OpenReview/virtual page) during this research session. Items marked **(from prior knowledge)** are well-known papers I did not re-fetch in this session; their arXiv IDs and headline facts are given to the best of my knowledge and should be spot-checked before final citation. Items marked **(uncertain)** have shaky details.

---

## Q1. Which methods decide intervention timing/location adaptively, on what signal, and is any of it causal rather than correlational?

### Takeaway
The field has moved from **static, always-on steering** (ITI, ActAdd, CAA, RepE) to **gated steering**, where a linear probe or projection on hidden states decides *whether* to steer (CAST, SafeSwitch) and *how strongly* (AdaSteer, SADI, BarrierSteer, weighted-controller methods). In almost every case the gating signal is **correlational**: a linear probe or projection onto a contrastive mean-difference direction. "Causal" evidence is mostly limited to (a) choosing layers/heads by intervention effect (ITI's head selection, refusal-direction ablation) and (b) a few 2026 preprints. I found **no** method that picks *when* to intervene through formal causal identification (do-calculus, backdoor adjustment, or counterfactual effect estimation at each step).

### Cited Findings

**Static (non-adaptive) baselines — the "what" that adaptive methods gate**
- **ITI: Inference-Time Intervention** (Li et al., **NeurIPS 2023 [A*]**). Linear probes on each attention head's activations rank heads by truthfulness-probe accuracy. A fixed shift along the truthful direction is added to the top-K heads at *every* generated token. On LLaMA-7B fine-tuned (Alpaca), TruthfulQA truthfulness rose from 32.5% to 65.1%. The authors note a truthfulness vs. helpfulness trade-off controlled by intervention strength α and K. *Location is chosen by probe accuracy (correlational), timing is not adaptive.* — [arXiv 2306.03341](https://arxiv.org/abs/2306.03341) (from prior knowledge)
- **ActAdd: Activation Addition** (Turner et al., 2023). The steering vector is the activation difference of a single prompt pair (e.g., "Love" − "Hate"), added at one layer to the residual stream at the first token positions. No optimization. Venue uncertain (later versions were revised). — [arXiv 2308.10248](https://arxiv.org/abs/2308.10248) (from prior knowledge, venue uncertain)
- **CAA: Contrastive Activation Addition** (Rimsky/Panickssery et al., **ACL 2024 [A*]**). The steering vector is the mean residual-stream difference over many A/B multiple-choice contrast pairs (sycophancy, refusal, hallucination, etc.), added at all token positions after the prompt on Llama-2-Chat. It works on top of fine-tuning and system prompts. — [arXiv 2312.06681](https://arxiv.org/abs/2312.06681) (from prior knowledge)
- **RepE: Representation Engineering** (Zou et al., 2023 preprint). This is a top-down framework: LAT (linear artificial tomography) reads concept directions, then "representation control" adds or subtracts them (honesty, harmlessness, power-seeking, etc.). It is the conceptual parent of Circuit Breakers. — [arXiv 2310.01405](https://arxiv.org/abs/2310.01405) (from prior knowledge)
- **Refusal is mediated by a single direction** (Arditi et al., **NeurIPS 2024 [A*]**). A difference-of-means "refusal direction" exists across 13 open chat models. Ablating it (directional ablation) removes refusal, and adding it induces refusal. This is *causal in the interventional sense*: ablation and addition change behavior. — [arXiv 2406.11717](https://arxiv.org/abs/2406.11717) (from prior knowledge)

**Adaptive "whether to intervene" (conditional gating)**
- **CAST: Conditional Activation Steering** (Lee, Padhi, et al., IBM Research and UPenn, **ICLR 2025 [A*]**). The method checks activation patterns at inference time (a "condition vector": the projection of prompt hidden states onto a category direction, compared to a threshold) and applies a refusal steering vector only when the condition fires. This allows rules like "refuse if about hate speech or adult content, otherwise behave normally" without weight changes. Code is open source. — [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e2dd53601de57c773343a7cdf09fae1c-Abstract-Conference.html); [arXiv 2409.05907](https://arxiv.org/abs/2409.05907) (verified venue; the mechanism description is partly from prior knowledge)
- **SafeSwitch** (Han et al., UIUC and Stevens, **EMNLP 2025 Findings** — Findings track, not main conference). A prober on internal states detects harmful intent and activates a lightweight "safety head" (less than 6% of parameters tuned) only when triggered. It reports about 80% fewer harmful outputs on harmful queries while keeping utility, a Pareto-optimal trade-off among compared methods, and "more context-aware refusals". — [arXiv 2502.01042](https://arxiv.org/abs/2502.01042); [ACL Anthology 2025.findings-emnlp.366](https://preview.aclanthology.org/credits/2025.findings-emnlp.366) (verified)

**Adaptive "how strongly" (per-input / per-token strength)**
- **AdaSteer** (HIT et al., **EMNLP 2025 main [A*]**). Training-free. Two empirical laws:
  - **Rejection Law (R-Law):** jailbreak inputs that sit further against the rejection direction need stronger steering.
  - **Harmfulness Law (H-Law):** benign inputs are separated along a harmfulness direction.
  
  Per-input coefficients along both directions come from logistic regression on the projections. Evaluated on LLaMA-3.1, Gemma-2, and Qwen2.5, it beats baselines across jailbreaks with fewer false refusals. **Stated limitations:** only 7–9B models were tested (30B+ is unverified), and it relies on linear steering. — [arXiv 2504.09466](https://arxiv.org/abs/2504.09466); [ACL Anthology 2025.emnlp-main.1248](https://aclanthology.org/2025.emnlp-main.1248.pdf) (verified)
- **SADI: Semantics-Adaptive Dynamic Intervention** (**ICLR 2025 [A*]**). Builds a dynamic steering vector from the input's own activations (a binary mask over key heads, neurons, and hidden states combined with input semantics) instead of using a fixed vector. — [arXiv 2410.12299](https://arxiv.org/abs/2410.12299); [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/c4d26a95fd83f8e590f81c54ae670b5d-Paper-Conference.pdf) (verified that it exists; details from the search snippet)
- **BarrierSteer** (2026 preprint). Uses control barrier functions learned in latent space to give *per-token* adaptive corrections toward a learned safety boundary. An appendix ablation shows that *conditioning fixed-direction steering on a learned detector* beats no defense but does not match BarrierSteer. This suggests the "when" gate alone is not sufficient and the "how" (state-dependent correction) matters. — [arXiv 2602.20102](https://arxiv.org/html/2602.20102v2) (verified via search snippet; venue unknown)
- **Guiding Giants: Lightweight controllers for weighted activation steering** (2025 preprint). A small controller predicts per-layer steering weights from activations. — [arXiv 2505.20309](https://arxiv.org/pdf/2505.20309) (existence only; details not verified)
- **"Knowing without Acting": disentangled geometry of safety mechanisms** (2026 preprint). Uses causal intervention and an *adaptive* closed-form steering with a negative-feedback control loop. This was the closest match I found to "causal + adaptive". — [arXiv 2603.05773](https://arxiv.org/pdf/2603.05773) (verified only via search snippet; uncertain)
- **ALTSTEER** (2026 preprint). Selective safety steering that produces constructive alternatives instead of hard refusals. — [arXiv 2608.30197](https://arxiv.org/pdf/2608.30197) (existence only; details not checked)

**Adaptive in *time* (which generation steps)**
- **SafeDecoding** (Xu et al., **ACL 2024 [A*]**). A LoRA "safety expert" is fine-tuned on refusals. At decoding time, the top tokens of the base and expert models are intersected and re-weighted (amplifying safety disclaimers, attenuating attack-aligned tokens). Crucially, this is applied **only to the first m tokens** (small m, e.g. about 2), then normal decoding resumes, to limit helpfulness loss and latency. *Timing is a fixed heuristic (the first tokens), not adaptive.* — [arXiv 2402.08983](https://arxiv.org/abs/2402.08983) (from prior knowledge)
- **Shallow safety alignment** (Qi et al., "Safety Alignment Should Be Made More Than Just a Few Tokens Deep", **ICLR 2025 [A*]**). Alignment mostly changes the distribution over the first few output tokens. This explains why prefilling and harmful-prefix attacks work, and why "first-token" interventions (SafeDecoding) are both effective and brittle. — [arXiv 2406.05946](https://arxiv.org/abs/2406.05946) (from prior knowledge)

**Training-time representation intervention (for contrast)**
- **Circuit Breakers / Representation Rerouting (RR)** (Zou et al., **NeurIPS 2024 [A*]**). LoRA training reroutes harmful-process representations to an orthogonal or random direction (RepE-style loss) while a retain loss preserves benign representations. It is "always-on" but triggers implicitly, because only harmful representations are affected. Reported large reductions in harmful compliance under diverse attacks with little capability loss (Llama-3-8B-Instruct, Mistral-7B). — [arXiv 2406.04313](https://arxiv.org/abs/2406.04313) (from prior knowledge; exact percentages not re-verified)

**Runtime monitors / early detection (the "trigger" side)**
- **Streaming Content Monitor (SCM)** (Li et al., **NeurIPS 2025 [A*]**). Identifies a training–inference gap: moderators trained on full responses degrade on partial responses. Introduces **FineHarm** (29K prompt–response pairs with token-level labels) and trains with response-level plus token-level labels. Reaches 0.95+ macro-F1 while reading only about the first 18% of response tokens on average, then stops generation early. — [arXiv 2506.09996](https://arxiv.org/abs/2506.09996v3); [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/116186) (verified)
- **Kelp** (2025 preprint). A streaming guardrail that runs in parallel with decoding, using "latent dynamics" of the LLM's intermediate representations. It compares against ShieldHead (per-token classification head) and DSA (disentangled safety adapters). — [arXiv 2510.09694](https://arxiv.org/pdf/2510.09694) (verified existence; numbers not checked)
- **Segment-level coherence for harmful-intent probes** (2026 preprint, with Anthropic-affiliated authors). Streaming linear probes over-rely on a few high-scoring tokens and so false-alarm on benign uses of CBRN terms. The fix is an objective requiring multiple consistent evidence tokens. It also reports that attention and MLP outputs can carry stronger signal than the residual stream. — [alphaxiv 2604.14865](https://www.alphaxiv.org/abs/2604.14865.md) (verified via snippet only)
- **SIREN** (2026). Per-token harmfulness score from safety-relevant neurons, evaluated on prefixes. — [hyper.ai summary of arXiv 2604.18519](https://hyper.ai/en/papers/2604.18519) (aggregator; uncertain)

### Inferences
- The dominant pattern for adaptivity is a **probe-gated intervention**: a linear read-out decides *whether* and *how much*, and a linear write-in changes behavior. CAST (binary gate), AdaSteer (continuous strength), SafeSwitch (gated module), and the dLLM methods below (gated steering plus detector-triggered remasking) are all instances.
- The *write* directions have causal support: ablation and addition of the refusal direction flips behavior (Arditi et al.). The *read* (gating) signals are trained classifiers or projections, so they are correlational. None of the surveyed papers validates that the gating feature is a cause of the harmful output rather than a correlate (e.g., lexical harm cues). The SCM and segment-coherence papers' false-alarm discussion is indirect evidence of this.
- "When" is rarely learned. In AR LLMs, timing is either every token (ITI, CAA), the first m tokens (SafeDecoding), or detector-triggered stopping (SCM). Learned *temporal* schedules exist in image diffusion (SafeGuide, below) but not, as far as I found, in AR LLM steering.

### Gaps
- I found no paper that selects intervention steps or positions through formal causal effect estimation (e.g., per-step average causal effect with counterfactual resampling). This looks like an open niche.
- I could not verify CAST's quantitative numbers (refusal rates per category) in this session.
- Circuit Breakers' exact attack-success numbers were not re-verified.

---

## Q2. What breaks: over-refusal, capability loss, adaptive attackers, latency, calibration?

### Takeaway
Fixed steering has **high per-input variance** (it sometimes steers the wrong way) and **degrades capability as strength grows**. Steering for non-safety goals can **silently weaken safety** because steering vectors overlap with the refusal direction. Gated methods trade these against **probe calibration**: false alarms cause over-refusal, and misses let jailbreaks through. Representation-level defenses (Circuit Breakers) and probe-based monitors are vulnerable to **adaptive or embedding-space attackers**. dLLM defenses show measurable reasoning drops (e.g., GSM8K) and need per-model hyperparameters.

### Cited Findings
- **Reliability:** Tan et al., "Analyzing the Generalization and Reliability of Steering Vectors" (**NeurIPS 2024 [A*]**, venue from prior knowledge). Steerability varies widely per sample. For some datasets, nearly half of inputs move *opposite* to the intended direction. Effectiveness is partly driven by a spurious "steerability bias", and out-of-distribution generalization is imperfect. — [arXiv 2407.12404 overview](https://alphaxiv.org/overview/2407.12404v8)
- **Safety pitfalls of steering:** "Analysing the Safety Pitfalls of Steering Vectors" (2026 preprint). A CAA safety audit finds that steering for other behaviors shifts jailbreak ASR by up to +57% or −50%. The cause is overlap of steering vectors with the refusal direction, framed as a controllability–safety trade-off. — [arXiv 2603.24543](https://www.alphaxiv.org/abs/2603.24543)
- **Mitigation:** "Safety Cost of Steering Vectors Is Separable and Reducible" (2026 preprint). A learned correction (also abbreviated "CAST" — **not** the ICLR 2025 CAST; name collision) removes the safety-degrading component, with no systematic degradation beyond the original vector on MMLU, GSM8K, IFEval, and TriviaQA. — [arXiv 2608.08383](https://arxiv.org/pdf/2608.08383) (excerpt only)
- **Evaluation gap:** 2026 preprint arguing that steering evidence comes from clean controllability tests, not deployment-time reliability. — [arXiv 2603.18329](https://arxiv.org/pdf/2603.18329v1) (snippet only)
- **Capability vs. strength:**
  - ITI reports a truthfulness–helpfulness trade-off as α grows. — [arXiv 2306.03341](https://arxiv.org/abs/2306.03341) (from prior knowledge)
  - AdaSteer is motivated by fixed coefficients giving both weak protection and false refusals on benign prompts. — [arXiv 2504.09466](https://arxiv.org/abs/2504.09466)
  - SafeDecoding restricts intervention to the first m tokens specifically to protect helpfulness and latency. — [arXiv 2402.08983](https://arxiv.org/abs/2402.08983) (from prior knowledge)
- **SAE-feature steering:** O'Brien et al., "Steering Language Model Refusal with Sparse Autoencoders" (2024 preprint). Clamping SAE refusal features in Phi-3 Mini improves jailbreak robustness but degrades general benchmark performance. — [arXiv 2411.11296](https://arxiv.org/abs/2411.11296) (from prior knowledge; **uncertain** on exact findings)
- **Adaptive attackers vs. representation defenses:** Schwinn & Geisler, "Revisiting the Robust Alignment of Circuit Breakers" (2024). Embedding-space attacks reportedly break Circuit Breakers with high success, contrary to the original robustness claims. — [arXiv 2407.15902](https://arxiv.org/abs/2407.15902) (from prior knowledge; **uncertain** on numbers)
- **Adaptive attackers vs. dLLM monitors:** SRI Guard's authors report that adaptive attacks that suppress detectable internal "hesitation" reduce, but do not eliminate, detector effectiveness. — [alphaxiv 2602.02600](https://www.alphaxiv.org/abs/2602.02600)
- **Probe false alarms:** streaming harmful-intent probes rely on a few high-scoring tokens and false-alarm on benign CBRN terms. — [alphaxiv 2604.14865](https://www.alphaxiv.org/abs/2604.14865.md)
- **Latency:**
  - SRI Guard adds about 0.01–0.04% overhead (reported as 150–300× cheaper than external moderators or self-reflection). — [alphaxiv 2602.02600](https://www.alphaxiv.org/abs/2602.02600)
  - In the dLLM adaptive steering+remasking paper, detecting at every block gives a small safety gain (ASR 59.3% → 52.6% on Dream/DIJA/JBB) at "much higher compute", so they detect once. — [arXiv 2605.13043](https://arxiv.org/html/2605.13043)
- **Capability loss in dLLM steering:** LLaDA-8B GSM8K drops from 82.5 to 73.0 under adaptive steering+remasking. MATH-500 and TruthfulQA are almost unchanged. — [arXiv 2605.13043](https://arxiv.org/html/2605.13043)
- **Over-refusal in dLLM steering:** XSTest over-refusal on LLaDA-8B rises from 11.5 to 13.8, while on Dream it falls from 32.9 to 23.7. — [arXiv 2605.13043](https://arxiv.org/html/2605.13043)
- **Static guidance in image diffusion:** static schedules over-intervene on benign prompts (fidelity loss) and under-intervene on malicious ones. — [SafeGuide, ICLR 2026](https://www.iclr.cc/virtual/2026/10019138)

### Inferences
- The failure modes split along the gate/write decomposition:
  - Gate errors cause over-refusal (false positives) or jailbreak leakage (false negatives).
  - Write errors cause capability loss and unreliable direction effects.
- Adaptive defenses move the attack surface onto the gate. An attacker who keeps the probe score below threshold (in-distribution activations) bypasses the intervention entirely.
- Calibration of gating thresholds is per-model and hand-tuned in every paper examined. For example, the dLLM paper uses different steering and remask thresholds per backbone (7.0/7.0/2.706; 0.12/0.25/0.387). No paper offered principled calibration (e.g., conformal guarantees) for the gate.

### Gaps
- I found no systematic study of **calibration** (ECE, conformal coverage) of safety gating probes under distribution shift or adaptive attack.
- I found no head-to-head adaptive-attack evaluation of CAST or AdaSteer by third parties.

---

## Q3. What has (and has not) transferred to dLLMs' denoising timeline (when to intervene across steps, which masked positions)?

### Takeaway
dLLM safety work (LLaDA, Dream, MMaDA-class models) appeared in 2025–2026 and is converging on three findings:
1. **Early denoising steps and leading response positions decide safety** ("denoising-path dependence").
2. **Remasking gives dLLMs a recovery mechanism AR models lack.**
3. **Diffusion-specific attacks (DIJA: interleaved mask/text prompts) bypass response-level refusal.**

Defenses include remasking-schedule changes (DiffuGuard), token-level [EOS] alignment at any masked position (A2D), early-commitment decoding (RAEC), gated steering restricted to masked positions plus detector-triggered remasking (Lee et al.), trajectory monitors (SRI Guard), and denoiser modification toward "provably safe regions" (SAD). **What is missing:** per-step or per-position *learned* or *causal* scheduling of interventions. Existing choices are heuristics: steer every step with a probe gate, detect once after the first block, or commit early refusals.

### Cited Findings

**Vulnerabilities and dynamics**
- **DIJA: "The Devil behind the Mask"** (Wen et al.). An emergent safety vulnerability of dLLMs: jailbreak prompts become interleaved text–mask templates that exploit bidirectional, parallel decoding and bypass safety alignment. — [arXiv 2507.11097](https://arxiv.org/html/2507.11097v1) (verified existence). Cited as Wen et al. 2026 in [2605.13043](https://arxiv.org/html/2605.13043); the final venue is uncertain.
- **"Jailbreaking Large Language Diffusion Models: Revealing Hidden Safety Flaws in Diffusion-Based Text Generation"** (2025). Reports that classical AR jailbreaks such as GCG do not substantially change LLaDA behavior. — [ResearchGate](https://www.researchgate.net/publication/394048951_Jailbreaking_Large_Language_Diffusion_Models_Revealing_Hidden_Safety_Flaws_in_Diffusion-Based_Text_Generation) (snippet only; uncertain)
- **Step-Wise Refusal Dynamics in AR and Diffusion LMs** (Rahimi et al., Technion/BGU/Haifa; arXiv preprint).
  - dLLMs often recover from harmful intermediate states. Harmful Remasking Rate is 0.81/0.96/0.92 and Full Recovery Rate 0.63–0.73 for LLaDA, Dream, and LLaDA-1.5.
  - With the *same* LLaDA-1.5 weights, switching AR → diffusion sampling raises refusal by 0 points (PAIR) up to 52 points (refusal suppression).
  - Dynamic decoding that commits positions earlier *reduces* recovery.
  - Proposes **SRI** (cosine distance of mean-pooled per-step activations to benign/harmful prototypes) and **SRI Guard**, a roughly 3K-parameter autoencoder trained on benign trajectories that flags anomalies. Average AUROC is 0.920 with last-layer SRI, vs. 0.573 for text signals and 0.584 for static activations. ASR is 0.29 vs. DiffuGuard's 0.45 (PAD) / 0.59 (DIJA).
  - **Limitations:** cross-model confounds, keyword/LLM-judge labels, needs hidden activations and anchors, partial vulnerability to adaptive attacks, and broader testing needed.
  
  — [alphaxiv 2602.02600](https://www.alphaxiv.org/abs/2602.02600) (verified)

**Defenses**
- **DiffuGuard: "How Intrinsic Safety is Lost and Found in Diffusion LLMs"** (Li, Nie, et al., **ICLR 2026 [A*]**). Diagnoses that greedy (confidence-based) remasking creates a harmful bias. Also identifies "Denoising-path Dependence": early tokens largely determine final safety. The training-free defense has two parts: (1) **Stochastic Annealing Remasking**, which injects controlled randomness early and anneals it; (2) **Block-level Audit and Repair**, which uses internal representations to detect risky blocks and remask/regenerate them. ASR falls from 47.9% to 14.7% across 4 dLLMs × 6 jailbreaks, with utility and efficiency preserved. — [arXiv 2509.24296](https://arxiv.org/html/2509.24296v1); [ICLR 2026](https://iclr.cc/virtual/2026/poster/10006479) (verified)
- **A2D: Any-Order, Any-Step Safety Alignment** (Jeung, Yoon, et al., Yonsei, **ICLR 2026 [A*]**). *Training-time*, token-level alignment: under randomized masking, the model learns to emit [EOS] at *any masked position* where harmful content would begin. DIJA ASR drops from over 80% to 1.3% (LLaDA-8B-Instruct) and 0.0% (Dream-v0-Instruct-7B). Thresholding [EOS] probability gives **early rejection**, up to 19.3× faster safe termination. This is the closest thing to a position-level "intervene here" signal in dLLMs. — [arXiv 2509.23286](https://arxiv.org/pdf/2509.23286); [ICLR 2026 proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/956e5427549a82a7472e02adc88360e9-Abstract-Conference.html) (verified)
- **RAEC: "Beyond Token Positions: Safety Alignment Across Denoising Steps in dLLMs"** (Wang, Shi, Ouyang, Wang; **EMNLP 2026 [A*]**, per the arXiv comment). Traces intermediate token distributions under harmful prompts and finds **refusal signals are concentrated in early denoising steps and leading response positions**. The *step* at which refusal tokens are committed, and their *persistence*, matter. **Refusal-Aware Early Commitment (RAEC)** is training-free and commits persistent early refusal signals. It reduces ASR on LLaDA and Dream while largely preserving utility. — [arXiv 2609.00495](https://arxiv.org/abs/2609.00495); code at github.com/Glresearch1/RAEC (verified)
- **Activation Steering for Masked Diffusion LMs** (Shnaidman, …, Levi, Lapid; **ReALM-GEN workshop @ ICLR 2026** — workshop, not main track).
  - A single direction is extracted from contrastive prompt sets with one prompt-only forward pass and applied *globally* to residual-stream activations throughout reverse diffusion.
  - Refusal in several MDLMs is governed by an approximately 1-D subspace.
  - Directions can be extracted from **pre-instruction tokens**, which is generally ineffective in AR models.
  - **Greatest leverage comes from early denoising steps and mid-to-late layers.**
  - Directions transfer EN↔ZH within a bilingual MDLM but **do not transfer to AR architectures**.
  
  Steering can also *remove* refusal, a dual-use risk. — [arXiv 2512.24143](https://arxiv.org/abs/2512.24143) (verified)
- **Adaptive Steering and Remasking for Safe Generation in dLLMs** (Lee, Kim, Han; arXiv 2605.13043, preprint; code at github.com/leeyejin1231/DLM_Steering_Remasking). Training-free. This is the most explicit "when/where" design for dLLMs:
  - **Gate:** a harmful-vs-benign mean-difference direction at layer ℓg (LLaDA-8B: 18), projected from the mean hidden state of committed positions (masked positions at step 1). A clipped linear ramp gives g_t ∈ [0,1], **recomputed every denoising step**.
  - **Steer:** g_t·α·v_steer (α = 1) is added **only at currently masked positions**, at layer ℓs (LLaDA-8B: 25, LLaDA-1.5: 22, Dream: 14). v_steer is built from refusal-vs-compliance responses under random masks (ratios 0.3/0.5/0.7/0.8).
  - **Detect:** a roughly 4K-parameter logistic-regression detector runs **once after the first 32-token block** (AUROC 0.988 LLaDA, 0.992 Dream).
  - **Remask:** if the detector fires, the first block **plus 80% of prompt tokens** is remasked and regenerated under steering.
  - **Results (avg ASR, vanilla → method):** LLaDA-8B JBB 41.4 → 22.7, HarmBench 50.4 → 24.9, StrongReject 47.5 → 25.6. Dream gains are small (e.g., 18.5 → 17.3), and Self-reminder is better there in some settings.
  - **Timing ablation:** detecting at 25% progress gives 59.3% ASR vs. 76% at 100% (Dream/DIJA/JBB), so **earlier detection is better**.
  - **Component ablation (LLaDA-8B/DIJA/JBB):** no prompt masking 82.0%, no detection 84.0%, no steering 73.5%, full method 63.4%.
  - **No explicit limitations section.** Visible weaknesses: GSM8K drop on LLaDA, per-backbone thresholds and layers, single detection point, small Dream gains.
  
  — [arXiv 2605.13043](https://arxiv.org/html/2605.13043) (verified)
- **Safety-Aware Denoiser (SAD)** (Yusuf, Jiang, Park; arXiv 2605.08116, preprint). Inference-time modification of the denoising process that steers the final sample toward "provably safe regions of text space" without retraining. Evaluated on hazard taxonomies, memorization, and jailbreaks. Reports large reductions in unsafe generations while preserving fluency. — [arXiv 2605.08116](https://arxiv.org/abs/2605.08116); code at github.com/ParkLabML/SAD (abstract only verified; details unknown)
- **Priming vulnerability** (Yamabe & Sakuma, 2026), "Toward safer diffusion language models: discovery and mitigation of priming vulnerability". — cited in [arXiv 2605.13043](https://arxiv.org/html/2605.13043) (title only; not fetched)

**Discrete diffusion guidance (general mechanisms, not safety-specific)**
- **Simple Guidance Mechanisms for Discrete Diffusion Models** (Schiff, Sahoo, et al., Cornell/InstaDeep, **ICLR 2025 [A*]**). Derives classifier-free and classifier-based guidance for discrete diffusion. Introduces uniform-noise diffusion, which can revise tokens over multiple passes (relevant to "self-correction"). Tested on genomics, molecules, and discretized images — *not* on safety. — [arXiv 2412.10193](https://arxiv.org/abs/2412.10193); [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6cc31b44d88dce8380d36e81485cd07f-Abstract-Conference.html) (verified)
- **Unlocking Guidance for Discrete State-Space Diffusion and Flow Models** (Nisonoff et al., ICLR 2025). Exact predictor guidance for CTMC-based discrete diffusion. — [arXiv 2406.01572](https://arxiv.org/abs/2406.01572) (from prior knowledge; **uncertain** venue)

### Inferences
- **Consensus on "when":** four independent groups (DiffuGuard, RAEC, Shnaidman et al., Lee et al.) find that the **early denoising steps** carry most safety leverage. Rahimi et al. complicate this: dLLMs *can* recover from harmful intermediate states via remasking, and committing early (dynamic decoding) *reduces* recovery. Together: early commitment of *refusal* helps (RAEC), early commitment of *harm* hurts. That makes timing a causal question about which committed tokens propagate. Nobody has estimated this with interventional, per-step causal effects.
- **"Where":** methods restrict steering to **masked positions** (Lee et al.), emit [EOS] at **any masked position** (A2D), or focus on **leading response positions** (RAEC). Lee et al. also remask **prompt tokens** (80%), and the ablation shows this matters a lot (82.0% vs. 63.4% ASR). No paper scores *individual* masked positions by a causal or attribution criterion for targeted remasking. Selection is by block, by position heuristic, or by confidence.
- **Not transferred yet:**
  - learned per-step schedules (as SafeGuide does for images);
  - per-step classifier or classifier-free *safety* guidance on discrete logits (the Schiff et al. machinery exists but has not been applied to safety, as far as I found);
  - CAST-style categorical conditional refusal in dLLMs;
  - Circuit-Breaker-style representation rerouting for dLLMs;
  - streaming or early-exit monitors trained on token-level labels across denoising steps (SRI Guard is unsupervised, Lee et al. detect once).
- The finding that steering directions do not transfer AR → MDLM (Shnaidman et al.) implies that AR-derived probes and vectors cannot be reused for dLLMs. They must be re-extracted under masking.

### Gaps
- SAD's mechanism ("provably safe regions") and numbers could not be read (PDF not parsable).
- I found no dLLM paper using SAE features for safety steering.
- The DIJA and priming papers' final venues are not confirmed.

---

## Q3b. Image/continuous diffusion: safety guidance and which timesteps matter

### Takeaway
In text-to-image diffusion, safety intervention moved from static negative guidance with a warm-up (SLD) to **detection-gated, region-adaptive guidance** (Detect-and-Guide) and **learned per-prompt temporal schedules** (SafeGuide). The repeated finding is that unsafe semantics appear in **phase-specific, prompt-dependent** windows, and very early steps (layout not yet formed) are ambiguous for detection.

### Cited Findings
- **Classifier-Free Guidance** (Ho & Salimans, 2022). This is the basis of "negative prompt" and safety guidance. — [arXiv 2207.12598](https://arxiv.org/abs/2207.12598) (from prior knowledge)
- **Safe Latent Diffusion (SLD)** (Schramowski et al., **CVPR 2023 [A*]**). Adds a safety-guidance term (pushing away from an "unsafe concept" text embedding) to CFG, with a **warm-up step δ** (no safety guidance in the first steps so that composition forms), element-wise thresholding, and momentum. Introduces the I2P benchmark. *Timing is a fixed hyperparameter.* — [arXiv 2211.05105](https://arxiv.org/abs/2211.05105) (from prior knowledge)
- **Detect-and-Guide (DAG)** (Li, Zhang, Sun, Yang, **CVPR 2025 [A*]**). Optimized "guideline tokens" detect unsafe concepts in noisy latents via refined cross-attention maps. Safety guidance is then applied with **adaptive strength and editing region**. The edit mask applies only where the detection map exceeds 0.01, and editing is skipped when the mask covers almost the whole image, which signals the early phase before objects form. No diffusion fine-tuning is needed. — [CVPR 2025 open access](https://openaccess.thecvf.com/content/CVPR2025/html/Li_Detect-and-Guide_Self-regulation_of_Diffusion_Models_for_Safe_Text-to-Image_Generation_via_CVPR_2025_paper.html); [arXiv 2503.15197](https://arxiv.org/abs/2503.15197) (verified)
- **SafeGuide** (Zhou, Bai, Xu, Ren, **ICLR 2026 [A*]**). Argues that static schedules fail because unsafe semantics are non-uniform, prompt-dependent, and phase-specific. The safety intervention is parameterized as a **bell-shaped guidance schedule over timesteps**, and a **policy network trained with risk-aware RL** chooses when and how strongly per prompt. Reports better safety–quality decoupling than static guidance methods. — [ICLR 2026 virtual](https://www.iclr.cc/virtual/2026/10019138) (verified; no numbers on the page)
- **"Not Every Time and Frequency Need to Be Forgotten in Diffusion Unlearning"** (J. Park & M. Park; ICML 2026 workshop FoGen per comments). Forgetting is uneven across time and frequency. Selectively unlearning along timesteps and frequencies improves both unlearning success and quality, with theory on the forgetting–utility trade-off. — [arXiv 2510.17917](https://arxiv.org/abs/2510.17917) (verified; specific timestep bands not given on the abstract page)
- **Other steering methods for image diffusion:**
  - CASteer: training-free concept steering vectors on hidden representations. — [arXiv 2503.09630](https://arxiv.org/pdf/2503.09630v2)
  - SteerDiff: text-embedding-space adaptor. — [arXiv 2410.02710](https://fugumt.com/fugumt/paper_check/2410.02710v2_enmode)
  - EraseFlow: GFlowNet over denoising trajectories. — [arXiv 2511.00804](https://huggingface.co/papers/2511.00804.md)
  
  (All existence-only, from snippets.)

### Inferences
- Image diffusion has the most mature "when to intervene" literature: fixed warm-up (SLD), then detection-gated (DAG), then learned schedule (SafeGuide). dLLM work is roughly at the "detection-gated" stage. A **learned or causal temporal schedule for dLLM safety is an unfilled gap**.

### Gaps
- I did not verify SAFREE (ICLR 2025, training-free adaptive filtering) or Concept-Steerers/SAE-based image steering details in this session.

---

## Q4. Explicit open problems stated by authors

### Takeaway
The open problems authors state are scale (beyond 7–9B), dependence on linear steering, needing white-box activations, adaptive attacks against internal monitors, labeling cost and false alarms for streaming monitors, broader dLLM architecture coverage, and AR→dLLM non-transferability.

### Cited Findings
- **AdaSteer:** validated only at 7–9B (30B+ open), and limited to linear steering. — [arXiv 2504.09466](https://arxiv.org/abs/2504.09466)
- **SRI Guard:**
  - requires hidden activations and predefined anchors;
  - adaptive attacks suppressing internal hesitation partially succeed;
  - keyword/LLM-judge labeling noise;
  - needs broader testing across architectures, attacks, and deployment settings.
  
  — [alphaxiv 2602.02600](https://www.alphaxiv.org/abs/2602.02600)
- **Activation steering for MDLMs:** safety representations appear architecture-dependent, since directions do not reliably generalize to AR models. — [arXiv 2512.24143](https://arxiv.org/abs/2512.24143)
- **Streaming monitors:** dependence on costly token-level labels (SCM's FineHarm, Kelp) and false alarms on benign sensitive terms. — [arXiv 2506.09996](https://arxiv.org/abs/2506.09996v3); [alphaxiv 2604.14865](https://www.alphaxiv.org/abs/2604.14865.md)
- **Steering evaluation:** evidence comes from clean controllability, not deployment reliability. — [arXiv 2603.18329](https://arxiv.org/pdf/2603.18329v1); steering for non-safety goals can erode safety. — [arXiv 2603.24543](https://www.alphaxiv.org/abs/2603.24543)
- **Static guidance (image):** fixed schedules cannot serve both benign fidelity and malicious suppression. — [SafeGuide](https://www.iclr.cc/virtual/2026/10019138)
- **dLLM adaptive steering+remasking:** no explicit limitations section. Implied issues are per-backbone tuning, a single detection point (per-block detection costs more for a small gain), and weak gains on Dream. — [arXiv 2605.13043](https://arxiv.org/html/2605.13043)

### Inferences (candidate research directions implied by the gaps)
1. **Causal timing for dLLMs:** estimate, per denoising step t and masked position i, the interventional effect of committing, remasking, or steering on final harmfulness. This would replace correlational gates (probe projection) with causal-effect gates. Rahimi et al.'s recovery statistics and RAEC's early-commitment finding motivate it, but neither does causal estimation.
2. **Learned step schedules for dLLMs**, analogous to SafeGuide's RL bell-shaped schedule for images.
3. **Calibrated gating** (e.g., conformal thresholds) to control over-refusal rates with guarantees. No surveyed paper does this.
4. **Adaptive-attack evaluation of gated dLLM defenses.** Attackers can target the gate signal, e.g., DIJA variants optimized to keep first-block detector scores low.
5. **Discrete safety guidance:** apply Schiff et al. / Nisonoff et al. classifier guidance with a harmfulness classifier on masked-position logits, gated by step.

### Gaps
- Most dLLM safety papers are 2025–2026 preprints. Only DiffuGuard and A2D (ICLR 2026) and RAEC (EMNLP 2026, per arXiv comment) have confirmed A* acceptance. Lee et al. (2605.13043), SAD (2605.08116), and SRI/Rahimi (2602.02600) are unreviewed as of this check.
- Name collision: "CAST" refers both to Conditional Activation Steering (ICLR 2025) and to an unrelated 2026 steering-correction method (arXiv 2608.08383). Disambiguate when citing.
