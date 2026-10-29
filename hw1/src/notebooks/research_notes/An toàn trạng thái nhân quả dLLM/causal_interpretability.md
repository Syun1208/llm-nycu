# Causal & Mechanistic Interpretability for Localizing Safety Behavior in LLMs, dLLMs and Diffusion Models

Scope: 2023–2026 (today 2026-10-19). Venue key: **[A*]** = NeurIPS/ICML/ICLR/ACL/EMNLP/CVPR/ICCV/ECCV/AAAI/IJCAI/KDD/USENIX Sec/S&P/CCS (main conference). "Findings" or workshop papers are not marked A*.
Verification legend: **(fetched)** = details confirmed from the paper page or abstract during this session. **(search-snippet)** = confirmed only through search-result summaries. **(from memory; verify)** = well-known paper whose details and arXiv ID come from prior knowledge and were not re-checked this session.

---

## Q1. Which causal-identification techniques are SOTA for localizing safety behavior, and what are their known failure modes?

### Takeaway
Safety behavior in aligned AR LLMs has been localized at four levels: a residual-stream **direction** (refusal direction, later refined to multi-dimensional cones), a few **attention heads** (Ships/Sahara), sparse **neurons/parameters** (pruning or low-rank), and a contiguous block of middle **"safety layers"**. Each was found with a different causal tool: difference-in-means plus directional ablation, head ablation scoring, pruning attribution, and layer-wise similarity plus parameter scaling. The current SOTA for fine-grained, local explanation pairs activation or attribution patching with SAE feature bases (e.g., LOCA, sparse feature circuits). The known failure modes are:
- subspace-patching illusions;
- dependence on metric and corruption choices;
- non-identifiability: many circuits or abstractions fit the same behavior;
- polysemantic or entangled features: SAE steering hurts capabilities;
- refusal being multi-dimensional, not a single direction;
- first-token refusal proxies that miss harmful content emitted later.

### Cited Findings

**A. Refusal / safety directions (representation-level causal interventions)**
- *Refusal in Language Models Is Mediated by a Single Direction* (Arditi, Obeso, Syed, Paleka, Panickssery, Gurnee, Nanda), **NeurIPS 2024 [A*]**.
  - Method: compute a difference-in-means direction between harmful and harmless instruction activations at post-instruction token positions. Then (i) project the direction out of every layer and position ("directional ablation"), which removes refusal, or (ii) add it, which induces refusal on harmless prompts. Weight orthogonalization yields a white-box jailbreak.
  - Scope: 13 open chat models up to 72B. The paper also shows that adversarial suffixes suppress propagation of the refusal direction.
  - Stated limits: covers only open models and refusal behavior; the direction-selection heuristic is ad hoc; it does not explain how the direction is computed upstream; ablation causes small capability changes.
  - Source: [arXiv 2406.11717](https://arxiv.org/abs/2406.11717) (from memory; verify). It is referenced as the canonical difference-in-means refusal-direction method in [LOCA](https://arxiv.org/html/2605.00123v1) (fetched).
- *The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence* (Wollschläger et al., 2025).
  - Finding: a gradient-based method finds **several** refusal directions that form **multi-dimensional concept cones**. The paper introduces "representational independence," because orthogonality alone does not imply independence under intervention.
  - Implication: this directly challenges the single-direction picture.
  - Venue: I could not confirm it from the search results; it is reportedly ICML 2025 (verify).
  - Source: [arXiv 2502.17420](https://arxiv.org/pdf/2502.17420) (search-snippet).
- Follow-ups extend the multi-dimensional view: *Over-Refusal and Representation Subspaces: A Mechanistic Analysis of Task-Conditioned Refusal in Aligned LLMs* ([arXiv 2603.27518](https://arxiv.org/html/2603.27518), preprint, search-snippet) and the Alignment Forum/LessWrong post "Exploring the multi-dimensional refusal subspace in reasoning models" ([LessWrong](https://www.lesswrong.com/posts/ixJrmYrgHM4TenN7D/exploring-the-multi-dimensional-refusal-subspace-in-1), secondary source).
- *LLMs encode harmfulness and refusal separately* (Zhao et al., 2025). It argues that harmfulness and refusal are distinct representations, so a refusal direction is not the same thing as a "harm detector." It is cited in [LOCA](https://arxiv.org/html/2605.00123v1) (fetched citation only; venue and details unverified).
- *Beyond Surface Alignment: Rebuilding LLMs Safety Mechanism via Probabilistically Ablating Refusal Direction* uses refusal-direction ablation as a worst-case simulation for robust training. Source: [arXiv 2509.15202](https://arxiv.org/pdf/2509.15202) (search-snippet; venue unknown).

**B. Safety attention heads**
- *On the Role of Attention Heads in Large Language Model Safety* (Zhou et al.), **ICLR 2025 [A*], Oral**.
  - Method: **Ships** (Safety Head ImPortant Score) scores how much ablating each head changes refusal on harmful queries, generalized to the dataset level. **Sahara** is a search algorithm that attributes the critical safety heads.
  - Findings: ablating **one** safety head in Llama-2-7b-chat makes it answer **16× more harmful queries** while modifying only **0.006%** of parameters, compared with about 5% in prior work. Heads act mainly as **feature extractors** for safety. Models fine-tuned from the same base share overlapping safety heads.
  - Sources: [arXiv 2410.13708](https://arxiv.org/pdf/2410.13708), [ICLR oral page](https://www.iclr.cc/virtual/2025/oral/31798) (search-snippet). Code: github.com/ydyjya/SafetyHeadAttribution.

**C. Safety layers**
- *Safety Layers in Aligned Large Language Models: The Key to LLM Security* (Shen Li, Liuyi Yao, Lan Zhang, Yaliang Li), **ICLR 2025 [A*]**.
  - Method: analyze how input vectors for harmful vs. normal queries diverge across layers, then use the over-rejection phenomenon and parameter scaling to pinpoint a small contiguous block of **middle layers** that distinguish malicious from normal queries.
  - Defense: **SPPFT** freezes these layers during fine-tuning to preserve safety.
  - Note: earlier arXiv versions had a different title.
  - Sources: [arXiv 2408.17003](https://arxiv.org/html/2408.17003v3), [ICLR page](https://www.iclr.cc/virtual/2025/poster/28578) (search-snippet).

**D. Safety-critical neurons and parameters**
- *Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications* (Wei et al.), **ICML 2024 [A*]**. Method: set-difference attribution, i.e., parameters important for safety but not for utility. Finding: about 3% of parameters (neuron level) and about 2.5% of rank are safety-critical; removing them breaks safety while keeping utility. Source: [arXiv 2402.05162](https://arxiv.org/abs/2402.05162) (from memory; verify).
- *Diffusion LLMs as Targets and Adversaries: Mechanistic Safety Exploits* (Dumitrescu, Decouchant, Lek, Chen), arXiv preprint, Aug 2026.
  - Method: FFN safety neurons are found from activations at the **prompt-encoding step only**, using a benign-vs-malicious logistic-regression probe and keeping the top 0.8–3% of neurons.
  - Findings: self-pruning raises attack success rate (ASR) on **LLaDA from 2.6% to 73.8%** and on **Dream from 1.9% to 86.6%**. AR-initialized dLLMs inherit their source model's safety neurons: about **41.8%** overlap between Fast-dLLM and Qwen2.5, and about **30.9%** for Dream. LLaDA (trained from scratch) has 0% overlap with Llama. Transfer pruning raises Dream ASR to 73.2%.
  - Limits: transfer requires identical architecture; results rely on automatic judges; the "Jailbreak Zone" may be a dataset artifact; gradient (Gumbel-softmax) token optimization failed.
  - Source: [arXiv 2608.07430](https://arxiv.org/html/2608.07430) (fetched).
- *NeuroStrike* (Wu et al., 2026a), a safety-neuron pruning attack on AR models, is cited by the paper above (citation only; not verified).

**E. Patching-based causal localization (activation, path and attribution patching; SAE circuits)**
- Foundational toolkit:
  - Causal tracing / ROME (Meng et al., NeurIPS 2022), [arXiv 2202.05262](https://arxiv.org/abs/2202.05262) (from memory).
  - Path patching (Goldowsky-Dill et al. 2023), [arXiv 2304.05969](https://arxiv.org/abs/2304.05969) (from memory).
  - ACDC automated circuit discovery (Conmy et al., **NeurIPS 2023 [A*]**), [arXiv 2304.14997](https://arxiv.org/abs/2304.14997) (from memory).
  - Attribution patching / AtP* (Kramár et al. 2024), [arXiv 2403.00745](https://arxiv.org/abs/2403.00745) (from memory).
  - Edge attribution patching (Syed et al. 2023), [arXiv 2310.10348](https://arxiv.org/abs/2310.10348) (from memory).
  - *Sparse Feature Circuits* (Marks et al., **ICLR 2025 [A*]**), [arXiv 2403.19647](https://arxiv.org/abs/2403.19647) (from memory).
- **Distributed alignment search (DAS) / causal abstraction**: the interchange-intervention framework (Geiger et al.), [arXiv 2301.04709](https://arxiv.org/abs/2301.04709) (from memory); Boundless DAS (Wu et al., **NeurIPS 2023 [A*]**), [arXiv 2305.08809](https://arxiv.org/abs/2305.08809) (from memory). I found no safety-specific DAS paper in this session (see Gaps).
- *LOCA: Minimal, Local, Causal Explanations for Jailbreak Success in LLMs* (Kumar & Ahuja, UIUC), arXiv preprint, Apr 2026.
  - Method: tokens are matched between the harmful prompt and its jailbreak. Activations are then patched **only along SAE decoder directions**, scoring each (token, direction) pair with a token-specific first-order approximation, recomputed iteratively. The objective is KL divergence on the **first output token**.
  - Findings: refusal is restored with about **6–8 patches (Llama-3.1-8B)** and **12–16 (Gemma-2-2B)**, whereas the baselines of Lee et al. 2025 and Yeo et al. 2025 often fail even after 20 patches. Early-layer patches fall on instruction tokens; middle-layer patches fall on **post-instruction template tokens and punctuation**. Jailbreaks work by strengthening "harmless" concepts at those positions.
  - Limits: the first-token refusal proxy fails when outputs share a first token; the HarmBench grader makes errors; early-layer explanations are hard to interpret; SAE directions may be suboptimal; the method is slower.
  - Source: [arXiv 2605.00123](https://arxiv.org/html/2605.00123v1) (fetched).
- *Circuit Discovery Helps Detect LLM Jailbreaking: A Mechanistic Interpretability Study* (arXiv 2608.27504, 2026).
  - Method: edge attribution patching plus subnetwork probing on Llama-2-chat, to find circuits that produce affirmative ("Sure") first tokens.
  - Finding: **ablating these circuits at the first-token prediction cuts ASR by up to 80%**.
  - Caveats: the first-token proxy can miss harmful content later in the response. Venue and authors were not verified.
  - Source: [arXiv 2608.27504](https://arxiv.org/html/2608.27504) (search-snippet).
- *Understanding Refusal in Language Models with Sparse Autoencoders* (Yeo et al.), **Findings of EMNLP 2025** (not main-track A*). Cited as a baseline in [LOCA](https://arxiv.org/html/2605.00123v1) (fetched citation).
- *Surgical Activation Steering via Generative Causal Mediation* (2026 preprint) performs head-level causal mediation for steering in LMs. Source: [arXiv 2602.16080](https://arxiv.org/html/2602.16080v1) (search-snippet only).

**F. SAE features for safety**
- *Steering Language Model Refusal with Sparse Autoencoders* (O'Brien et al., Microsoft), **ICML 2025 [A*]**. Method: clamp or amplify SAE refusal features in Phi-3 Mini at inference time. Finding: this improves jailbreak robustness, including multi-turn attacks, **but degrades general benchmark performance**. The authors conclude that the features are "more deeply entangled with general capabilities than previously understood." Sources: [ICML page](https://icml.cc/virtual/2025/50928), [arXiv 2411.11296](https://arxiv.org/html/2411.11296v2) (search-snippet).
- *Graph-Regularized Sparse Autoencoders for LLM Safety Steering* (**ICML 2026 [A*]** listing) adds a co-activation-graph smoothness term and claims better selective refusal on JailbreakBench, HarmBench and XSTest. Source: [ICML 2026](https://icml.cc/virtual/2026/76972) (search-snippet; claims unverified).

**G. Methodological failure modes of patching**
- *Is This the Subspace You Are Looking For? An Interpretability Illusion for Subspace Activation Patching* (Makelov, Lange, Nanda), **ICLR 2024 [A*]**.
  - Finding: a 1-D subspace patch can change outputs by activating a **dormant parallel pathway** that is causally disconnected from normal computation. This is the "MLP-in-the-middle" illusion, shown on IOI and linked to rank-1 fact editing.
  - Implication: a causal effect from a subspace patch is not proof that the subspace encodes the feature.
  - Sources: [mlanthology](https://mlanthology.org/iclr/2024/makelov2024iclr-subspace), [arXiv 2311.17030](https://ar5iv.labs.arxiv.org/html/2311.17030) (search-snippet).
- *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods* (Zhang & Nanda), **ICLR 2024 [A*]**. Finding: localization results change materially with the corruption method (Gaussian noise vs. symmetric token replacement) and with the metric (logit difference vs. probability). Gaussian-noise corruption can create **OOD activations**. Source: [arXiv 2309.16042](https://arxiv.org/abs/2309.16042) (from memory; verify).
- *How to Use and Interpret Activation Patching* (Heimersheim & Nanda, 2024, tutorial/preprint) covers denoising vs. noising, and backup/self-repair behavior that hides component importance. Source: [arXiv 2404.15255](https://arxiv.org/abs/2404.15255) (from memory).
- *Everything, Everywhere, All at Once: Is Mechanistic Interpretability Identifiable?* (Méloux et al., ICLR 2025 [A*], from memory; verify). Finding: many distinct circuits and algorithmic interpretations fit the same behavior, so mechanistic explanations are **non-identifiable**. Source: [arXiv 2502.20914](https://arxiv.org/abs/2502.20914) (from memory).
- *PatchBench: Measuring Collateral Damage in Activation Patching* (arXiv 2610.10276, Oct 2026) is a new benchmark on side effects of patching. Source: [arXiv 2610.10276](https://arxiv.org/html/2610.10276v1) (search-snippet; title only verified).

### Inferences
- The "location" of safety depends on the method, and the results are not mutually consistent. Direction-level results (one direction, or a cone), head-level results (one head gives 16×), layer-level results (middle block) and neuron-level results (about 3%) all show **sparsity/fragility**, but they are measured with different interventions and metrics. None of them establishes a unique mechanism, consistent with the non-identifiability results.
- Most safety localization uses a **first-token or refusal-string metric**: Arditi's refusal score, LOCA's first-token KL, and the circuit paper's "Sure" token. That is itself a temporal simplification (only t = 0 of generation), and several papers flag it as a limitation.
- Ablation-based scores (Ships, pruning) measure **necessity**. Steering (adding a direction) measures **sufficiency**. Few papers report both on the same component. Self-repair and backup heads (Heimersheim & Nanda) mean necessity scores can understate importance.
- SAE-based safety features trade interpretability for capability damage, which is evidence of polysemantic or entangled features (O'Brien et al.).

### Gaps
- I did not find in this session a **safety-specific causal abstraction / DAS / interchange-intervention study** (aligning a high-level "harmful → refuse" causal model to LLM internals). This looks like a genuine gap, or one my search missed.
- The venue of Wollschläger et al. (ICML 2025?) and of Zhao et al. ("harmfulness vs. refusal") was not confirmed.
- Specific "safety-critical neurons" papers (e.g., Chen et al., *Finding Safety Neurons in LLMs*, arXiv 2406.14144) were not re-verified this session.
- The quantitative limitations sections of Arditi et al. and Wei et al. were not re-fetched; the limits above come from memory.

---

## Q2. Has anyone applied causal identification across the TIME / denoising dimension (diffusion timesteps, or AR generation steps) to locate safety-critical states? What is missing?

### Takeaway
**Partially, and mostly behaviorally rather than mechanistically.**
- **AR LLMs:** "shallow safety alignment" (Qi et al., ICLR 2025) shows that safety is concentrated in the first few output tokens. Circuit and patching work also localizes jailbreaks at the first generated token.
- **Masked dLLMs (LLaDA/Dream):** a 2026 cluster of preprints shows a **"shallow-step" / early-commitment** effect: safety is decided in roughly the first 8–16 of 64 denoising steps. The evidence comes from **interventions on token commitments** (prefill and forced-token experiments), not from activation patching on internal states.
- **Image diffusion:** causal tracing (Basu et al., ICLR 2024 / ICML 2024) localizes concepts in the text encoder and UNet layers, and some safety editors intervene only at causally relevant timestep windows (NeurIPS 2024).

**Missing:** I found no paper doing **activation patching or causal mediation over the (layer × denoising step × position) grid** of a dLLM to locate the internal states that determine refusal. Such a study would amount to "causal tracing across timesteps" for dLLM safety.

### Cited Findings

**AR: temporal (token-depth) localization**
- *Safety Alignment Should Be Made More Than Just a Few Tokens Deep* (Qi et al.), **ICLR 2025 [A*], Outstanding Paper**.
  - Method: per-token KL divergence between aligned and base models, plus prefilling, decoding-parameter and fine-tuning attacks.
  - Finding: alignment mainly changes the distribution over the **first few output tokens** ("shallow safety alignment"), which explains prefilling, adversarial-suffix and fine-tuning attacks.
  - Fixes: (i) data augmentation with "safety recovery" examples (harmful prefix → recover to refusal); (ii) a token-wise constrained fine-tuning objective that limits drift on the initial tokens.
  - Source: [arXiv 2406.05946](https://arxiv.org/abs/2406.05946) (from memory; verify). It is cited as the AR basis for dLLM work in [Beyond Token Positions](https://arxiv.org/html/2609.00495v1) (fetched).
- Jailbreak circuits are localized at the first-token prediction: ablating them cuts ASR by up to 80%. Source: [arXiv 2608.27504](https://arxiv.org/html/2608.27504) (search-snippet).
- LOCA localizes jailbreak success to **prompt-side positions**: post-instruction template tokens and punctuation in middle layers. Source: [arXiv 2605.00123](https://arxiv.org/html/2605.00123v1) (fetched).

**Diffusion LLMs: step-wise safety**
- *Beyond Token Positions: Safety Alignment Across Denoising Steps in Diffusion Language Models* (Guoli Wang, Haonan Shi, Tu Ouyang, An Wang; CWRU), arXiv 2609.00495, 31 Aug 2026, no venue.
  - Models: LLaDA-8B and Dream-7B (instruct and base).
  - Method: prefill interventions at varied response positions, forcing a single refusal token at a given denoising step, and tracking refusal-token probability mass across steps. **No activation patching or probing.**
  - Findings:
    - **"Shallow-step alignment":** early-step commitments govern safety even at non-leading positions. A compliance prefill at the 1% position raises LLaDA-Instruct ASR from **1.29% to 93.93%**.
    - Forcing refusal after about **step 8** gives little benefit on base models; on instruct models, ASR returns toward baseline by steps 16–32.
    - Refusal signals appear in 98% of *unsafe* LLaDA traces too. What separates safe from unsafe traces is **persistence/commitment**: safe traces commit refusal tokens 98% of the time vs. 4.10% for unsafe ones, and sustain them longer (21.28 vs. 7.41 steps).
  - Defense: **RAEC** (Refusal-Aware Early Commitment), a training-free decoding rule. It lowers ASR (e.g., PAIR on LLaDA from 48% to 36%) with utility loss within about 0.5 points, though StrongREJECT ASR rose slightly (1.28% to 1.60%).
  - Stated limits: vanilla dLLMs only; fine-tuning dynamics across steps are not studied; human and Beaver-Dam validation covers only LLaDA on JBB/DIJA-JBB; defaults were partly chosen to match the earlier analysis.
  - Source: [arXiv 2609.00495](https://arxiv.org/html/2609.00495v1) (fetched).
- *Re-Mask and Redirect: Exploiting Denoising Irreversibility in Diffusion Language Models* (arXiv 2604.08557). Finding: safety-aligned dLLMs commit refusal tokens with high confidence in the **first 8–16 of 64 steps**. LLaDA commits about **8.5** refusal tokens by step 16 and Dream about **3.1** ("Early Commitment Effect"). Re-masking those tokens undoes safety. Source: [arXiv 2604.08557](https://arxiv.org/html/2604.08557v1) (search-snippet).
- Other dLLM safety papers (attack/defense, mostly non-mechanistic), all arXiv:
  - *Jailbreaking Large Language Diffusion Models* (PAD), [2507.19227](https://arxiv.org/pdf/2507.19227). Fewer steps (more parallel tokens) lowers attack success by up to about 10%, but this is unsuitable as a defense.
  - *The Devil Behind the Mask* (DIJA), cited in [2609.00495](https://arxiv.org/html/2609.00495v1).
  - *A2D: Any-Order, Any-Step Safety Alignment for dLLMs*, [2509.23286](https://arxiv.org/pdf/2509.23286).
  - *Adaptive Steering and Remasking for Safe Generation in dLLMs*, [2605.13043](https://arxiv.org/html/2605.13043). A detector flags unsafe trajectories after the first block and remasks them.
  - *The Safety-Aware Denoiser for Text Diffusion Models*, [2605.08116](https://arxiv.org/pdf/2605.08116).
  - *MaskForge*, [2606.04027](https://arxiv.org/pdf/2606.04027).
  - *Where to Start Alignment? dLLM May Demand a Distinct Position* (MOSA, 2026), cited in [2609.00495](https://arxiv.org/html/2609.00495v1).
  - *Toward Safer dLLMs: Priming Vulnerability* (Yamabe & Sakuma 2025), cited in [2609.00495](https://arxiv.org/html/2609.00495v1).
  - DiffuGuard (stochastic annealing remasking), seen in search snippets only.
- **Safety neurons in dLLMs are measured only at the prompt-encoding step.** Later denoising steps are explicitly ignored. Source: [arXiv 2608.07430](https://arxiv.org/html/2608.07430) (fetched).

**Diffusion LLMs: general (non-safety) mechanistic interpretability across steps**
- *DLM-Scope: Mechanistic Interpretability of Diffusion Language Models via Sparse Autoencoders* (Xu Wang et al., HKU / Alibaba Tongyi), arXiv 2602.05859, Feb 2026.
  - Setup: Top-K SAEs (16K width) at 6 residual layers of Dream-7B and LLaDA-8B, with Mask-SAE and Unmask-SAE variants.
  - Findings: steering during denoising often **outperforms** single-pass LLM steering, especially in deep layers. Inserting an SAE can even *reduce* masked-token loss in early layers. Base SAEs transfer to the instruct model except at the deepest layer.
  - Limits: **no safety evaluation**; decoding-order signals are only **correlational**; covers 2 models.
  - Source: [arXiv 2602.05859](https://arxiv.org/html/2602.05859v1) (fetched).
- *Steering Without Breaking: Mechanistically Informed Interventions for Discrete Diffusion Language Models* (Hanhan Zhou, Shamik Roy, Rashmi Gangadharaiah; AWS), arXiv 2605.10971, v2 Oct 2026.
  - Method: TopK SAEs on MDLM, SEDD, LLaDA and Dream across masking rates, plus a "diffusion logit lens" and per-step feature tracking.
  - Findings: attributes **commit at different steps**. On MDLM, topic reaches 50% emergence at about 2% of the trajectory and sentiment at about 23%. **LLaDA's emergence is gradual; Dream defers emergence to the final quarter**, which the authors attribute to its AR initialization. No fixed steering schedule is optimal.
  - Limits: the theory's assumptions are not formally verified; no safety experiments are visible.
  - Source: [arXiv 2605.10971](https://arxiv.org/html/2605.10971) (fetched, partial).
- *Answer First, Reason Later* (arXiv 2608.05687). LLaDA-1.5 and Dream commit the scaffolded answer at a median of about 2 steps. The authors caution that **visible commit order does not reveal the order of latent computation**. Source: [arXiv 2608.05687](https://arxiv.org/html/2608.05687v2) (search-snippet).
- *Subliminal Clocks: Latent Time Modelling in Diffusion Language Models*, [arXiv 2607.01774](https://arxiv.org/pdf/2607.01774), and *The Path Matters: token-commitment policy*, [arXiv 2605.24697](https://arxiv.org/html/2605.24697). Titles only (search-snippet).

**Image diffusion: transferable analogues**
- *Localizing and Editing Knowledge in Text-to-Image Generative Models* (Basu, Zhao, Morariu, Feizi, Manjunatha), **ICLR 2024 [A*]**.
  - Method: causal mediation analysis / causal tracing on Stable Diffusion's UNet and text encoder, across visual attributes (objects, style, color, action, …).
  - Findings: in the CLIP text encoder there is a **single causal state**, the **first self-attention layer at the last subject token**, unlike LLMs where causal states sit in mid-layer MLPs. In the UNet, knowledge is **distributed** across attribute-dependent sets of components.
  - Editing: DiffQuickFix, a closed-form edit for concept removal.
  - Sources: [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4bfcebedf7a2967c410b64670f27f904-Abstract-Conference.html), [slides](https://iclr.cc/media/iclr-2024/Slides/18667.pdf) (search-snippet).
- *On Mechanistic Knowledge Localization in Text-to-Image Generative Models* (Basu et al.), **ICML 2024 [A*]**.
  - Finding: causal tracing **fails to localize** knowledge in newer models (SD-XL, DeepFloyd).
  - Method: **LocoGen** measures the direct effect of intervening on UNet cross-attention layers. LocoEdit is a closed-form editor.
  - Lesson: classical causal tracing does not transfer across architectures.
  - Sources: [PMLR v235](https://proceedings.mlr.press/v235/basu24b.html), [arXiv 2405.01008](https://arxiv.org/abs/2405.01008v1), [code](https://github.com/samyadeepbasu/LocoGen) (search-snippet).
- *Towards Safe Concept Transfer of Multi-Modal Diffusion via Causal Representation Editing*, **NeurIPS 2024 [A*]**. Method: intervene with unsafe-concept steering vectors **only at diffusion timesteps causally linked to the unsafe concept**, found by an "assess-with-exclusion" procedure. This is one of the few explicit **timestep-wise causal** safety methods. Source: [NeurIPS 2024 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/1741917e3df34daa1a4c564e2980bb59-Paper-Conference.pdf) (search-snippet; author list not verified).
- Other related work:
  - *Detect-and-Guide* optimizes guideline-token cross-attention maps per timestep for nudity detection and guidance. [arXiv 2503.15197](https://arxiv.org/html/2503.15197) (search-snippet).
  - *Concept Replacer* localizes sensitive concepts via attention. [arXiv 2412.01244](https://arxiv.org/pdf/2412.01244) (search-snippet).
  - *Attention, May I Have Your Decision? Localizing Generative Choices in Diffusion Models*. [arXiv 2604.06052](https://arxiv.org/html/2604.06052) (title only).
  - *ForceForget* (2026) notes that cross-attention-only erasure is bypassed by implicit adversarial prompts. [arXiv 2606.14351](https://arxiv.org/pdf/2606.14351) (search-snippet).
  - Classic concept-erasure baselines (not causal localization per se): ESD (Gandikota et al., **ICCV 2023 [A*]**, [arXiv 2303.07345](https://arxiv.org/abs/2303.07345)) and UCE (WACV 2024, [arXiv 2308.14761](https://arxiv.org/abs/2308.14761)) (from memory).

### Inferences
- **Time is the missing causal axis.**
  - AR work localizes over layers × heads × prompt positions, and evaluates at the *first generated* token.
  - dLLM work identifies *when* safety is decided (steps ≤ 8–16 of 64) only through **output-token interventions**: prefill, forced tokens, re-masking.
  - dLLM safety-neuron work uses *only the prompt-encoding step*.
  - Nobody (as far as I found) runs denoising-step-indexed activation patching (clean = refusing trajectory, corrupt = compliant/jailbroken trajectory) to find *which internal (layer, step, position) states* are causally sufficient or necessary for refusal.
- Such a study is directly analogous to Basu et al.'s causal tracing in SD, but along denoising steps, and to the NeurIPS 2024 timestep-window causal editing for images.
- dLLM-specific confounds for any such study:
  - Commitment order ≠ latent computation order (Answer First, Reason Later).
  - Model-dependent emergence: Dream (AR-initialized) emerges late, LLaDA gradually (Steering Without Breaking).
  - Inherited safety neurons from the AR parent (Dumitrescu et al.).
  - Patching a state at step *t* also changes which tokens get unmasked, so effects compound over subsequent steps. This is a path-dependence problem that has no analogue in single-pass patching.
- Safety in dLLMs appears to be "**shallow in steps**" rather than only "shallow in positions." This is the dLLM analogue of Qi et al., and suggests a "safety recovery across steps" training augmentation analogous to Qi's.

### Gaps
- I found no causal-mediation or patching study of dLLM refusal across denoising steps. I also found no attribution-patching or SAE-circuit study of LLaDA/Dream safety.
- I found no study testing whether Arditi's refusal direction exists or transfers in dLLMs and how its projection evolves over denoising steps (not found in this session's searches; might exist).
- For AR, I found no systematic generation-step-wise patching of mid-response "safety recovery" states (e.g., when a model recovers to refusal after a harmful prefix). Qi et al. analyze this only at the level of the output distribution.
- Authors of the NeurIPS 2024 causal representation editing paper and of the circuit-discovery jailbreak paper (2608.27504) were not verified.

---

## Q3. What open problems are explicitly listed?

### Takeaway
Explicitly stated open problems cluster around:
1. multi-dimensional or non-independent refusal geometry;
2. safety features entangled with capabilities;
3. reliance on first-token or automated-judge proxies;
4. localization methods that do not transfer across architectures (SD-XL, AR → dLLM);
5. dLLM-specific gaps: fine-tuning dynamics across steps, correlational-only decoding-order evidence, only vanilla architectures, prompt-step-only neuron analysis.

### Cited Findings
- **Refusal is not one direction:** multiple directions form cones; orthogonal ≠ independent under intervention. Source: [Wollschläger et al., arXiv 2502.17420](https://arxiv.org/pdf/2502.17420).
- **SAE safety features entangle with capabilities:** steering refusal features hurts benchmarks. Source: [O'Brien et al., ICML 2025](https://icml.cc/virtual/2025/50928).
- **First-token proxies:** LOCA fails when the original and jailbroken responses share a first token. The circuit-discovery paper warns that harmful content can come later in the response. Sources: [LOCA](https://arxiv.org/html/2605.00123v1); [2608.27504](https://arxiv.org/html/2608.27504).
- **Subspace patching can be illusory** (dormant pathways), so a causal effect is not sufficient evidence. Source: [Makelov et al., ICLR 2024](https://mlanthology.org/iclr/2024/makelov2024iclr-subspace).
- **Causal tracing fails on newer diffusion architectures** (SD-XL, DeepFloyd), so localization methods are architecture-specific. Source: [Basu et al., ICML 2024](https://proceedings.mlr.press/v235/basu24b.html).
- **dLLM step-wise safety** (stated in Beyond Token Positions): fine-tuning dynamics across steps are not studied; only vanilla architectures are covered; human validation is limited. Source: [arXiv 2609.00495](https://arxiv.org/html/2609.00495v1).
- **dLLM SAEs:** decoding-order evidence is correlational, not causal; only 2 models; transfer breaks at the deepest layer. Source: [DLM-Scope](https://arxiv.org/html/2602.05859v1).
- **dLLM steering theory:** assumptions are "plausible but not formally verified"; perplexity is unreliable for LLaDA. Source: [Steering Without Breaking](https://arxiv.org/html/2605.10971).
- **dLLM safety neurons:** transfer requires identical architectures; the "Jailbreak Zone" may be a benchmark artifact; gradient-based discrete optimization failed. Source: [arXiv 2608.07430](https://arxiv.org/html/2608.07430).
- **Shallow alignment remains:** Qi et al. propose deepening alignment beyond the first tokens as an open direction (data augmentation and a constrained objective are only partial fixes). Source: [arXiv 2406.05946](https://arxiv.org/abs/2406.05946) (from memory).

### Inferences
Open problems emerging across sources, not stated as such by any single one:
1. **Temporal causal localization in dLLMs.** Patch or mediate internal states indexed by denoising step, and separate "decision" states from "commitment" events.
2. **Multi-direction refusal in dLLMs and over time.** Does the refusal cone rotate or move across steps?
3. **Faithfulness metrics beyond the first token** for both AR and dLLM, e.g., full-trajectory harmfulness judged on final text.
4. **Identifiability.** Several incompatible localizations (head vs. layer vs. direction vs. neurons) fit the data. Cross-method agreement tests are rare.
5. **AR → dLLM inheritance.** Do Dream and Fast-dLLM retain the parent model's safety heads and refusal direction, and does LLaDA (trained from scratch) develop different mechanisms?

### Gaps
- I did not access full "Limitations" sections for Arditi et al., the Ships paper, or the Safety Layers paper this session, so their explicitly stated open problems are not quoted here.
- Search-snippet items, flagged above, should be spot-checked before final citation. This applies especially to 2026 arXiv IDs (2604–2610.xxxxx), which are consistent with today's date but were not individually opened except 2609.00495, 2608.07430, 2605.00123, 2605.10971 and 2602.05859.
