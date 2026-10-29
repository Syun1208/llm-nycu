# Safety, Jailbreaks, and Safety Alignment of Diffusion LLMs (dLLMs), 2024–2026

Compiled 2026-10-19. Venue status comes from arXiv "Comments" fields or official proceedings pages and is marked per paper. Where a paper had no venue listed, it is marked "arXiv-only (as of fetch)". Numbers are the authors' own claims. A* venues are marked **[A*]**.

## Q0. Paper catalog (supports all key questions)

### Takeaway
I found about 20 papers on dLLM-specific safety. Attacks clearly outnumber defenses. Five papers have verified A* acceptances: DIJA (ICLR'26), A2D (ICLR'26), DiffuGuard (ICLR'26), MOSA (AAAI'26 oral), and Energy Landscape (NeurIPS'26, per arXiv journal-ref). Three more were accepted at EMNLP'26: Step-Wise Refusal Dynamics, Beyond Token Positions, and DIVA. Everything else was arXiv-only when I fetched it.

### Cited Findings

#### A. Attacks (inference-time)
1. **DIJA: "The Devil behind the mask: An emergent safety vulnerability of Diffusion LLMs"**
   - **Authors:** Zichen Wen, Jiashu Qu, Zhaorun Chen, Xiaoya Lu, Dongrui Liu, Zhiyuan Liu, Ruixi Wu, Yicun Yang, Xiangqi Jin, Haoyun Xu, Xuyang Liu, Weijia Li, Chaochao Lu, Jing Shao, Conghui He, Linfeng Zhang.
   - **Venue:** **ICLR 2026 [A*]**, arXiv 2507.11097.
   - **Method:** Black-box "interleaved mask-text" prompts. The harmful request stays fully visible, and [MASK] spans plus benign separators force the model to fill content consistent with that request. An attacker LLM (Qwen2.5-7B or GPT-4o) generates the prompts by in-context learning. The authors attribute success to two dLLM properties: bidirectional context modeling and parallel decoding, the latter limiting per-token "on-the-fly" refusal.
   - **Models:** LLaDA-Instruct, LLaDA-1.5, Dream-Instruct, MMaDA-MixCoT, plus code dLLMs (DiffuCoder, Dream-Coder) in the appendix.
   - **Results:** HarmBench keyword-ASR is 96.3/95.8/98.3/97.5% on the four models. JailbreakBench evaluator-ASR is 81/79/90/79%. Keyword-ASR reaches up to 100% on Dream. The abstract claims evaluator-ASR is up to 78.5% above ReNeLLM, and StrongREJECT up to +37.7. DIJA stays effective against Self-reminder and RPO.
   - **Proposed defense:** "Refusal-Aware Denoising Alignment" fine-tunes LLaDA on about 10k DIJA-style prompts. HarmBench ASR-k drops 96.3→33.8 and JBB 95.0→19.0. GPQA falls 32.4→27.5, and BBH, HumanEval, and MMLU drop modestly. The authors call this defense "preliminary".
   - **Analyses:** Short generations tend to refuse while long ones leak detail. Very large mask counts (e.g., 50) reduce effectiveness.
   - **Sources:** [arXiv abs](https://arxiv.org/abs/2507.11097); [HTML v2](https://arxiv.org/html/2507.11097v2); [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/0a0eba34ab2ff40ca2d2843324dcc4ab-Abstract-Conference.html)
2. **PAD: "Jailbreaking Large Language Diffusion Models: Revealing Hidden Safety Flaws in Diffusion-Based Text Generation"**
   - **Authors:** Yuanhe Zhang, Fangzhou Xie, Zhenhong Zhou, Zherui Li, Hao Chen, Kun Wang, Yufei Guo.
   - **Venue:** arXiv-only (2507.19227, no venue listed).
   - **Method:** A PArallel Decoding jailbreak that uses a "Multi-Point Attention Attack" to steer parallel generation toward affirmative or harmful patterns at multiple positions.
   - **Results:** 97% ASR across four LLDMs. The authors also claim LLDMs produce harmful content 2× faster than same-size AR LLMs, and that existing AR jailbreaks are of limited effectiveness on LLDMs because of architectural differences.
   - **Source:** [arXiv](https://arxiv.org/abs/2507.19227)
3. **"Safer by Diffusion, Broken by Context: Diffusion LLM's Safety Blessing and Its Failure Mode"**
   - **Authors:** Zeyuan He, Yupeng Chen, Lang Lin, Yihan Wang, Shenxu Chang, Eric Sommerlade, Philip Torr, Junchi Yu, Adel Bibi, Jialin Yu.
   - **Venue:** arXiv-only (2602.00388, v2 Apr 2026). Another paper's reference list may cite it as "A Fragile Guardrail: …"; this is unverified.
   - **Claims:**
     - dLLMs are intrinsically more robust to AR-designed jailbreaks because of a "stepwise reduction effect". This is stated as a *hypothesis* (Assumption 3.2) and supported by a keyword-based Monte-Carlo "safety distance" trend.
     - GCG reaches 16% on LLaDA-1.5, versus about 50% for Llama-3-8B (the Llama figure comes from prior work).
     - "Context nesting" is a black-box attack that embeds the harmful request in structured benign contexts. JBB ASR-E is 86/87/33% on LLaDA, LLaDA-1.5, and Dream. HarmBench ASR-E is 78/79/14%.
     - The paper calls itself the first jailbreak of **Gemini Diffusion**, but that test was manual: 5 of 10 JBB prompts were unsafe, and the authors call it "preliminary evidence".
   - **Internal inconsistency:** The text claims context nesting beats ReNeLLM by more than 50% on HarmBench, but its table shows gaps of 34, 33, and 8 points.
   - **Limitations:** There is no limitations section. The ethics statement flags LLM-judge bias.
   - **Sources:** [arXiv](https://arxiv.org/abs/2602.00388); [HTML v2](https://arxiv.org/html/2602.00388v2)
4. **TrajHijack: "Re-Mask and Redirect: Exploiting Denoising Irreversibility in Diffusion Language Models"**
   - **Authors:** Arth Singh (sole author).
   - **Venue:** arXiv-only (2604.08557).
   - **Threat model:** White-box. The attacker can observe and modify intermediate denoising states (e.g., self-hosted weights or exposed denoising hooks), but cannot change the prompt or the weights.
   - **Method:** Re-mask already-committed refusal tokens and inject an affirmative prefix.
   - **Results:**
     - 74–82% HarmBench ASR on three safety-tuned dLLMs, rising to 92–98% with an 8-token compliance prefix.
     - Ablations: re-masking alone gives 4.4% and the prefix alone 5.7%.
     - Gumbel-softmax gradient optimization *lowers* ASR (41.5% vs 76.1%).
     - **"Defense Inversion Effect"**: A2D-aligned models are *more* vulnerable (89.9%) than undefended ones (76.1%).
   - **Version discrepancy:** The abstract (latest version) says three models and includes the A2D result. The v1 HTML tests only LLaDA-8B and Dream-7B and doesn't use the term.
   - **Early-commitment evidence:** Refusal tokens are typically committed by step 8 of 64. ASR is 82–92% for interventions anywhere from step 4 to step 48.
   - **Proposed defenses:**
     - A runtime monotonicity check (mask count never increases), which catches 100% of these attacks by construction.
     - Safety-aware unmasking schedules.
     - Step-conditional prefix detection.
     - Post-commitment re-verification.
   - **Stated limitations:**
     - Only two dLLMs were tested.
     - The attack needs white-box access.
     - LLaDA ASR falls from 94% to 52% as generation length grows from 64 to 512.
     - Ablations use n=50 (±10pp CI).
     - Experiments use greedy decoding with a linear schedule only.
     - A single LLM judge (Claude Sonnet 4) was used.
   - **Sources:** [arXiv](https://arxiv.org/abs/2604.08557); [HTML v1](https://arxiv.org/html/2604.08557v1)
5. **MaskForge: "Structure-Aware Adaptive Attacks for Jailbreaking Diffusion Large Language Models"**
   - **Authors:** Yingzi Ma, Zhengyue Zhao, Xiaogeng Liu, Minhui Xue, Yue Zhao, Chaowei Xiao.
   - **Venue:** arXiv-only preprint (2606.04027).
   - **Method:** A fully black-box adaptive attack with a structure-aware mask-template pattern library. It argues that prior attacks ignore native infilling or use low-diversity templates.
   - **Results:** 79.3% average ASR across five dLLMs and three benchmarks, a 17.6% relative gain over the best dLLM baseline. It transfers to AdvBench at 88.2% (+67% relative). The evaluation adapts three families of AR-era defenses to the mask-fill setting.
   - **Gap:** The abstract doesn't say whether it tests against A2D or DiffuGuard.
   - **Sources:** [arXiv](https://arxiv.org/abs/2606.04027); [HTML](https://arxiv.org/html/2606.04027v1)
6. **"Diffusion LLMs as Targets and Adversaries: Mechanistic Safety Exploits"**
   - **Authors:** Elena Dumitrescu, Gert Lek, Lydia Y. Chen, Jérémie Decouchant.
   - **Venue:** arXiv-only (2608.07430).
   - **Finding:** dLLM safety is sparse and inherited from AR predecessors.
   - **Pruning results:**
     - Pruning safety neurons raises ASR 2.6→73.8% on LLaDA and 1.9→86.6% on Dream.
     - Transfer-pruning from Qwen2.5 raises ASR 1.9→73.2% on Dream and 7.0→86.3% on Fast-dLLM.
   - **SN-Guided Diffusion:** A black-box jailbreak that steers the denoising trajectory away from safety-neuron activations. It reaches AUROC=1.0 for benign/jailbreak discrimination, and transfer ASR up to 77.1% (Llama-3-8B), 86.9% (Qwen2.5-7B), and 74.3% (Gemini-2.5-Flash-Lite), with 20 episodes per prompt.
   - **Sources:** [arXiv](https://arxiv.org/abs/2608.07430)
7. **"Diffusion LLMs are Natural Adversaries for any LLM"**
   - **Authors:** David Lüdke, Tom Wollschläger, Paul Ungermann, Stephan Günnemann, Leo Schwinn.
   - **Venue:** arXiv-only (2511.00203).
   - **Method:** Uses dLLMs, which model the joint distribution of prompt and response, to amortize adversarial-prompt search by conditional sampling.
   - **Results:** Produces low-perplexity, diverse, transferable jailbreaks against black-box, robustly trained, and proprietary LLMs. The abstract gives no numbers.
   - **Source:** [arXiv](https://arxiv.org/abs/2511.00203)
8. **GPO-V: "Jailbreak Diffusion Vision Language Model by Global Probability Optimization"**
   - **Authors:** Yu Pan, Andi Zhang, Yi Wang, Sibei Yang, Wenjie Wang.
   - **Venue:** arXiv-only (2605.07399).
   - **Method:** Global Probability Optimization over the masked-diffusion denoising trajectory. The authors call it the first visual-modality jailbreak for dVLMs, and argue that dVLMs look robust to prefix attacks while their progressive refusal opens another attack surface.
   - **Results:** The abstract gives no numbers or model names.
   - **Source:** [arXiv](https://arxiv.org/abs/2605.07399)
9. **DIVA: "Exploiting Cross-Step Conditional Propagation for Visual Jailbreaks in Discrete Diffusion Vision-Language Models"**
   - **Authors:** Guorui Song, Runqing Tang, Jingye Zhang, … Haoqian Wang (14 authors).
   - **Venue:** **EMNLP 2026 Main [A*]**, per arXiv comment (2609.05525).
   - **Vulnerability:** The visual embedding conditions *every* denoising step, so adversarial visual content is re-propagated and amplified. The authors support this with stage-sensitivity analysis, switch rates, and denoising-bin disagreement metrics, checked with bootstrap.
   - **Method:** A white-box, multi-timestep adversarial optimization combined with intent obfuscation.
   - **Results:** HADES ASR of 58.8%, 67.7%, and 69.1% on three dVLMs.
   - **Sources:** [arXiv](https://arxiv.org/abs/2609.05525)

#### B. Attacks (training-time / backdoor)
10. **BadDLM: "Backdooring Diffusion Language Models with Diverse Targets"**
    - **Authors:** Shengfang Zhai, Xiaoyang Ji, Yuling Shi, Haoran Gao, Fanyu Meng, Yan Zeng, Yuejian Fang, Yinpeng Dong, Jiaheng Zhang.
    - **Venue:** arXiv preprint (2605.09397).
    - **Method:** A trigger-aware objective that the authors claim is equivalent to an induced forward-masking distribution. It targets four levels: concept injection, attribute steering, alignment bypass, and code payload injection.
    - **Results:** Reported effective against AR-backdoor defenses. The abstract gives no numbers.
    - **Source:** [arXiv](https://arxiv.org/abs/2605.09397)
11. **SHADOWMASK: "Backdooring Masked Diffusion Language Models"**
    - **Authors:** Daniel Yiming Cao, Chengzhong Wang, Sheng-Yen Chou, Chengyu Huang, Pin-Yu Chen, Shengwei An.
    - **Venue:** arXiv-only (2605.19262).
    - **Method:** Replaces the all-mask terminal distribution with a trigger-mask mixture.
    - **Results:** "Near-100%" ASR on a DiT model and LLaDA-8B-Instruct. The backdoor survives full and PEFT fine-tuning and resists representative defenses.
    - **Source:** [arXiv](https://arxiv.org/abs/2605.19262)

#### C. Defenses / alignment
12. **A2D: "Any-Order, Any-Step Safety Alignment for Diffusion Language Models"**
    - **Authors:** Wonje Jeung, Sangyeon Yoon, Yoonjun Cho, Dongjae Jeon, Sangwoo Shin, Hyesoo Hong, Albert No (Yonsei).
    - **Venue:** **ICLR 2026 [A*]**.
    - **Method:**
      - Token-level alignment: harmful responses are randomly masked, and the model is trained to emit [EOS] at any masked harmful position.
      - A retain set preserves helpfulness.
      - LoRA on LLaDA, LLaDA-1.5, and Dream.
      - The data size is internally inconsistent: the main text says 30k BeaverTails examples, while the appendix says 3,021.
    - **Results:**
      - DIJA ASR: 82.9→1.3 (LLaDA), 82.7→3.5 (LLaDA-1.5), 84.4→0.0 (Dream).
      - Average ASR over attacks: 60.2→6.8 (LLaDA). For PAIR, 77.5→12.3. For prefilling, 69.6→1.9.
      - XSTest: 0% false positives.
      - Utility is roughly preserved, with a slight drop in general and math scores; for LLaDA, 66.6/41.4/32.6 → 66.2/40.6/35.0.
      - Early rejection by thresholding the step-1 [EOS] probability gives a 6× speedup at τ=0.9 (1.6% over-refusal) and up to 19.3× at τ=0.8.
      - Training costs 9.7T FLOPs, versus 29.0T for VRPO.
    - **Adaptive tests:** "Long-Answer" and "Do-Not-EOS" prompts did not raise ASR (LLaDA: 17.5 → 13.1/12.5).
    - **Stated limitation:** Only a few representative open dLLMs were tested; extension to future architectures is future work.
    - **Broken by:** TrajHijack (white-box re-masking), with 89.9% ASR on A2D.
    - **Sources:** [arXiv](https://arxiv.org/abs/2509.23286); [HTML](https://arxiv.org/html/2509.23286); [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/956e5427549a82a7472e02adc88360e9-Abstract-Conference.html); [TrajHijack](https://arxiv.org/abs/2604.08557)
13. **DiffuGuard: "How Intrinsic Safety is Lost and Found in Diffusion Large Language Models"**
    - **Authors:** Zherui Li, Zheng Nie, Zhenhong Zhou, Yue Liu, Yitong Zhang, Yu Cheng, Qingsong Wen, Kun Wang, Yufei Guo, Jiaheng Zhang.
    - **Venue:** **ICLR 2026 [A*]**. The PDF header and the ICLR poster page both confirm this. A NeurIPS 2025 virtual page also exists, possibly a workshop; that is **uncertain**.
    - **Training-free, two parts:**
      - *Stochastic Annealing Remasking:* blends confidence with uniform noise, strongest at early steps.
      - *Block-level Audit & Repair:* computes a "Safety Divergence" (cosine distance between hidden states of the core request and the full jailbreak prompt). If it exceeds λ=0.1, the first block is remasked and regenerated with the original harmful tokens' logits set to −∞, using 8 extra steps.
    - **Results:**
      - Average ASR over six attacks (WildJailbreak, JBB, PAD, DIJA, AutoDAN, GCG) and four dLLMs (LLaDA, Dream, LLaDA-1.5, MMaDA) falls 47.9→14.7% *when combined with Self-reminder*. DiffuGuard alone reaches 27.8%.
      - On PAD and DIJA, ASR drops 96.8→27.9%.
      - GSM8K: LLaDA 74.7→71.7, Dream 77.5→76.4.
      - Latency is described as "negligible", without numbers in the text.
    - **Adaptive tests (App. D.3):** multi-sampling, GCG against Safety Divergence, and o3-mini threshold probing. ASR stays nonzero; Dream threshold-probing reaches 84.62%.
    - **Stated limitations:**
      - Mostly AR-validated attacks, since only DIJA and PAD were dLLM-specific at the time.
      - Inference-time threats only, with no backdoor or poisoning.
      - Training-based alignment is complementary.
      - Future work: combine with adversarial safety training.
    - **Sources:** [arXiv](https://arxiv.org/abs/2509.24296); [HTML](https://arxiv.org/html/2509.24296); [ICLR poster](https://iclr.cc/virtual/2026/poster/10006479); [NeurIPS'25 page](https://neurips.cc/virtual/2025/133165)
14. **MOSA: "Where to Start Alignment? Diffusion Large Language Model May Demand a Distinct Position"**
    - **Authors:** Zhixin Xie, Xurui Song, Jun Luo.
    - **Venue:** **AAAI 2026 (oral) [A*]**.
    - **Finding:** *Middle* response tokens matter more for dLLM safety than initial tokens. Because dLLMs tend to generate sequentially, attackers struggle to manipulate those middle tokens.
    - **Method:** MOSA (Middle-tOken Safety Alignment) uses RL to align middle-token generation to refusals.
    - **Results:** Evaluated against eight attacks on two benchmarks, with utility checked on coding, math, and reasoning. The abstract gives no numbers.
    - **Sources:** [arXiv](https://arxiv.org/abs/2508.12398); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/37106)
15. **RAEC: "Beyond Token Positions: Safety Alignment Across Denoising Steps in Diffusion Language Models"**
    - **Authors:** Guoli Wang, Haonan Shi, Tu Ouyang, An Wang.
    - **Venue:** **EMNLP 2026 [A*]**, per arXiv comment (2609.00495).
    - **Method:** Refusal-Aware Early Commitment, a training-free decoder that commits persistent refusal tokens early.
    - **Results:**
      - Dream DIJA-HarmBench: 26.5→12.75.
      - LLaDA DIJA-HarmBench: 40.5→34.0.
      - LLaDA PAIR: 48→36%.
      - Utility is roughly unchanged (LLaDA GSM8K 54.66→54.44).
      - It was not compared experimentally against A2D, DiffuGuard, or MOSA.
    - **Stated limitations:**
      - Vanilla architectures only.
      - May not generalize to newer architectures.
      - Doesn't cover safety dynamics during fine-tuning (future work).
    - **Sources:** [arXiv](https://arxiv.org/abs/2609.00495v1); [HTML](https://arxiv.org/html/2609.00495)
16. **"Adaptive Steering and Remasking for Safe Generation in Diffusion Language Models"**
    - **Authors:** Yejin Lee, Ungsik Kim, Yo-Sub Han.
    - **Venue:** arXiv-only (2605.13043).
    - **Method:**
      - A state-dependent gating direction combined with steering on masked positions.
      - A lightweight detector runs after the first block and, when triggered, remasks the response and part of the prompt, then regenerates under steering.
      - No parameter changes.
    - **Results:** LLaDA-8B average JBB ASR 41.4→22.7. This comes from a search snippet and should be treated as unverified detail. The authors note that Self-reminder is strong on Dream but doesn't transfer to LLaDA.
    - **Criticism of prior defenses:** They rely on "fixed interventions or aggressive remasking" that degrade quality.
    - **Sources:** [arXiv](https://arxiv.org/abs/2605.13043); [code](https://github.com/leeyejin1231/DLM_Steering_Remasking)
17. **SRI Guard: "Step-Wise Refusal Dynamics in Autoregressive and Diffusion Language Models"**
    - **Authors:** Eliron Rahimi, Elad Hirshel, Rom Himelstein, Amit LeVi, Avi Mendelson, Chaim Baskin.
    - **Venue:** **EMNLP 2026 Main [A*]**.
    - **Finding:** Remasking lets dLLMs recover from harmful intermediate states. Switching LLaDA from AR to diffusion sampling *with the same weights* lowers ASR, a robustness improvement (ΔASR: Wild +18, Flip +36, PAIR +5).
    - **Cross-model ASR:** dLLMs 9–21% vs AR models 48–62%.
    - **Detector:** Uses SRI signals and is trained only on benign data. Last-layer AUROC is 0.920, with about 0.01% overhead.
    - **Stated limitations:**
      - Not a complete safety solution.
      - Misses jailbreaks with no internal refusal hesitation.
      - Combination with other defenses is untested.
    - **Sources:** [arXiv](https://arxiv.org/abs/2602.02600); [HTML](https://arxiv.org/html/2602.02600)
18. **"Why Jailbreaks Succeed in Diffusion Language Models: An Energy Landscape Analysis"**
    - **Authors:** Thong Bach, Dung Nguyen, Thao Minh Le, Truyen Tran.
    - **Venue:** Journal-ref **NeurIPS 2026 [A*]** (from arXiv metadata, not independently verified).
    - **Framework:** Safety is an energy barrier, and every attack either hides intent at initialization or forces a barrier crossing mid-trajectory.
    - **Detection signals:** Three training-free signals: a step-0 logit ratio and two trajectory-velocity ("kinetic energy") signals.
    - **Models:** LLaDA-8B, LLaDA-1.5, Dream-7B, and LLaDA-MoE-7B.
    - **Stress test:** Every attack configuration that evaded detection also failed to produce harmful content.
    - **Sources:** [arXiv](https://arxiv.org/abs/2609.30841v1)

## What attack surfaces are unique to dLLMs vs autoregressive LLMs?

### Takeaway
Five dLLM-specific surfaces are documented:
1. Mask infilling of attacker-structured templates, which exploits bidirectional context (DIJA, MaskForge, context nesting).
2. Parallel, multi-point decoding (PAD).
3. Early-step commitment combined with irreversible unmasking (TrajHijack, anchoring).
4. Trajectory-level conditioning, where every step re-reads the condition (DIVA, GPO-V).
5. The forward-masking process as a backdoor channel (BadDLM, SHADOWMASK).

dLLMs are also *more* robust than AR models to AR-designed attacks such as GCG.

### Cited Findings
- Bidirectional modeling plus parallel decoding prevents per-token dynamic refusal, so interleaved mask-text prompts force harmful infills — [DIJA](https://arxiv.org/abs/2507.11097)
- A Multi-Point Attention Attack on parallel decoding reaches 97% ASR, and AR jailbreaks transfer poorly — [PAD](https://arxiv.org/abs/2507.19227)
- Structure-aware adaptive mask templates reach 79.3% average ASR; prior work underused native infilling — [MaskForge](https://arxiv.org/abs/2606.04027)
- AR attacks are weak on dLLMs: GCG gets 16% on LLaDA-1.5 vs about 50% on Llama-3-8B. Embedding the request in structured benign context breaks this robustness, and Gemini Diffusion produced unsafe output on 5/10 prompts — [Safer by Diffusion](https://arxiv.org/html/2602.00388v2)
- Refusal is committed early (by about step 8/64). Re-masking committed refusal tokens plus an affirmative prefix gives 74–98% ASR — [TrajHijack](https://arxiv.org/abs/2604.08557)
- Prefilling/template attacks bypass response-level refusal because dLLMs can generate in any order — [A2D](https://arxiv.org/abs/2509.23286)
- In dVLMs the visual condition is applied at every denoising step ("cross-step conditional propagation") — [DIVA](https://arxiv.org/abs/2609.05525); see also [GPO-V](https://arxiv.org/abs/2605.07399)
- Backdoors can exploit the forward masking distribution or terminal mask distribution — [BadDLM](https://arxiv.org/abs/2605.09397); [SHADOWMASK](https://arxiv.org/abs/2605.19262)
- dLLM safety is sparse and inherited from AR initializations, so pruning a few neurons collapses it (2.6→73.8% ASR on LLaDA) — [Mechanistic Exploits](https://arxiv.org/abs/2608.07430)
- dLLMs can also be *used* as amortized jailbreak generators against any LLM — [Natural Adversaries](https://arxiv.org/abs/2511.00203)

### Inferences
- The common thread is that dLLM safety appears to depend on a few early, high-confidence commitments plus training data that assumed left-to-right responses. Attacks either pre-fill the response canvas (templates, prefill), overwrite early commitments (re-mask), or remove the sparse safety circuitry.
- dLLMs look "safer" only against AR-shaped threat models. The robustness and vulnerability claims (Safer-by-Diffusion, SRI) are consistent once the threat model is specified.

### Gaps
- I found no safety evaluation specific to SDAR, block-diffusion (BD3-LM), or Fast-dLLM v2 in particular. Fast-dLLM appears only as a pruning target in the Mechanistic Exploits paper.
- I found no paper evaluating Mercury (Inception). Gemini Diffusion was tested only in a 10-prompt manual case study.
- I found no confirmation that the "anchoring attack" named in an energy-landscape search snippet is a standalone paper.

## What defenses exist, what do they assume, and where do they fail?

### Takeaway
There are two families of defense:
- **Training-based:** A2D, MOSA, and DIJA's Refusal-Aware Denoising Alignment.
- **Training-free decoding/inference:** DiffuGuard, RAEC, Adaptive Steering+Remasking, SRI Guard, and energy-landscape detectors.

All of them assume the attacker controls only the prompt, a black-box threat model. The first explicit break of a published defense is TrajHijack's white-box re-masking of A2D. Adaptive-attack evaluations in the defense papers themselves are thin.

### Cited Findings
- **A2D:**
  - Assumes black-box prompt attacks and an open dLLM that can be fine-tuned.
  - Near-zero DIJA ASR and 0% XSTest false positives.
  - Small math/general utility drop, no inference overhead, and early termination up to 19.3× faster.
  - Self-tested adaptive prompts (Long-Answer, Do-Not-EOS) fail to break it.
  - — [A2D](https://arxiv.org/html/2509.23286)
- **TrajHijack vs A2D:**
  - A2D is *more* vulnerable than the base model to white-box re-masking (89.9% vs 76.1%).
  - The suggested countermeasure is a runtime check that the mask count never increases.
  - — [TrajHijack](https://arxiv.org/abs/2604.08557)
- **DiffuGuard:**
  - Assumes access to hidden states and a known "core harmful prompt" signal.
  - Reaches 27.8% ASR alone and 14.7% with Self-reminder.
  - Adaptive threshold-probing still gets 84.62% on Dream.
  - GSM8K drops by up to 3 points on LLaDA.
  - Limitations: no backdoor threat model, and mostly AR-designed attacks.
  - — [DiffuGuard](https://arxiv.org/html/2509.24296)
- **DIJA's own alignment defense:** Leaves 19–34% ASR and costs about 5 points on GPQA — [DIJA](https://arxiv.org/html/2507.11097v2)
- **RAEC:** Gives modest reductions on LLaDA (DIJA-HBB 40.5→34.0) and larger ones on Dream (26.5→12.75), at negligible utility cost — [RAEC](https://arxiv.org/html/2609.00495)
- **SRI Guard:**
  - Detection-only, at 0.01% overhead.
  - Misses attacks that produce no internal refusal hesitation.
  - — [SRI](https://arxiv.org/html/2602.02600)
- **Adaptive Steering+Remasking:** Argues that prior defenses' "fixed interventions or aggressive remasking" degrade generation quality — [Lee et al.](https://arxiv.org/abs/2605.13043)
- **Energy-landscape detectors:** Detection and barrier-crossing thresholds are hard to separate: every attack that evaded detection also failed to produce harm — [Energy Landscape](https://arxiv.org/abs/2609.30841v1)
- **Backdoors:** Reported robust to AR-designed backdoor defenses and to fine-tuning — [BadDLM](https://arxiv.org/abs/2605.09397); [SHADOWMASK](https://arxiv.org/abs/2605.19262)

### Inferences
- The defenses rely on early-step control (A2D's step-1 [EOS], RAEC's early commitment, DiffuGuard's first-block audit). That makes them exactly the target of state-manipulation attacks such as TrajHijack. A defense that locks in refusal early becomes a single point of failure once the attacker can re-mask.
- None of the defenses I found reports a systematic adaptive evaluation against MaskForge-style adaptive template search or white-box trajectory attacks.

### Gaps
- MaskForge's results against A2D or DiffuGuard are not stated in its abstract.
- I found no exact DiffuGuard latency numbers (they appear only in charts).
- MOSA's quantitative results were not retrievable from the abstract.
- I found no head-to-head benchmark across all defenses.

## Which denoising timesteps / positions / states are safety-critical, and how was it established?

### Takeaway
The evidence converges on early denoising steps and leading response positions. MOSA dissents on positions, arguing that middle tokens matter most. Some of the evidence is causal (token-forcing and prefill interventions in DiffuGuard, RAEC, and TrajHijack, plus the fixed-weight sampler swap in SRI). Much of it is still correlational (Safer-by-Diffusion's stepwise-reduction hypothesis, SRI signal–robustness associations).

### Cited Findings
- **DiffuGuard** ("Denoising-path Dependence"), *interventional*:
  - Forcing the first token to "Sure" raises ASR by 76.9%; forcing "Sorry" lowers it by 24.3%.
  - Inserting "Sorry" earlier works better than inserting it later.
  - Greedy low-confidence remasking prunes safe tokens in favor of confident harmful ones; random remasking raises safety but also perplexity.
  - Analysis was done mainly on LLaDA-8B.
  - — [DiffuGuard](https://arxiv.org/html/2509.24296)
- **RAEC:**
  - *Observational:* Refusal tokens ("sorry", "can't") peak in the first ~8% of steps and the first 2–8% of positions.
  - *Interventional:* Refusal/compliance prefixes at positions from 1% to 75%, and single refusal-token forcing at steps 1–64.
  - — [Beyond Token Positions](https://arxiv.org/html/2609.00495)
- **TrajHijack:**
  - Refusal is committed by about step 8–16 of 64.
  - Re-masking interventions anywhere in steps 4–48 yield 82–92% ASR. Early commitment is necessary but not irreversible.
  - — [TrajHijack](https://arxiv.org/html/2604.08557v1)
- **A2D:** The step-1 [EOS] probability at the leftmost masked position is predictive enough for early rejection (6× speedup at 1.6% over-refusal) — [A2D](https://arxiv.org/html/2509.23286)
- **MOSA:** Middle response tokens are more critical than initial ones, because sequential-ish generation makes them hard for attackers to control — [MOSA](https://arxiv.org/abs/2508.12398)
- **SRI:**
  - *Causal manipulation:* Swapping the sampler with fixed weights shows sampling matters.
  - *Correlational:* The recovery mechanism (remasking repairs harmful intermediates) is inferred from step-wise statistics. Middle and last layers carry the signal (AUROC 0.864 and 0.920).
  - — [SRI](https://arxiv.org/html/2602.02600)
- **Energy Landscape:** The step-0 logit distribution already encodes safety disposition, and trajectory "velocity" reveals mid-trajectory barrier crossings — [Energy](https://arxiv.org/abs/2609.30841v1)
- **Safer-by-Diffusion:** Shrinking "safety distance" across steps is an *assumption* (Assumption 3.2), supported by a keyword-based Monte-Carlo trend — [Safer by Diffusion](https://arxiv.org/html/2602.00388v2)
- **DIVA:** Stage sensitivity and denoising-bin disagreement analyses (with bootstrap) show that adversarial visual signals act across steps — [DIVA](https://arxiv.org/abs/2609.05525)
- **Mechanistic:** Safety-critical *neurons* are sparse, and pruning them is a causal ablation — [Mechanistic Exploits](https://arxiv.org/abs/2608.07430)

### Inferences
- Early-step and leading-position dominance is supported by multiple independent interventions, which is reasonably strong evidence. But TrajHijack shows that the dominance comes from the irreversible unmasking schedule, not from the model weights. With remasking allowed (SRI) or an adversarial re-mask (TrajHijack), later steps can override early commitments in either direction.
- The MOSA "middle-token" claim conflicts in emphasis with RAEC and DiffuGuard. The difference may come from different notions of position: where refusal is committed versus where harmful content concentrates. This has not been reconciled.
- None of these studies does causal tracing or activation patching *across* denoising steps (e.g., patching step-k hidden states). That is an open methodological gap.

### Gaps
- I found no study that patches intermediate denoising states between safe and unsafe runs to localize causality per (step, layer, position).
- Results come almost entirely from LLaDA and Dream with fixed or linear schedules and greedy decoding.

## What open problems do these papers explicitly list?

### Takeaway
The stated open problems are:
- Moving from AR-borrowed alignment to dLLM-native alignment.
- Extending results to new architectures (block diffusion, MoE, multimodal).
- Covering threat models beyond prompt-only attacks (backdoors, fine-tuning, white-box state access).
- Combining training-free and training-based defenses.

### Cited Findings
- DIJA calls for "rethinking safety alignment" and for "novel alignment strategies" for bidirectional/parallel decoding, and calls its own alignment defense preliminary — [DIJA](https://arxiv.org/html/2507.11097v2)
- A2D: future work is extending to new dLLM architectures given rapid evolution — [A2D](https://arxiv.org/html/2509.23286)
- DiffuGuard:
  - Needs more dLLM-specific attacks and defenses.
  - Should be combined with alignment training, such as adversarial training on samples it detects.
  - Backdoor and poisoning threats are not covered.
  - — [DiffuGuard](https://arxiv.org/html/2509.24296)
- RAEC: safety dynamics during fine-tuning (fine-tuning attacks) and newer architectures are future work — [RAEC](https://arxiv.org/html/2609.00495)
- SRI: integrating the detector with other defenses, and handling jailbreaks with no internal refusal hesitation — [SRI](https://arxiv.org/html/2602.02600)
- TrajHijack:
  - Stochastic decoding and alternative schedules are unexplored.
  - Defenses should enforce mask monotonicity and re-verify committed tokens.
  - — [TrajHijack](https://arxiv.org/html/2604.08557v1)
- Safer-by-Diffusion: alignment must account for structured, higher-level context attacks, and the evidence on proprietary models (Gemini Diffusion) is preliminary — [Safer by Diffusion](https://arxiv.org/html/2602.00388v2)
- Energy Landscape: detection and harm thresholds are hard to separate, which suggests a limit on detection-based evaluation — [Energy](https://arxiv.org/abs/2609.30841v1)

### Inferences
- The most under-served directions are:
  - Certified or monotonic decoding guarantees against state manipulation.
  - Causal localization of safety across denoising steps.
  - Safety for block/semi-AR dLLMs (SDAR, Fast-dLLM v2) and commercial dLLMs (Mercury, Gemini Diffusion).
  - Joint evaluation of defenses against adaptive dLLM-native attacks (MaskForge, TrajHijack).

### Gaps
- I couldn't read full limitations sections for MOSA, PAD, MaskForge, BadDLM, SHADOWMASK, GPO-V, Energy Landscape, or Mechanistic Exploits; only their abstracts were fetched.
- MOSA's numeric results are also unretrieved.
- There may be additional 2026 papers this search missed, particularly workshop papers and OpenReview-only submissions.
