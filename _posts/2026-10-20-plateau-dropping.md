---
layout: post
title: "Plateau Dropping: a simple SFT rule for better agentic post-training"
description: "Dropping each token from the SFT loss once its loss stops changing; after 10 epochs on SWE-agent data, 48.8% vs 42.2% pass@1 on SWE-bench Verified."
authors: Harshit Varma
list_description: false
math: true
---

<div class="post-intro" markdown="1">
<div class="post-summary" markdown="1">

**Summary.** 
Supervised fine-tuning (SFT) has become a key component of post-training recipes for almost every major LLM.
However, this is usually not the final step in standard post-training pipelines.
Several existing works show that although prolonged SFT improves pass@1, it usually ends up hurting diversity / coverage (as measured by pass@k, for some "large" k). This is called the "sharpening effect", wherein SFT "sharpens" the model to be even more confident at things it is already very confident at.
This behavior is not conducive to further post-training of the model as it inhibits exploration. For example, GRPO-based RL pipelines, which usually follow SFT in a lot of post-training pipelines, do not work well with a model that is not diverse enough to start with.
TailSFT is a recent method proposed to tackle this problem. However, we find that TailSFT (and some variants/extensions that we tried) does not help in our long-horizon agentic setup.
We instead propose a simple approach which we call *Plateau Dropping*. It is very loosely inspired by GRPO-based methods not learning from saturated (all pass) / impossible (none pass) tasks.
During SFT, we track the loss for every token, and drop it from the loss if it has not changed enough in the last few times it was seen. This effectively allows the model to focus on the "learnable band" of tokens at every step.
If a dropped token's loss moves again in the future, it gets re-added to the loss.
We find that this simple approach works well for long-horizon agentic tasks, as measured by pass@1 and pass@8 on SWE-bench Verified, Terminal-Bench 2.0, and OpenThoughts-TBLite for an open-source 67B-A2B model.
We in fact observe consistent gains for both pass@1 and pass@8 in some settings, not only leading to better starting points for further post-training, but also better checkpoints in general.

</div>
<div class="post-toc" markdown="1">

<p class="post-toc-title">Contents</p>

- [1. Introduction and motivation](#1-introduction-and-motivation)
- [2. Method](#2-method)
  - [2.1 The plateau test](#21-the-plateau-test)
  - [2.2 What gets dropped](#22-what-gets-dropped)
  - [2.3 Setup](#23-setup)
  - [2.4 Evaluation](#24-evaluation)
- [3. Results](#3-results)
  - [3.1 The claim: a better checkpoint from the same data](#31-the-claim-a-better-checkpoint-from-the-same-data)
  - [3.2 Check 1: training budget](#32-check-1-training-budget)
  - [3.3 Check 2: a tuned baseline](#33-check-2-a-tuned-baseline)
  - [3.4 A second dataset: terminal benchmarks](#34-a-second-dataset-terminal-benchmarks)
  - [3.5 Baselines: TailSFT](#35-baselines-tailsft)
  - [3.6 A second starting checkpoint](#36-a-second-starting-checkpoint)
- [4. Related work](#4-related-work)
- [5. Conclusion](#5-conclusion)
- [Appendix A: training bugs we found along the way](#appendix-a-training-bugs-we-found-along-the-way)
- [Appendix B: our data in context](#appendix-b-our-data-in-context)
- [References](#references)

</div>
</div>

---

## 1. Introduction and motivation

**The sharpening problem.** Post-training starts with SFT on demonstrations, and what RL can find afterwards depends on what the SFT model still assigns probability to. SFT's loss pushes every demonstrated token toward probability 1, whether or not the model predicts it already. For a token the model already gets right, the gradient is small, but it is never zero, and on every visit it pushes the same way: up for the demonstrated token, down for every alternative. Several recent papers measure what this costs. SFT's gains at pass@1 come with losses at pass@k and in the diversity RL explores ([Dang et al., 2025](https://arxiv.org/abs/2504.10478)); limiting the model's confidence during fine-tuning improves reasoning under test-time compute ([Chen et al., 2025](https://arxiv.org/abs/2502.07154)); clipping the gradient of tokens the model already predicts gives a better initialisation for RL ([ICML 2026](https://icml.cc/virtual/2026/poster/61171)); and TailSFT ([Malladi et al., 2026](https://arxiv.org/abs/2608.25756)) drops the training sequences the model already fits for the same reason. Their shared reading is that SFT keeps spending gradient on what it has already taught, and that this makes the model sharper without making it better. We do not test that mechanism directly here. We measure how many training tokens are in the already-learned state on agent data, and ask whether dropping them from the loss, token by token, gives a better model.

**Setting.** We fine-tune a 67B mixture-of-experts model (about 2B active parameters) on successful agent trajectories: 6.3k trajectories covering 1.6k command-line tasks, and 9.2k trajectories covering 7.0k software-engineering tasks. Plain SFT keeps learning for as long as we train it: on the SWE data the mean training loss falls from 0.55 on the first pass to 0.16 on the fourteenth (Figure 1a), and the evaluation scores rise with it (Section 3).

![Figure 1](/assets/img/blog/plateau-dropping/fig1_losses_corrected.png)

Figure 1. Training loss on the SWE data. (a) Mean training loss per pass over the data (103 steps each) for plain SFT (red) and Plateau Dropping over all tokens (green). (b) The 20-epoch runs step by step (15-step moving average): plain SFT over all tokens; Plateau Dropping over all tokens, over the tokens it still trains and over the tokens it has dropped; grey, right axis: the fraction of tokens Plateau Dropping keeps, 100% until step 192 and 32% at the end; the dotted line marks step 192, where dropping starts, before which the dropped-token curve is undefined.
{: .caption}

**Most training tokens are already learned.** Plateau Dropping's own bookkeeping (Section 2) measures this directly: the moment it starts judging tokens, at step 192 of a run on the SWE data, half of the training tokens already have a loss that has stopped moving, at an average loss of 0.21; by the end of a 20-epoch run that is two thirds of the tokens, at an average loss of 0.05, while the tokens still being learned sit at 0.43 (Figure 1b). Plain SFT averages its loss over all of them, so every update is shared with tokens that are already right.

**Sequence-level filtering: TailSFT.** TailSFT ([Malladi et al., 2026](https://arxiv.org/abs/2608.25756)) drops training sequences the model already fits, judged by how far each sequence's loss has fallen below the base model's, so that learning concentrates on the under-modelled tail. On OLMo-3 7B it reports gains of up to 17 points at pass@16 on math and coding, and up to 4 points of pass@1 after RL; in its own tables the method is a pass@k method, with pass@16 up in 15 of 18 settings and pass@1 up in 9 and down in 9. On our agent data it does not help, in any of four variants we tried (Section 3.5).

Our reading of why is that two things about sequence-level filtering do not fit agent data. A trajectory mixes tokens the model has long learned (tool-call syntax, repeated commands) with tokens it has not, so a sequence-level score blurs them; and a snapshot of confidence is a poor guide to what is still being learned, since many easy tokens are still moving while some hard tokens are stuck. This led us to select individual tokens rather than whole sequences, and to judge each token by its own loss history rather than by a snapshot of confidence. We propose *Plateau Dropping*, which does both: it removes a token from the training loss once that token's loss has plateaued. The token stays in its sequence, so its loss is still measured each time the sequence is revisited, and if that loss starts changing again, the token goes back into the training loss.

## 2. Method

### 2.1 The plateau test

For each token position, the trainer keeps the token's loss at its last two visits and a visit count. For our 6,301 terminal trajectories this bookkeeping takes about 2 GB of host (CPU) memory in each training process, about 8 GB on a four-GPU node. From the third visit on, if the loss changed by less than 10% between each of the last two visits (relative change, with a floor of 0.05), the token is dropped: its term is removed from the training loss, so it no longer contributes to the parameter update, but it still stays in the input as context for the tokens around it. If its loss moves again, it is trained again. Learned and stuck tokens are treated alike. Everything else is standard SFT, and the loss is averaged over the kept tokens.

Formally, for a training trajectory $$y = (y_1, \dots, y_T) \sim \mathcal{D}$$, the loss of token $$i$$ is its negative log-likelihood under the model,

$$
\ell_i(\theta) = -\log p_\theta\big(y_i \mid y_{<i}\big),
$$

where $$y_{<i}$$ is everything before it in the trajectory (the prompt, earlier turns and tool outputs). Let $$w_i \in \{0, 1\}$$ be its SFT weight (1 for the agent's own tokens, 0 for prompts and tool outputs). Plain SFT minimises the expected loss of a trained token,

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = \mathbb{E}_{(y,\, i) \sim \mathcal{D}_w}\big[\ell_i(\theta)\big],
$$

where $$\mathcal{D}_w$$ draws a token uniformly from all tokens with $$w_i = 1$$ in the data, together with its trajectory. Let $$\ell_i^{(v)}$$ be the value of $$\ell_i$$ at the token's $$v$$-th visit. With the relative change

$$
r(a, b) = \frac{|b - a|}{\max(a, \varepsilon)},
$$

token $$i$$ is plateaued at visit $$v$$ when

$$
P_i^{(v)} = \mathbb{1}\Big[\, v \ge 3 \;\wedge\; r\big(\ell_i^{(v-2)}, \ell_i^{(v-1)}\big) < \tau \;\wedge\; r\big(\ell_i^{(v-1)}, \ell_i^{(v)}\big) < \tau \,\Big],
$$

and the loss of a batch, with each token at its current visit, is

$$
\mathcal{L}_{\mathrm{PD}} = \frac{\sum_i w_i \, \big(1 - P_i\big) \, \ell_i}{\sum_i w_i \, \big(1 - P_i\big)},
$$

where $$P_i$$ is treated as a constant, so no gradient flows through the selection. We use $$\tau = 0.1$$ and $$\varepsilon = 0.05$$. Plain SFT is the special case $$P_i = 0$$ for every token. The history is updated at every visit, including for dropped tokens, so a dropped token whose loss moves again is trained again at its next visit.

### 2.2 What gets dropped

A token can only be dropped from its third visit, so nothing is dropped until every row has been seen twice: on the SWE data that is step 192 of a run, and until then a Plateau Dropping run is plain SFT (with the same data order, the two runs' losses agree step by step to within 0.001). Dropping then starts abruptly (Figure 1b): the kept fraction falls to 51% by step 227, 38% by step 759 and 32% by step 1,519. Of the dropped tokens, 93 to 97% are dropped as learned, with a loss that falls from 0.21 when dropping starts to 0.05 at the end of the 20-epoch run, and 2 to 5% as stuck at a higher loss. The tokens still trained have a loss of 0.74 at step 228, 0.62 at 760 and 0.43 at 1,520, while the loss over all tokens is 0.48, 0.35 and 0.18; because the training loss is now an average over the harder half of the tokens, it sits well above plain SFT's even though the loss over all tokens tracks plain SFT to within 0.02. On the terminal data, whose rows are shorter and revisited more often, the kept fraction falls further, to about 15% by step 440. The dropped set changes over time as tokens re-enter, so none of this shows that a fixed set of tokens stays learned.

Our packer fills fixed-length rows (64k tokens on the SWE data, 32k on the terminal data) with whole trajectories, which leaves about 26% of each SWE row and 12% of each terminal row as padding, and the data loader samples rows with replacement. So "epochs" throughout this post are nominal token counts: one real pass over the SWE data is 103 steps, so 228, 760 and 1,520 steps ("3, 10 and 20 epochs") are 2.2, 7.4 and 14.8 passes; one pass over the terminal data is about 29 steps, so 66, 220 and 440 steps are 2.3, 7.6 and 15.2 passes. The plateau test counts visits, not epochs.

### 2.3 Setup

- **Model.** A 67B mixture-of-experts model with about 2B parameters active per token: [open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21](https://huggingface.co/open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21).
- **Terminal data.** 6,301 successful trajectories (45M tokens, 1,610 tasks): 5,324 from the public [allenai/tmax-sft](https://huggingface.co/datasets/allenai/tmax-sft) dataset, written by a 27B teacher, and 977 of the base model's own successes on synthetic terminal tasks. Context 32k.
- **SWE data.** 9,229 resolved trajectories (7,005 tasks) from the public [nvidia/Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) dataset, restricted to one shell command per step to match our agent (319M tokens). Context 64k.
- **Training.** Learning rate 1e-4, cosine schedule over the run with 3% warmup, AdamH, no weight decay, global gradient-norm clipping at 4.0 for Plateau Dropping and TailSFT (Appendix A, item 5, explains why the usual 1.0 is wrong for them) and 1.0 for plain SFT, whose gradient norm rarely reaches 1.0, bf16 compute with fp32 master weights and optimizer state, expert capacity factor 2.0 (no dropped token-to-expert assignments), router biases frozen, 16 nodes of 4 GPUs each. Plain SFT and Plateau Dropping runs of the same length share the data order.
- **Compared models.** On the terminal data: SFT and Plateau Dropping for 3, 10 and 20 nominal epochs (66, 220 and 440 steps), and TailSFT for 3. On the SWE data: SFT and Plateau Dropping for 3, 10 and 20 nominal epochs (228, 760 and 1,520 steps), TailSFT for 3, and the 3-epoch pair again at a learning rate of 3e-4. Every Plateau Dropping and TailSFT model in the main tables is a run at gradient clip 4.0; the cells marked as in progress belong to runs being redone at that clip.
- **Harness.** We use the [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) harness everywhere: all training trajectories are in its format, and all evaluations run in it. At each step the model writes its reasoning and one shell command, then sees the command's output.

### 2.4 Evaluation

We use Terminal-Bench 2.0 (89 tasks), [OpenThoughts-TBLite](https://huggingface.co/datasets/open-thoughts/OpenThoughts-TBLite) (TB-Lite below; 100 command-line tasks), and a fixed random 100 tasks of SWE-bench Verified. Every model runs in mini-swe-agent at temperature 1, with three times the default time limit. Each seed is one independent attempt at every task, and every model has 8 seeds. A task is solved only with full credit. An attempt that hits the time limit is scored by the tests on whatever the agent left, so in rare cases it still counts as solved; an attempt that fails for infrastructure reasons (a sandbox or API error; between 0 and 15 of a model's 800 attempts, usually 1 to 7) counts as a failed attempt. So every model has exactly 8 attempts on every task, and every number in this post, in the tables and the figures alike, is computed on the same tasks: all 100 on SWE-bench Verified, 88 on Terminal-Bench 2.0 and 97 on TB-Lite (one and three tasks whose sandboxes never built for any model are excluded). pass@k is the unbiased estimate from the 8 attempts. Differences between two models are paired by task, and the standard errors quoted for them come from a bootstrap over tasks.

One change to the SWE-bench Verified grader: before running the tests, the standard Harbor task script deletes every file that git does not track, which also deletes new source files the agent created as part of its fix, so a correct fix that adds a file is graded as a failure. Our test script deletes only the untracked files that the benchmark's own test patch touches, and is otherwise unchanged. Every model in this post is graded the same way, so the comparisons between them are unaffected, but our absolute SWE-bench Verified numbers can be higher than those from the stock Harbor tasks.

## 3. Results

### 3.1 The claim: a better checkpoint from the same data

![Figure 2](/assets/img/blog/plateau-dropping/fig5_swebench_corrected.png)

Figure 2. SWE-bench Verified (random 100 tasks), pass@1 and pass@8 (%) against training length for plain SFT (red) and Plateau Dropping (green), 8 seeds, all 100 tasks (Section 2.4); dashed: the base model; hollow markers: the 3-epoch pair at a learning rate of 3e-4 (both arms at clip 4.0; the Plateau Dropping marker appears when its evaluation finishes). Plateau Dropping runs are at gradient clip 4.0; its 3- and 20-epoch points are being rerun at that clip and appear when their evaluations finish.
{: .caption}

**Table 1.** SWE-bench Verified, pass@1 and pass@8 in %, 8 seeds, all 100 tasks (Section 2.4). Rows marked LR 3e-4 were trained at three times the default learning rate; at that rate plain SFT's gradient norm exceeds 1.0 on 43% of its steps, so both arms of those rows use clip 4.0, whereas at the default rate plain SFT uses the standard clip of 1.0 (Section 2.3). Every Plateau Dropping row is a run at gradient clip 4.0; the rows marked in progress are being rerun at that clip. Bold marks the best value at each training length.

| training length | model | pass@1 | pass@8 |
|---|---|---|---|
| none | base model | 19.2 | 46.0 |
| 3 epochs (228 steps) | + SFT | 23.4 | 58.0 |
| 3 epochs (228 steps) | + Plateau Dropping | [clip-4.0 run in progress] | [clip-4.0 run in progress] |
| 3 epochs (228 steps), LR 3e-4 | + SFT | 34.4 | 68.0 |
| 3 epochs (228 steps), LR 3e-4 | + Plateau Dropping | [clip-4.0 run in progress] | [clip-4.0 run in progress] |
| 10 epochs (760 steps) | + SFT | 42.2 | 70.0 |
| 10 epochs (760 steps) | + Plateau Dropping | **48.8** | **78.0** |
| 20 epochs (1,520 steps) | + SFT | 44.9 | 75.0 |
| 20 epochs (1,520 steps) | + Plateau Dropping | [clip-4.0 run in progress] | [clip-4.0 run in progress] |

The result the rest of the post rests on is the 10-epoch pair. After 10 nominal epochs on the same SWE data, Plateau Dropping reaches 48.8% pass@1 on SWE-bench Verified against 42.2% for plain SFT, and 78.0% against 70.0% at pass@8: 6.5 points at pass@1 (three standard errors, paired over tasks) and 8.0 at pass@8 (2.7 standard errors). The two runs' training-loss curves over all tokens are nearly indistinguishable (Figure 1), so the difference is not visible in training; it shows up in what the models can do, and most at pass@8.

### 3.2 Check 1: training budget

Plain SFT on this data is strong and keeps improving: 20 nominal epochs take the base model from 19.2% to 44.9% pass@1 and from 46.0% to 75.0% pass@8, and the loss keeps falling throughout (Figure 1). The question is how Plateau Dropping's lead behaves along that curve. The 3- and 20-epoch Plateau Dropping runs are being completed at the method's clip setting (Section 2.3) and appear in Figure 2 and Table 1 when their evaluations finish; our first runs, before that setting was fixed, put the lead largest at 3 epochs and closing by 20 (Appendix A).

### 3.3 Check 2: a tuned baseline

The runs above use a learning rate of 1e-4, which turns out to be low for a 228-step run. Repeating the 3-epoch plain-SFT run at 3e-4 (Table 1, rows marked LR 3e-4) lifts it by 11.0 points at pass@1 and 10.0 at pass@8, to 34.4% and 68.0%; it still sits 7.9 points below plain SFT at the default rate for 10 epochs, so the extra passes do most of the work there. The Plateau Dropping run at the higher rate is being evaluated, and the same pair at 10 epochs is training; together they say whether the gain survives a baseline trained at three times the rate.

### 3.4 A second dataset: terminal benchmarks

![Figure 3](/assets/img/blog/plateau-dropping/fig3_tb2_corrected.png)

Figure 3. Terminal-Bench 2.0, pass@1 and pass@8 (%) against training length (66, 220 and 440 steps) for plain SFT (red) and Plateau Dropping (green), averaged over 8 seeds; dashed: the base model. All 88 tasks (Section 2.4). The Plateau Dropping runs are being redone at clip 4.0 and appear when their evaluations finish.
{: .caption}

![Figure 4](/assets/img/blog/plateau-dropping/fig4_tblite_corrected.png)

Figure 4. TB-Lite, pass@1 and pass@8 (%) against training length (66, 220 and 440 steps) for plain SFT (red) and Plateau Dropping (green), averaged over 8 seeds; dashed: the base model. All 97 tasks (Section 2.4). The Plateau Dropping runs are being redone at clip 4.0 and appear when their evaluations finish.
{: .caption}

**Table 2.** TB-Lite, pass@1 and pass@8 in %, 8 seeds, all 97 tasks (Section 2.4).

| training length | model | pass@1 | pass@8 |
|---|---|---|---|
| none | base model | 16.6 | 50.5 |
| 3 epochs (66 steps) | + SFT | 27.2 | 67.0 |
| 3 epochs (66 steps) | + Plateau Dropping | [clip-4.0 run in progress] | [clip-4.0 run in progress] |
| 10 epochs (220 steps) | + SFT | 33.9 | 64.9 |
| 10 epochs (220 steps) | + Plateau Dropping | [clip-4.0 run in progress] | [clip-4.0 run in progress] |
| 20 epochs (440 steps) | + SFT | 33.2 | 68.0 |
| 20 epochs (440 steps) | + Plateau Dropping | [clip-4.0 run in progress] | [clip-4.0 run in progress] |

**Table 3.** Terminal-Bench 2.0, pass@1 and pass@8 in %, 8 seeds, all 88 tasks (Section 2.4).

| training length | model | pass@1 | pass@8 |
|---|---|---|---|
| none | base model | 7.1 | 28.4 |
| 3 epochs (66 steps) | + SFT | 10.2 | 30.7 |
| 3 epochs (66 steps) | + Plateau Dropping | [clip-4.0 run in progress] | [clip-4.0 run in progress] |
| 10 epochs (220 steps) | + SFT | 16.8 | 39.8 |
| 10 epochs (220 steps) | + Plateau Dropping | [clip-4.0 run in progress] | [clip-4.0 run in progress] |
| 20 epochs (440 steps) | + SFT | 14.5 | 35.2 |
| 20 epochs (440 steps) | + Plateau Dropping | [clip-4.0 run in progress] | [clip-4.0 run in progress] |

On the terminal data plain SFT peaks at 220 steps on both benchmarks (TB-Lite 33.9% pass@1, Terminal-Bench 2.0 16.8%) and slips by 440 steps (33.2% and 14.5%), with pass@8 between 64.9 and 68.0% on TB-Lite and between 35.2 and 39.8% on Terminal-Bench 2.0. The Plateau Dropping runs at the method's clip setting are being evaluated and appear in the figures and tables when they finish.

### 3.5 Baselines: TailSFT

**Table 4.** TailSFT against plain SFT at 3 nominal epochs (66 steps on the terminal data, 228 on the SWE data), pass@1 and pass@8 in %, averaged over 8 seeds. All rows of a benchmark use the same tasks (100, 88 and 97; Section 2.4). TailSFT drops half of each batch's trajectories, with the drop share ramped up linearly over the run, the setting the TailSFT paper selected for its math runs. All TailSFT rows are clip-1.0 runs that were clipped no more than plain SFT was (on 4% of logged steps on the SWE data; on the same three early steps on the terminal data), so they stand (Appendix A, item 5).

| benchmark | model | pass@1 | pass@8 |
|---|---|---|---|
| SWE-bench Verified | SFT | 23.4 | 58.0 |
| SWE-bench Verified | TailSFT, per trajectory | 23.4 | 51.0 |
| Terminal-Bench 2.0 | SFT | 10.2 | 30.7 |
| Terminal-Bench 2.0 | TailSFT, per trajectory | 9.9 | 27.3 |
| TB-Lite | SFT | 27.2 | 67.0 |
| TB-Lite | TailSFT, per trajectory | 22.0 | 59.8 |

TailSFT per trajectory is level with plain SFT at pass@1 on SWE-bench Verified (23.4% against 23.4%) and 7 points below it at pass@8 (51.0% against 58.0%, −7.0 ± 4.3 paired over tasks); on the terminal data it is 5 points below plain SFT at pass@1 on TB-Lite (22.0% against 27.2%) and level on Terminal-Bench 2.0 (9.9% against 10.2%), and 3 to 7 points below at pass@8. Three agent-aware variants from our first runs (the full drop share from the start, per agent step, per span) did no better. Two further baselines are being evaluated and will be added here: TailSFT with a smaller drop share (0.25), the setting the TailSFT paper found best on code, and a token-level TailSFT that applies the same reference-based rule to single tokens, the reference-snapshot counterpart of Plateau Dropping's history rule.

### 3.6 A second starting checkpoint

The same 10-epoch pair is being trained from a stronger starting checkpoint (the 02 October Antidoom RLVR checkpoint of the same model), with the same recipe, and will be reported here as one pair of rows.

## 4. Related work

**Token selection with a reference model.** Rho-1 ([Lin et al., 2024](https://arxiv.org/abs/2404.07965)) trains only on tokens whose loss is high relative to a reference model. Plateau Dropping needs no reference model, and its selection follows the model's own loss history.

**Example selection by loss.** TailSFT, Selective Backprop ([Jiang et al., 2019](https://arxiv.org/abs/1910.00762)) and loss truncation ([Kang and Hashimoto, 2020](https://arxiv.org/abs/2004.14589)) drop or prioritise whole examples. Plateau Dropping selects individual tokens instead, and drops stuck tokens as well as learned ones.

**Training dynamics.** Forgetting events ([Toneva et al., 2019](https://arxiv.org/abs/1812.05159)) and early-training scores ([Paul et al., 2021](https://arxiv.org/abs/2107.07075)) rank examples by how they behave during training. Plateau Dropping does this online, per token, over two passes.

**Down-weighting easy cases.** Focal loss ([Lin et al., 2017](https://arxiv.org/abs/1708.02002)) down-weights well-classified examples. Its hard-threshold, per-token version, dropping every token above a probability threshold, is the snapshot-of-confidence approach that Section 1 argues against.

**Minority tokens in RL.** Updating only the 20% highest-entropy tokens matches full updates in RL for reasoning ([Wang et al., 2025](https://arxiv.org/abs/2506.01939)), and DAPO ([Yu et al., 2025](https://arxiv.org/abs/2503.14476)) discards sample groups that carry no learning signal. Plateau Dropping is an SFT counterpart.

## 5. Conclusion

Plateau Dropping is a small change to standard SFT: the trainer remembers each token's loss from its last two visits, and once that loss stops changing, the token is removed from the training loss until it changes again. On long-horizon agentic data it gives a better checkpoint from the same data and compute. At 10 nominal epochs on the SWE data it is 6.5 points above plain SFT at pass@1 and 8.0 at pass@8 (48.8% against 42.2%, 78.0% against 70.0%), with a training loss that looks the same, and the gain is largest at pass@8. Two checks are in progress on that result: how the lead behaves across training budgets (Section 3.2) and whether it survives a baseline trained at three times the learning rate (Section 3.3). On the terminal benchmarks the comparison is being completed (Section 3.4). Sequence-level filtering of the same data (TailSFT) does not help (Section 3.5). The cells marked in progress are runs being completed at the method's clip setting and will fill in over the coming days, together with the pair from a stronger starting checkpoint (Section 3.6).

## Appendix A: training bugs we found along the way

Our first runs were all on one cluster. When we ported the setup to a second cluster to reproduce them, the Plateau Dropping result came out 4 points lower, and the search for that difference found two bugs in the trainer that had affected every one of those runs, plus two smaller issues. All four are fixed; both clusters now run byte-identical trainer code, and every run in this post passed an audit (no restarts, no non-finite losses, plain and Plateau Dropping losses identical step by step until dropping starts, every exported checkpoint validated against the training checkpoint).

![Figure 5](/assets/img/blog/plateau-dropping/fig_bugs_loss_curves.png)

Figure 5. Left: training loss over all tokens on the SWE data for our first, buggy runs (dashed and dotted, both clusters, plain SFT and Plateau Dropping in two data orders) and the fixed runs (solid): the buggy runs stay at 0.54 to 0.55 for 1,520 steps; the fixed runs fall to 0.15 to 0.18. Right: the fraction of tokens Plateau Dropping keeps, and the loss of the dropped tokens, in the buggy and the fixed runs.
{: .caption}

**1. The master weights were bf16.** The script that imported the base checkpoint wrote the weights in bf16, and the trainer kept whatever precision it loaded, so the optimizer applied its fp32 updates to bf16 weights and rounded the result back to bf16 at every step. A bf16 number has an 8-bit mantissa, so any update smaller than about 0.4% of a weight's magnitude rounds to zero, which is most Adam updates at a learning rate of 1e-4. After 1,520 steps, 92 to 95% of the weights were bit-identical to the base model; the training loss stopped at 0.55 (Figure 5) and the models barely changed. The fixed trainer keeps fp32 master weights (the standard mixed-precision recipe: bf16 compute, fp32 weights and optimizer state); in the fixed runs 65% of the weights have changed after 228 steps, 83% after 760 and 89% after 1,520, measured on the bf16 export. Every one of our first runs was trained this way: plain SFT, Plateau Dropping, TailSFT and its variants, the ablations, the runs from a stronger starting checkpoint, and the SFT starting points of the RL runs. (The Qwen3-8B experiments we ran alongside used a different trainer and were not affected.)

**2. The mixture of experts dropped token-to-expert assignments.** The model was trained with an expert capacity factor of 1.0, so whenever a batch routed more tokens to an expert than its even share, the excess assignments were dropped. On this data 14 to 19% of assignments were dropped, and the fraction depended on the hardware's rounding (18.6% on the GH200 cluster, 15.5% on the GB200 cluster), which cost 0.067 to 0.082 nats of loss and is what made the two clusters disagree. With the capacity factor raised to 2.0 the overflow is 0.16% at the start of a run and 0 by the end, the two clusters agree to four decimals, and the trainer's loss matches an independent inference implementation to 0.0002.

**3. The TailSFT reference pass zeroed the router bias.** The forward-only scoring pass that produces TailSFT's per-sequence reference losses applied a zero router bias instead of the frozen one from the checkpoint. The TailSFT runs in Table 4 use references from the fixed pass. The effect of the bug on the references was small (per-document error with a standard deviation of 0.006 and a rank correlation of 0.998 with the correct references), so it did not change the TailSFT result.

**4. Epochs were overstated.** We first counted passes from token totals and ignored row padding, so what we called "10 epochs" on the terminal data were 7.6 real passes and "20 epochs" on the SWE data were 14.8. This post keeps the nominal names and gives the real pass counts in Section 2.2.

**5. Gradient clipping was silently penalising Plateau Dropping.** The trainer clips the global gradient norm at 1.0, the usual setting, and the loss Plateau Dropping optimises is a mean over the kept tokens only. Once it has dropped half the tokens its gradient norm is two to three times plain SFT's, so the clip, which touches 3 to 5% of plain SFT's steps, hit 50% of the steps of the 10-epoch Plateau Dropping run on the SWE data and 82 to 100% of the steps of the terminal runs (Appendix A, item 5). Nothing in the training logs flagged it: the loss curves of the clipped and unclipped runs are indistinguishable, and only the evaluations differ, by 2 to 6 points at 10 epochs. The interaction belongs to any method that averages its loss over a kept subset of the batch, not to Plateau Dropping in particular; the TailSFT paper's runs use a clip of 1.0 with the loss averaged over the kept examples and report no gradient norms, so its results are likely affected the same way. At 10 epochs on the SWE data the two settings compare as follows (8 seeds, all 100 tasks; the loss curves of the two Plateau Dropping runs are indistinguishable, 0.347 against 0.346 over all tokens at the end):

| 10 epochs (760 steps) | pass@1 | pass@8 |
|---|---|---|
| plain SFT | 42.2 | 70.0 |
| Plateau Dropping, clip 1.0 | 46.6 | 72.0 |
| Plateau Dropping, clip 4.0 | **48.8** | **78.0** |

Clip 4.0 beats clip 1.0 by 2.1 ± 1.7 points at pass@1 and 6.0 ± 2.3 at pass@8, paired over tasks. Every Plateau Dropping and TailSFT run of our first pass was trained at clip 1.0; at that setting Plateau Dropping was ahead of plain SFT by 8.5 points at pass@1 and 9.0 at pass@8 after 3 epochs on SWE-bench Verified, by 4.4 and 2.0 at 10 epochs, level at 20, ahead by 4.9 and 6.2 on TB-Lite at 220 steps, and below its own 220-step result at 440 steps, where the clip had engaged on 91% of its steps. The TailSFT runs were clipped no more than plain SFT was (on 4% of logged steps on the SWE data, on the same three early steps on the terminal data) and stand as they are. Every Plateau Dropping run is being redone at clip 4.0, including the two on which the clip barely engaged at 1.0 (the 3-epoch SWE run, never; the 66-step terminal run, on the same three early steps as plain SFT), and only clip-4.0 runs appear in the main text. Normalising the kept-token loss by the full token count instead would remove the interaction by construction, since dropping would then never rescale the gradient; under Adam it should behave like the raised clip, and it has not been run.

**What the bugs did to our conclusions.** In the buggy runs the training loss flattened after one pass, and plain SFT barely moved the base model (19.5% pass@1 on SWE-bench Verified after 20 nominal epochs, against 19.3% for the base) while Plateau Dropping reached 31.1%. Both observations were the first bug: with the fixed trainer the loss falls throughout and plain SFT reaches 44.9% pass@1 and 75.0% pass@8. We do not have a complete account of why Plateau Dropping still improved under the bug. Its gradients are larger, because its loss is averaged over fewer, harder tokens, which would let more of its updates survive the rounding; but the fraction of weights that moved at all was the same in both arms, so that comparison was between two models that had barely trained, and its 10-point gap at 20 epochs does not survive the fix: with both arms training properly, the two are level (Table 1). The terminal-benchmark results of the buggy runs came from the same regime. What survives, once the gradient clip is set correctly for the method (item 5), is a lead of 6.5 points at pass@1 and 8.0 at pass@8 after 10 epochs on SWE-bench Verified; the fixed runs at the other budgets are in progress. The ablations, the runs from a stronger starting checkpoint, and the comparison of RL started from plain SFT and from Plateau Dropping were all trained under the bugs, and we leave them out of this post until they are rerun (the RL runs themselves were unaffected; their SFT starting points were not). The fixes (fp32 master weights, the capacity-factor setting, the corrected scoring pass, and a gradient-clipping setting for the ablation of Appendix A, item 5) are in our fork of the trainer.

## Appendix B: our data in context

By the standards of public agentic SFT data, our datasets are small to mid-sized (Table 5). In trajectory count they are close to the sets several SWE agents were fine-tuned on (SWE-smith, R2E-Gym, SWE-Dev) and to the terminal set we draw from, and one to two orders of magnitude below the largest public releases (Open-SWE-Traces, Nemotron-Terminal-Corpus). None of the public releases reports a total token count.

**Table 5.** Size of public agentic SFT datasets, as stated in each paper or dataset card ("–": not reported).

| dataset | domain | tasks | trajectories | tokens |
|---|---|---|---|---|
| [SWE-Gym](https://arxiv.org/abs/2412.21139) | SWE | 2,438 | 491 used for SFT | – |
| [R2E-Gym](https://arxiv.org/abs/2504.07164) | SWE | 2,048 environments | 3,321 | – |
| [SWE-smith](https://arxiv.org/abs/2504.21798) | SWE | 50k instances (128 repos) | 5,016 used for SFT | – |
| [SWE-Dev](https://arxiv.org/abs/2506.07636) | SWE | 38k instances | 2.3k to 16.6k across runs | – |
| [SWE-rebench OpenHands trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) | SWE | 3,792 issues resolved | 67,074 (32,161 successful) | – |
| [nvidia/Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) v1.2 | SWE | – | 263,371 | – |
| [allenai/tmax-sft](https://huggingface.co/datasets/allenai/tmax-sft) | terminal | about 2.2k environments | 10,726 (5,795 successful) | – |
| [OpenThoughts-Agent-v1-SFT](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-SFT) | terminal | – | 15,209 | – |
| [Nemotron-Terminal-Corpus](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus) | terminal | – | 366,154 | – |
| **ours, terminal data** | terminal | 1,610 | 6,301 | 45M (28M trained) |
| **ours, SWE data** | SWE | 7,005 | 9,229 | 319M |

## References

1. Malladi, S. et al. (2026). TailSFT: Filtered Fine-Tuning Improves Post-Training Performance. [arXiv:2608.25756](https://arxiv.org/abs/2608.25756).
2. Dang, X., Baek, C., Wen, K., Kolter, Z. and Raghunathan, A. (2025). Weight Ensembling Improves Reasoning in Language Models. [arXiv:2504.10478](https://arxiv.org/abs/2504.10478).
3. Chen, F., Raventos, A., Cheng, N., Ganguli, S. and Druckmann, S. (2025). Rethinking Fine-Tuning when Scaling Test-Time Compute: Limiting Confidence Improves Mathematical Reasoning. [arXiv:2502.07154](https://arxiv.org/abs/2502.07154).
4. Clipping Low-Probability Tokens in SFT Yields a Generalizable Initialization for RL. ICML 2026. [Poster page](https://icml.cc/virtual/2026/poster/61171).
5. Lin, Z. et al. (2024). Rho-1: Not All Tokens Are What You Need. [arXiv:2404.07965](https://arxiv.org/abs/2404.07965).
6. Jiang, A. H. et al. (2019). Accelerating Deep Learning by Focusing on the Biggest Losers. [arXiv:1910.00762](https://arxiv.org/abs/1910.00762).
7. Kang, D. and Hashimoto, T. (2020). Improved Natural Language Generation via Loss Truncation. ACL 2020. [arXiv:2004.14589](https://arxiv.org/abs/2004.14589).
8. Toneva, M. et al. (2019). An Empirical Study of Example Forgetting during Deep Neural Network Learning. [arXiv:1812.05159](https://arxiv.org/abs/1812.05159).
9. Paul, M., Ganguli, S. and Dziugaite, G. K. (2021). Deep Learning on a Data Diet. [arXiv:2107.07075](https://arxiv.org/abs/2107.07075).
10. Lin, T.-Y. et al. (2017). Focal Loss for Dense Object Detection. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002).
11. Wang, S. et al. (2025). Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning. [arXiv:2506.01939](https://arxiv.org/abs/2506.01939).
12. Yu, Q. et al. (2025). DAPO: An Open-Source LLM Reinforcement Learning System at Scale. [arXiv:2503.14476](https://arxiv.org/abs/2503.14476).
13. Pan, J. et al. (2025). Training Software Engineering Agents and Verifiers with SWE-Gym. [arXiv:2412.21139](https://arxiv.org/abs/2412.21139).
14. Jain, N. et al. (2025). R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents. [arXiv:2504.07164](https://arxiv.org/abs/2504.07164).
15. Yang, J. et al. (2025). SWE-smith: Scaling Data for Software Engineering Agents. [arXiv:2504.21798](https://arxiv.org/abs/2504.21798).
16. Wang, H. et al. (2025). SWE-Dev: Building Software Engineering Agents with Training and Inference Scaling. [arXiv:2506.07636](https://arxiv.org/abs/2506.07636).
17. Nebius (2025). SWE-rebench OpenHands trajectories. [Hugging Face](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories).
18. NVIDIA (2026). Nemotron-Terminal-Corpus. [Hugging Face](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus); [arXiv:2602.21193](https://arxiv.org/abs/2602.21193).
19. OpenThoughts (2025). OpenThoughts-Agent-v1-SFT. [Hugging Face](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-SFT).
