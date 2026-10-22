---
layout: post
title: "Plateau Dropping: per-token loss-plateau gating for agentic SFT"
description: "Dropping tokens from the SFT loss once their loss plateaus teaches agents in far fewer passes; with longer training, plain SFT catches up."
authors: Harshit Varma
math: true
---


**Summary.** Supervised fine-tuning on agent trajectories spends most of its gradient on tokens the model has already learned. We propose *Plateau Dropping*, a per-token change to SFT: the trainer tracks each training token's loss across passes over the data and removes the token from the training loss once its loss has stopped changing, so each update is driven only by tokens the model is still learning. What this buys is speed: the model acquires the agent's behaviour in far fewer passes. On SWE-bench Verified (random 100 tasks), after 3 nominal epochs on the same data, Plateau Dropping reaches 31.9% pass@1 against 23.4% for plain SFT, and 67.0% against 58.3% at pass@8. The trajectories show why: at that budget the plain-SFT agent still writes 55% longer reasoning per step, breaks the tool-call format in 1 attempt in 9, tests half as often and submits a third less, while the Plateau Dropping agent already behaves like a fully trained one. On the terminal benchmarks the advantage appears at 220 steps (+4.9 points of pass@1 on TB-Lite, +1.8 on Terminal-Bench 2.0). With more passes the gap closes, and for Plateau Dropping the closing looks like sharpening: between 10 and 20 nominal epochs its pass@1 stays put or slips while its pass@8 falls on all three benchmarks (by 2.0, 6.6 and 2.4 points on SWE-bench Verified, TB-Lite and Terminal-Bench 2.0), so that after 20 epochs on the SWE data the two methods are level (44.9% against 45.8% pass@1, 74.7% against 72.7% pass@8) and on the terminal data the 440-step Plateau Dropping run sits below its own 220-step result while plain SFT holds. Filtering whole sequences the model already fits (TailSFT) does not help at any budget we tested. One reason the long Plateau Dropping runs may be held back is gradient clipping, which hits them on 86 to 100% of their steps and plain SFT on 3 to 5%; an ablation with the clip raised is being evaluated.

---

## 1. Introduction and motivation

We fine-tune a 67B mixture-of-experts model (about 2B active parameters) on successful agent trajectories: 6.3k trajectories covering 1.6k command-line tasks, and 9.2k trajectories covering 7.0k software-engineering tasks. Plain SFT keeps learning for as long as we train it: on the SWE data the mean training loss falls from 0.55 on the first pass to 0.16 on the fourteenth (Figure 1a), and the evaluation scores rise with it (Section 3). The question this post asks is where that gradient goes, and whether an SFT recipe that concentrates its updates on the tokens the model is still learning gets to the trained behaviour faster, or further.

![Figure 1](/assets/img/blog/plateau-dropping/fig1_losses_corrected.png)

Figure 1. Training loss on the SWE data. (a) Mean training loss per pass over the data (103 steps each) for plain SFT (red) and Plateau Dropping over all tokens (green). (b) The 20-epoch runs step by step (15-step moving average): plain SFT over all tokens; Plateau Dropping over all tokens, over the tokens it still trains and over the tokens it has dropped; grey, right axis: the fraction of tokens Plateau Dropping keeps, 100% until step 192 and 32% at the end.
{: .caption}

**Most of the gradient goes to tokens that are already learned.** Plateau Dropping's own bookkeeping (Section 2) measures this directly: the moment it starts judging tokens, at step 192 of a run on the SWE data, half of the training tokens already have a loss that has stopped moving, at an average loss of 0.21; by the end of a 20-epoch run that is two thirds of the tokens, at an average loss of 0.05, while the tokens still being learned sit at 0.43 (Figure 1b). Plain SFT averages its loss over all of them, so the tokens that are already right dilute every update.

**Sequence-level filtering: TailSFT.** TailSFT ([Malladi et al., 2026](https://arxiv.org/abs/2608.25756)) drops training sequences the model already fits, judged by how far each sequence's loss has fallen below the base model's, so that learning concentrates on the under-modelled tail. On OLMo-3 7B it reports up to 17 points of pass@16 on math and coding, and up to 4 points of pass@1 after RL. On our agent data it does not help (Table 1): per-trajectory TailSFT for 3 nominal epochs, dropping half of each batch's trajectories with the drop share ramped up over the run, is level with plain SFT at pass@1 on SWE-bench Verified and Terminal-Bench 2.0 and 5 points below it on TB-Lite, and no better than plain SFT at pass@8 on either terminal benchmark.

**Table 1.** TailSFT against plain SFT at 3 nominal epochs (66 steps on the terminal data, 228 on the SWE data), pass@1 and pass@8 in %, averaged over 8 seeds. The last column is the paired difference, TailSFT minus SFT, with its standard error over tasks. Task counts (pass@1 / pass@8): Terminal-Bench 2.0 88 / 65; TB-Lite 97 / 80; SWE-bench Verified 100 / 94. The two agent-specific variants (per agent step, per span) and the full-drop-share-from-step-5 variant are still being evaluated.

| benchmark | model | pass@1 | pass@8 | TailSFT − SFT at pass@1 / pass@8 |
|---|---|---|---|---|
| SWE-bench Verified | SFT | 23.4 | 58.3 | |
| SWE-bench Verified | TailSFT, per trajectory | 24.3 | [pending] | +0.9 (1.7) / – [provisional, 710 of 800 trials] |
| Terminal-Bench 2.0 | SFT | 10.2 | 29.2 | |
| Terminal-Bench 2.0 | TailSFT, per trajectory | 10.1 | 27.7 | −0.1 (1.4) / −1.5 (5.1) |
| TB-Lite | SFT | 27.3 | 68.8 | |
| TB-Lite | TailSFT, per trajectory | 22.2 | 62.5 | −5.2 (2.0) / −6.2 (3.7) |
| terminal benchmarks | TailSFT, per trajectory, full drop share from step 5 | [pending] | [pending] | |
| terminal benchmarks | TailSFT, per agent step | [pending] | [pending] | |
| terminal benchmarks | TailSFT, per span (reasoning and command separately) | [pending] | [pending] | |

Two things about sequence-level filtering do not fit agent data. A trajectory mixes tokens the model has long learned (tool-call syntax, repeated commands) with tokens it has not, so a sequence-level score blurs them; and a snapshot of confidence is a poor guide to what is still being learned, since many easy tokens are still moving while some hard tokens are stuck. This led us to select individual tokens rather than whole sequences, and to judge each token by its own loss history rather than by a snapshot of confidence. We propose *Plateau Dropping*, which does both: it removes a token from the training loss once that token's loss has plateaued. The token stays in its sequence, so its loss is still measured each time the sequence is revisited, and if that loss starts changing again, the token goes back into the training loss.

## 2. Method

### 2.1 The plateau test

For each token position, the trainer keeps the token's loss at its last two visits and a visit count. For our 6,301 terminal trajectories this bookkeeping takes about 2 GB of host (CPU) memory in each training process, about 8 GB on a four-GPU node. From the third visit on, if the loss changed by less than 10% between each of the last two visits (relative change, with a floor of 0.05), the token is dropped: its term is removed from the training loss, so it no longer contributes to the parameter update, but it still stays in the input as context for the tokens around it. If its loss moves again, it is trained again. Learned and stuck tokens are treated alike. Everything else is standard SFT, and the loss is averaged over the kept tokens.

Formally, let $$\ell_i^{(v)}$$ be the loss of token $$i$$ at its $$v$$-th visit, and $$w_i \in \{0, 1\}$$ its SFT weight (1 for the agent's own tokens, 0 for prompts and tool outputs). With the relative change

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
- **Training.** Learning rate 1e-4, cosine schedule over the run with 3% warmup, AdamH, no weight decay, global gradient-norm clipping at 1.0, bf16 compute with fp32 master weights and optimizer state, expert capacity factor 2.0 (no dropped token-to-expert assignments), router biases frozen, 16 nodes of 4 GPUs each. Plain SFT and Plateau Dropping runs of the same length share the data order.
- **Compared models.** On the terminal data: SFT and Plateau Dropping for 3, 10 and 20 nominal epochs (66, 220 and 440 steps), and TailSFT for 3. On the SWE data: SFT and Plateau Dropping for 3 and 20 nominal epochs (228 and 1,520 steps), evaluated at the end and, for the 20-epoch runs, at their mid-schedule checkpoint (step 760), TailSFT for 3, and dedicated 10-epoch runs (760 steps with their own schedule) that are being evaluated.
- **Harness.** We use the [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) harness everywhere: all training trajectories are in its format, and all evaluations run in it. At each step the model writes its reasoning and one shell command, then sees the command's output.

**Our data in context.** By the standards of public agentic SFT data, our datasets are small to mid-sized (Table 2). In trajectory count they are close to the sets several SWE agents were fine-tuned on (SWE-smith, R2E-Gym, SWE-Dev) and to the terminal set we draw from, and one to two orders of magnitude below the largest public releases (Open-SWE-Traces, Nemotron-Terminal-Corpus). None of the public releases reports a total token count.

**Table 2.** Size of public agentic SFT datasets, as stated in each paper or dataset card ("–": not reported).

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

### 2.4 Evaluation

We use Terminal-Bench 2.0 (89 tasks), TB-Lite (100 command-line tasks), and a fixed random 100 tasks of SWE-bench Verified. Every model runs in mini-swe-agent at temperature 1, with three times the default time limit. Each seed is one independent attempt at every task, and every model has 8 seeds. A task is solved only with full credit. An attempt that hits the time limit is scored by the tests on whatever the agent left, so in rare cases it still counts as solved; infrastructure crashes are discarded. All numbers are averaged over seeds: pass@k is the unbiased estimate from the seeds' attempts, and each column uses the tasks with at least k valid attempts for every model in its comparison (the task counts are given with each table; four terminal tasks whose sandboxes are no longer available are excluded everywhere). Differences between two models are paired by task, and the standard errors quoted for them come from a bootstrap over tasks.

One change to the SWE-bench Verified grader: before running the tests, the standard Harbor task script deletes every file that git does not track, which also deletes new source files the agent created as part of its fix, so a correct fix that adds a file is graded as a failure. Our test script deletes only the untracked files that the benchmark's own test patch touches, and is otherwise unchanged. Every model in this post is graded the same way, so the comparisons between them are unaffected, but our absolute SWE-bench Verified numbers can be higher than those from the stock Harbor tasks.

## 3. Results

### 3.1 SWE-bench Verified

![Figure 2](/assets/img/blog/plateau-dropping/fig5_swebench_corrected.png)

Figure 2. SWE-bench Verified (random 100 tasks), pass@1 and pass@8 (%) against training length for plain SFT (red) and Plateau Dropping (green), averaged over 8 seeds; dashed: the base model. Each panel uses the tasks with at least k valid attempts for every model shown (100 at pass@1, 88 at pass@8). The 10-epoch point is the mid-schedule checkpoint of the 20-epoch runs until the dedicated 10-epoch runs are evaluated.
{: .caption}

**Table 3.** SWE-bench Verified, pass@1 and pass@8 in %, averaged over 8 seeds, all 100 tasks at pass@1 (pass@8: the number of tasks with 8 valid attempts for both models of a pair, in brackets). The last column is the paired difference, Plateau Dropping minus SFT, with its standard error over tasks.

| training length | model | pass@1 | pass@8 | PD − SFT at pass@1 / pass@8 |
|---|---|---|---|---|
| none | base model | 19.3 | 47.7 | |
| 3 epochs (228 steps) | + SFT | 23.4 | 58.3 (94) | |
| 3 epochs (228 steps) | + Plateau Dropping | **31.9** | **67.0** (94) | +8.4 (2.1) / +9.6 (4.6) |
| step 760 of the 20-epoch runs | + SFT | 45.0 | 72.7 (99) | |
| step 760 of the 20-epoch runs | + Plateau Dropping | **47.1** | **74.0** (99) | +2.1 (1.7) / +2.0 (2.9) |
| 10 epochs (760 steps, own schedule) | + SFT | [pending] | [pending] | |
| 10 epochs (760 steps, own schedule) | + Plateau Dropping | [pending] | [pending] | [pending] |
| 20 epochs (1,520 steps) | + SFT | 44.9 | **74.7** (98) | |
| 20 epochs (1,520 steps) | + Plateau Dropping | **45.8** | 72.7 (98) | +0.8 (1.7) / −3.1 (3.0) |

Three things stand out. First, plain SFT on this data is strong: 20 nominal epochs take the base model from 19.3% to 44.9% pass@1 and from 47.7% to 74.7% pass@8, and the loss keeps falling throughout (Figure 1). Second, at a small training budget Plateau Dropping is far ahead: after 3 nominal epochs (2.2 real passes) it is 8.4 points above plain SFT at pass@1, four standard errors, and 9.6 points above it at pass@8. The two runs are identical until step 192, so the whole difference comes from the last 36 steps of the run, during which Plateau Dropping trains on the half of the tokens whose loss is still moving. Third, with more passes the gap closes, and the way it closes is sharpening on Plateau Dropping's side: at the mid-schedule checkpoint (10 nominal epochs) it is about 2 points ahead at both pass@1 and pass@8 (about one standard error); from there to 20 epochs its pass@1 stays within noise (47.1% to 45.8%, −1.4 ± 1.6 paired) while its pass@8 falls (74.7% to 72.7%, −2.0 ± 1.4), whereas plain SFT's pass@1 is unchanged (45.0% to 44.9%) and its pass@8 keeps rising (72.4% to 74.5%, +2.0 ± 2.8). The two curves cross at pass@8 and end level at pass@1, with no coverage advantage for Plateau Dropping at 20 epochs (72.7% against 74.7%). TailSFT at 3 epochs lands with plain SFT (Table 1). The dedicated 10-epoch runs, and the clipping ablation of Section 3.3, are being evaluated.

**Why it helps at 3 epochs.** We parsed the evaluation trajectories of the 3-epoch and 20-epoch models (Table 4; 3-epoch rows: about 470 attempts per model on the 65 tasks finished at the time of the analysis; 20-epoch rows: about 500 attempts per model). At 3 epochs the plain-SFT agent is still half-trained as an agent: it writes 55% longer reasoning per step, so it fits fewer steps before running out of context; it breaks the tool-call format often enough that 1 attempt in 9 ends on three malformed calls in a row; it runs the repository's tests half as often, rarely after its last edit; and it submits a third less often. The Plateau Dropping agent at the same step count already looks like the 20-epoch models on every one of these measures, and its gain splits between better fix quality (pass given every reference file edited: 44.5% against 35.2%) and slightly better localisation (75.5% against 67.6% of attempts edit every reference file). By 20 epochs plain SFT has learned the same things from repetition and the two agents behave alike. These are correlations from one analysis of the evaluation trajectories, not a causal test.

**Table 4.** Behaviour of the evaluation attempts on SWE-bench Verified (per attempt; "gold files" are the files the reference fix edits).

| | SFT, 3 epochs | Plateau Dropping, 3 epochs | SFT, 20 epochs | Plateau Dropping, 20 epochs |
|---|---|---|---|---|
| pass rate on these attempts, % | 24.4 | 34.0 | 45.5 | 45.5 |
| attempt ends by submitting, % | 20.9 | 34.9 | 54.3 | 54.7 |
| attempt ends on 3 consecutive malformed tool calls, % | 11.5 | 0.9 | 1.3 | 1.8 |
| attempt ends by running out of context, % | 67.2 | 63.4 | 44.3 | 43.5 |
| malformed tool calls per attempt | 1.9 | 0.6 | 1.0 | 0.9 |
| reasoning characters per step | 3,180 | 2,040 | 2,340 | 2,210 |
| steps per attempt (median) | 171 | 205 | 144 | 147 |
| test runs per attempt | 8.5 | 14.9 | 15.0 | 15.8 |
| ran tests after its last edit, % | 18.0 | 32.6 | 44.5 | 45.9 |
| edited every gold file, % | 67.6 | 75.5 | 78.8 | 79.8 |
| pass given every gold file edited, % | 35.2 | 44.5 | 55.5 | 55.6 |

### 3.2 Terminal benchmarks

![Figure 3](/assets/img/blog/plateau-dropping/fig3_tb2_corrected.png)

Figure 3. Terminal-Bench 2.0, pass@1 and pass@8 (%) against training length (66, 220 and 440 steps) for plain SFT (red) and Plateau Dropping (green), averaged over 8 seeds; dashed: the base model (5 seeds, so no pass@8 line). Each panel uses the tasks with at least k valid attempts for every model shown (88 at pass@1, 72 at pass@8).
{: .caption}

![Figure 4](/assets/img/blog/plateau-dropping/fig4_tblite_corrected.png)

Figure 4. TB-Lite, pass@1 and pass@8 (%) against training length (66, 220 and 440 steps) for plain SFT (red) and Plateau Dropping (green), averaged over 8 seeds; dashed: the base model. Each panel uses the tasks with at least k valid attempts for every model shown (97 at pass@1, 79 at pass@8).
{: .caption}

**Table 5.** TB-Lite, pass@1 and pass@8 in %, averaged over 8 seeds, matched task set across the fine-tuned models (97 tasks at pass@1, 80 at pass@8; the base model is scored on its own task set). The last column is the paired difference, Plateau Dropping minus SFT, with its standard error over tasks.

| training length | model | pass@1 | pass@8 | PD − SFT at pass@1 / pass@8 |
|---|---|---|---|---|
| none | base model (own task set) | 16.6 | 52.2 | |
| 3 epochs (66 steps) | + SFT | 27.3 | 68.8 | |
| 3 epochs (66 steps) | + Plateau Dropping | 25.5 | 62.5 | −1.8 (1.9) / −6.2 (4.5) |
| 10 epochs (220 steps) | + SFT | 34.0 | 70.0 | |
| 10 epochs (220 steps) | + Plateau Dropping | **38.9** | **76.2** | +4.9 (1.7) / +6.2 (3.3) |
| 20 epochs (440 steps) | + SFT | 33.2 | 73.8 | |
| 20 epochs (440 steps) | + Plateau Dropping | 30.3 | 71.2 | −3.0 (1.7) / −2.5 (3.1) |

**Table 6.** Terminal-Bench 2.0, pass@1 and pass@8 in %, averaged over 8 seeds, matched task set across the fine-tuned models (88 tasks at pass@1, 65 at pass@8). The last column is the paired difference, Plateau Dropping minus SFT, with its standard error over tasks.

| training length | model | pass@1 | pass@8 | PD − SFT at pass@1 / pass@8 |
|---|---|---|---|---|
| none | base model | 8.8 | 30.3 | |
| 3 epochs (66 steps) | + SFT | 10.2 | 29.2 | |
| 3 epochs (66 steps) | + Plateau Dropping | 12.5 | 32.3 | +2.3 (1.4) / +3.1 (4.4) |
| 10 epochs (220 steps) | + SFT | 16.9 | 36.9 | |
| 10 epochs (220 steps) | + Plateau Dropping | **18.7** | **40.0** | +1.8 (1.5) / +3.1 (3.8) |
| 20 epochs (440 steps) | + SFT | 14.5 | 33.8 | |
| 20 epochs (440 steps) | + Plateau Dropping | 15.1 | 32.3 | +0.6 (1.4) / −1.5 (4.1) |

The terminal benchmarks show the same shape as SWE-bench Verified, shifted to a somewhat later budget. At 66 steps (2.3 passes) nothing separates the two arms: Plateau Dropping is slightly ahead on Terminal-Bench 2.0 and behind on TB-Lite, both within about two standard errors, and at that length it has only been dropping tokens for the last few steps. At 220 steps it is ahead on both benchmarks: +4.9 points of pass@1 on TB-Lite (2.9 standard errors) and +6.2 at pass@8, and +1.8 at pass@1 and +3.1 at pass@8 on Terminal-Bench 2.0. At 440 steps the picture reverses, and again it is Plateau Dropping that sharpens: its pass@1 falls below its own 220-step result (TB-Lite 38.9% to 30.3%, a paired drop of 8.6 ± 1.9; Terminal-Bench 2.0 18.7% to 15.1%, −3.7 ± 1.3) and so does its pass@8 (TB-Lite −6.6 ± 4.1, Terminal-Bench 2.0 −2.4 ± 4.2), while plain SFT holds at pass@1 (TB-Lite 34.0% to 33.2%; Terminal-Bench 2.0 16.9% to 14.5%, −2.4 ± 1.4) and its pass@8 rises on TB-Lite (+3.3 ± 3.9) though it falls on Terminal-Bench 2.0 (−6.0 ± 3.2). Section 3.3 describes one mechanism that could produce the Plateau Dropping decline. Taken together with the SWE results (+8.4 at 228 steps, +2 at 760, level at 1,520), Plateau Dropping's advantage is largest in the short-to-mid training range; with long training it fades on the SWE data and reverses on the terminal data, in both cases through a loss of coverage (pass@8) at roughly constant pass@1.

### 3.3 Open question: gradient clipping

An audit of the runs found one systematic difference between the two arms that is not part of the method. Plateau Dropping's loss is a mean over the kept tokens, which are the harder ones, so once dropping starts its gradient norm sits at 1.3 to 2.9 and is clipped at the trainer's global norm of 1.0. On the SWE data that happens on 50% of the steps of the 760-step run and 86% of the steps of the 1,520-step run, against 3 to 4% of plain SFT's steps (median norm 0.5). On the terminal data, where the kept fraction falls to about 15%, every logged step of the second half of the 220- and 440-step Plateau Dropping runs is clipped (82% and 91% of all steps; median second-half norm 1.7 and 2.6), against 0 to 5% for plain SFT (median 0.55). Under Adam a constant rescaling of the gradient mostly cancels, but clipping that varies from step to step does change the updates, so the convergence of the two arms at long training, and the decline of the 440-step terminal run, could be partly a clipping artifact. A 10-epoch Plateau Dropping run on the SWE data with the clip raised to 4.0 (so that none of its steps is clipped) has been trained; its training loss is indistinguishable from the clipped run's (0.347 against 0.346 over all tokens at the end), and its evaluation is [pending]. The same ablation on the terminal data has not been run.

## 4. Related work

**Token selection with a reference model.** Rho-1 ([Lin et al., 2024](https://arxiv.org/abs/2404.07965)) trains only on tokens whose loss is high relative to a reference model. Plateau Dropping needs no reference model, and its selection follows the model's own loss history.

**Example selection by loss.** TailSFT, Selective Backprop ([Jiang et al., 2019](https://arxiv.org/abs/1910.00762)) and loss truncation ([Kang and Hashimoto, 2020](https://arxiv.org/abs/2004.14589)) drop or prioritise whole examples. Plateau Dropping selects individual tokens instead, and drops stuck tokens as well as learned ones.

**Training dynamics.** Forgetting events ([Toneva et al., 2019](https://arxiv.org/abs/1812.05159)) and early-training scores ([Paul et al., 2021](https://arxiv.org/abs/2107.07075)) rank examples by how they behave during training. Plateau Dropping does this online, per token, over two passes.

**Down-weighting easy cases.** Focal loss ([Lin et al., 2017](https://arxiv.org/abs/1708.02002)) down-weights well-classified examples. Its hard-threshold, per-token version, dropping every token above a probability threshold, is the snapshot-of-confidence approach that Section 1 argues against.

**Minority tokens in RL.** Updating only the 20% highest-entropy tokens matches full updates in RL for reasoning ([Wang et al., 2025](https://arxiv.org/abs/2506.01939)), and DAPO ([Yu et al., 2025](https://arxiv.org/abs/2503.14476)) discards sample groups that carry no learning signal. Plateau Dropping is an SFT counterpart.

## 5. Conclusion

Plateau Dropping is a small change to standard SFT. The trainer remembers each token's loss from its last two visits, and once that loss stops changing, the token is removed from the training loss until it changes again. What it buys is speed: it gets the model to the trained agent's behaviour with far fewer passes over the data. After 3 nominal epochs on the SWE data it is 8 to 10 points above plain SFT on SWE-bench Verified (31.9% against 23.4% at pass@1, 67.0% against 58.3% at pass@8), a model that has reached the format, testing and submission habits plain SFT only acquires with many more passes, and at 220 steps on the terminal data it is 4.9 points ahead on TB-Lite and 1.8 on Terminal-Bench 2.0. With more passes the gap closes, and on Plateau Dropping's side it closes by sharpening: from 10 to 20 nominal epochs its pass@1 holds while its pass@8 falls on all three benchmarks, so that after 20 epochs on the SWE data the two are level at pass@1 and pass@8, and on the terminal data the 440-step Plateau Dropping run sits below its own 220-step result while plain SFT holds. Sequence-level filtering (TailSFT) does not help at any budget we tested. Whether gradient clipping, which hits the long Plateau Dropping runs on nearly every step, is what holds them back is being tested, and the next experiments are the clipping ablation on both datasets and the dedicated 10-epoch SWE runs.

## 6. Training bugs we found along the way

Our first runs were on one cluster. When we ported the setup to a second cluster to reproduce them, the Plateau Dropping result came out 4 points lower, and the search for that difference found two bugs in the trainer that had affected every one of those runs, plus two smaller issues. All four are fixed; both clusters now run byte-identical trainer code, and every run in this post passed an audit (no restarts, no non-finite losses, plain and Plateau Dropping losses identical step by step until dropping starts, every exported checkpoint validated against the training checkpoint).

![Figure 5](/assets/img/blog/plateau-dropping/fig_bugs_loss_curves.png)

Figure 5. Left: training loss over all tokens on the SWE data for our first, buggy runs (dashed and dotted, both clusters, plain SFT and Plateau Dropping in two data orders) and the fixed runs (solid): the buggy runs stay at 0.54 to 0.55 for 1,520 steps; the fixed runs fall to 0.15 to 0.18. Right: the fraction of tokens Plateau Dropping keeps, and the loss of the dropped tokens, in the buggy and the fixed runs.
{: .caption}

**1. The master weights were bf16.** The script that imported the base checkpoint wrote the weights in bf16, and the trainer kept whatever precision it loaded, so the optimizer applied its fp32 updates to bf16 weights and rounded the result back to bf16 at every step. A bf16 number has an 8-bit mantissa, so any update smaller than about 0.4% of a weight's magnitude rounds to zero, which is most Adam updates at a learning rate of 1e-4. After 1,520 steps, 92 to 95% of the weights were bit-identical to the base model; the training loss stopped at 0.55 (Figure 5) and the models barely changed. The fixed trainer keeps fp32 master weights (the standard mixed-precision recipe: bf16 compute, fp32 weights and optimizer state); in the fixed runs 65% of the weights have changed after 228 steps, 83% after 760 and 89% after 1,520, measured on the bf16 export. Every one of our first runs was trained this way: plain SFT, Plateau Dropping, TailSFT and its variants, the ablations, the runs from a stronger starting checkpoint, and the SFT starting points of the RL runs. (The Qwen3-8B experiments we ran alongside used a different trainer and were not affected.)

**2. The mixture of experts dropped token-to-expert assignments.** The model was trained with an expert capacity factor of 1.0, so whenever a batch routed more tokens to an expert than its even share, the excess assignments were dropped. On this data 14 to 19% of assignments were dropped, and the fraction depended on the hardware's rounding (18.6% on the GH200 cluster, 15.5% on the GB200 cluster), which cost 0.067 to 0.082 nats of loss and is what made the two clusters disagree. With the capacity factor raised to 2.0 the overflow is 0.16% at the start of a run and 0 by the end, the two clusters agree to four decimals, and the trainer's loss matches an independent inference implementation to 0.0002.

**3. The TailSFT reference pass zeroed the router bias.** The forward-only scoring pass that produces TailSFT's per-sequence reference losses applied a zero router bias instead of the frozen one from the checkpoint. The effect on the references was small (per-document error with a standard deviation of 0.006 and a rank correlation of 0.998 with the correct references), so it did not change the TailSFT result; the TailSFT runs in Table 1 use references from the fixed pass.

**4. Epochs were overstated.** We first counted passes from token totals and ignored row padding, so what we called "10 epochs" on the terminal data was 7.6 real passes and "20 epochs" on the SWE data was 14.8. This post keeps the nominal names and gives the real pass counts in Section 2.2.

**What the bugs did to our conclusions.** In our first runs the training loss flattened after one pass, and plain SFT barely moved the base model (19.5% pass@1 on SWE-bench Verified after 20 nominal epochs, against 19.3% for the base) while Plateau Dropping reached 31.1%. Both were the first bug: with the fixed trainer the loss falls throughout and plain SFT reaches 44.9% pass@1 and 74.7% pass@8. We do not have a complete account of why Plateau Dropping still improved under the bug. Its gradients are larger, because its loss is averaged over fewer, harder tokens, which would let more of its updates survive the rounding; but the fraction of weights that moved at all was the same in both arms, so those runs compared two models that had barely trained, and their 10-point gap at 20 epochs does not survive the fix: with both arms training properly, the two are level (Table 3). What survives is the short-training advantage, which is larger and better supported than anything in the buggy runs (+8.4 points at pass@1 and +9.6 at pass@8 after 3 epochs on SWE-bench Verified; +4.9 and +6.2 on TB-Lite at 220 steps), and a small edge mid-training on the SWE data. The ablations, the runs from a stronger starting checkpoint, and a comparison of RL started from plain SFT and from Plateau Dropping were all trained under the bugs, so we leave them out of this post until they are rerun (the RL runs themselves were unaffected; their SFT starting points were not). The fixes (fp32 master weights, the capacity-factor setting, the corrected scoring pass, and a gradient-clipping setting for the ablation of Section 3.3) are in our fork of the trainer.

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
