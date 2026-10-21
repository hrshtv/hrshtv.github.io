---
layout: post
title: "Plateau Dropping: per-token loss-plateau gating for agentic SFT"
description: "Dropping tokens from the SFT loss once their loss plateaus makes longer fine-tuning pay off for agents, and gives a better starting point for RL."
authors: Harshit Varma
math: true
---

**Summary.** SFT sharpens a model: every extra pass pushes probability further onto what it already predicts well. That raises pass@1, but it can shrink the range of solutions the model still samples, which is what reinforcement learning (RL) needs to build upon. So a heavily fine-tuned model can potentially be a worse starting point for RL. In our agentic SFT setup, the loss flattens after one pass and two thirds of the training tokens are already predicted with probability above 0.9 before any training, suggesting that further passes mostly sharpen what the model already knows. Filtering out whole sequences the model already fits (TailSFT) and its variants did not help. We propose *Plateau Dropping*, a per-token alternative: during SFT, it tracks each training token's loss across passes over the data and removes the token from the training loss once its loss has stopped changing, so each update is driven only by tokens the model is still learning. With the same data, this approach raises Terminal-Bench 2.0 pass@1 from 9.8% to 14.1%, while pass@8 stays about the same (31.8% against 33.3%), and on SWE-bench Verified it raises pass@1 from 19.5% to 31.1% and pass@8 from 47.7% to 61.6%. It is also a better starting point for RL: on its own it matches the best RL checkpoint from plain SFT, and RL from it improves further.

---

## 1. Introduction and motivation

We fine-tuned a 67B MoE model (with about 2B active parameters) on 6.3k successful agent trajectories covering 1.6k command-line tasks. The training loss averaged 0.48 over the first pass and 0.41 to 0.46 in each of the next seven (Figure 1, left). Ten epochs were barely better than three: 11.8% against 9.8% pass@1 on Terminal-Bench 2.0, and 21.4% against 21.8% on TB-Lite.

![Figure 1](/assets/img/blog/plateau-dropping/fig1_motivation.png)

Figure 1. Left: average training loss per pass over the data, plain SFT. Right: how confidently the model, before fine-tuning, already predicts the tokens it is trained on; the dark segment is the share above 0.99.
{: .caption}

**SFT can sharpen, but might reduce coverage.** Prior work finds that long SFT concentrates probability on a narrower set of outputs, and that this hurts coverage: during SFT on reasoning data, pass@1 keeps improving while pass@k deteriorates as output diversity collapses ([Dang et al., 2025](https://arxiv.org/abs/2504.10478)); training longer with cross-entropy lowers pass@N through overconfidence ([Chen et al., 2025](https://arxiv.org/abs/2502.07154)); and checkpoints that keep more diversity, or that are fine-tuned less aggressively, make better starting points for reinforcement learning (RL) ([Dang et al., 2025](https://arxiv.org/abs/2504.10478); [clipping low-probability tokens in SFT, ICML 2026](https://icml.cc/virtual/2026/poster/61171)). Our runs are consistent with this: from 3 to 10 epochs, Terminal-Bench 2.0 pass@1 rose from 9.8% to 11.8% while pass@8 fell from 33.3% to 31.8%.

**TailSFT** ([Malladi et al., 2026](https://arxiv.org/abs/2608.25756)) targets this issue. It drops training sequences the model already fits, judged by how far each sequence's loss has fallen below the base model's, so that learning concentrates on the under-modelled tail. On OLMo-3 7B it reports up to 17 points of pass@16 on math and coding, and up to 4 points of pass@1 after RL.

**TailSFT did not work on our agent data.** We ran it per trajectory, as published, and with two agent-specific variants: per step (reasoning plus its command) and per span (reasoning and command scored separately). Each variant dropped 50% of its units per batch and trained for 3 epochs.

**Table 1.** TailSFT and its agent extensions, pass@k in %. All runs train for 3 epochs (66 steps). Terminal-Bench 2.0 uses 89 tasks at pass@1, 87 at pass@3 and 83 at pass@8; TB-Lite uses 99, 98 and 94. pass@8 needs 8 seeds, so the 3-seed variants have none.

| model | seeds | TB2 pass@1 | TB2 pass@3 | TB2 pass@8 | TB-Lite pass@1 | TB-Lite pass@3 | TB-Lite pass@8 |
|---|---|---|---|---|---|---|---|
| SFT, 3 epochs | 8 | 9.7 | 20.4 | 33.7 | 21.6 | 39.0 | 55.3 |
| TailSFT, per trajectory (drop share ramped up over the run) | 8 | 10.4 | 19.4 | 30.1 | 22.1 | 39.4 | 57.4 |
| TailSFT, per trajectory (full drop share from step 5) | 3 | 9.4 | 17.2 | – | 21.7 | 38.8 | – |
| TailSFT, per agent step | 3 | 8.6 | 21.8 | – | 22.9 | 40.8 | – |
| TailSFT, per span (reasoning and command separately) | 3 | 8.2 | 16.1 | – | 20.2 | 42.9 | – |

None moved pass@1. At pass@3, no variant improved on SFT on both benchmarks by more than a couple of points, well within the noise of three seeds, and per-trajectory TailSFT, with 8 seeds, gave no coverage gain either: pass@8 was 30.1% against 33.7% on Terminal-Bench 2.0, and 57.4% against 55.3% on TB-Lite. Scoring reasoning and commands separately did not help either; the two start from very different losses (0.82 against 0.26 on average).

**What the tokens show.** Before training, the model gives a probability above 0.9 to 65% of the training tokens and above 0.99 to 52% (Figure 1, right). For shell commands, 78% are above 0.9. The 2% below 0.01 are mostly ordinary words (such as "Check", "which", "Step"), not identifiers or hashes. This led us to select individual tokens rather than whole sequences, and to judge each token by its own loss history rather than by a snapshot of confidence. We propose *Plateau Dropping*, which does both: it removes a token from the training loss once that token's loss has plateaued. The token stays in its sequence, so its loss is still measured each time the sequence is revisited, and if that loss starts changing again, the token goes back into the training loss.

## 2. Method

### 2.1 The plateau test

For each token position, the trainer keeps the token's loss at its last two visits and a visit count. For our 6,301 trajectories this bookkeeping takes about 2 GB of host (CPU) memory in each training process, about 8 GB on a four-GPU node. From the third visit on, if the loss changed by less than 10% between each of the last two visits (relative change, with a floor of 0.05), the token is dropped: its term is removed from the training loss, so it no longer contributes to the parameter update, but it still stays in the input as context for the tokens around it. If its loss moves again, it is trained again. Learned and stuck tokens are treated alike. Everything else is standard SFT, and the loss is averaged over the kept tokens.

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

where $$P_i$$ is treated as a constant, so no gradient flows through the selection. We use $$\tau = 0.1$$ and $$\varepsilon = 0.05$$. Plain SFT is the special case $$P_i = 0$$ for every token. The history is updated at every visit, including for dropped tokens, so a dropped token whose loss moves again is trained again at its next visit (Figure 2).

![Figure 2](/assets/img/blog/plateau-dropping/fig2_method.png)

Figure 2. Plateau Dropping on real tokens, from a traced 10-epoch run. (a) The token "maybe" in "…<code>curl -sI $ENDPOINT &#124; head -5</code> maybe. But we risk…" (training step of each visit in brackets): it is trained while its loss falls; at its 4th and 5th visits both of the last two relative changes are below 10%, so it is dropped (hollow), while still being read as context and having its loss measured; at its 6th visit its loss moves by 14% and it is trained again. (b) Fourteen consecutive reasoning tokens at step 140: plateaued tokens (white) are left out of the average, so the loss over this stretch is 0.68, against 0.35 for the plain SFT average over all of them.
{: .caption}

### 2.2 What gets dropped

Once tokens have three visits, about 60% of each batch is dropped: roughly 55% learned (loss below about 0.1) and 5% stuck at higher loss. The kept tokens have a loss near 0.8. The tokens dropped at any given step have a loss near 0.2 (Figure 3), and the loss over all tokens tracks plain SFT. The dropped set changes over time as tokens re-enter, so this does not show that a fixed set of tokens stays learned.

![Figure 3](/assets/img/blog/plateau-dropping/fig2_not_forgotten.png)

Figure 3. Training loss over 220 steps (training curves; no evaluation seeds). Red: plain SFT, over all tokens. Green: Plateau Dropping, over all tokens (medium), tokens still trained (dark) and dropped tokens, which are excluded from the training loss (light, dashed). The red curve and the medium-green curve overlap: both runs start from the same model and see the same batches in the same order, so they are identical until dropping starts around step 30, and afterwards the Plateau Dropping curve sits only 0.002 to 0.005 higher, far less than the step-to-step variation between batches.
{: .caption}

Our data loader samples rows with replacement, so epochs are nominal: one full pass is about 25.6 steps, and 66, 220 and 440 steps are about 2.6, 8.6 and 17.2 passes, with irregular gaps between visits. The plateau test counts visits, not epochs.

**Which tokens get dropped.** We traced every trained token of 300 randomly chosen trajectories through a 10-epoch run of each method, with the same data order, and grouped the tokens by how well the model predicted them at their first visit (Figure 4). The easy tokens, two thirds of the total, are almost all dropped as learned (87%). Medium and hard tokens are mostly kept, and on them Plateau Dropping brings the loss down further than plain SFT: by the ninth visit it falls 24% on medium tokens against 13% for plain SFT, and 15% against 12% on hard tokens. On the 7% of very hard tokens, 40% end up dropped as stuck, and their loss stays higher than under plain SFT (down 4% against 10%). Shell commands and their tool-call syntax are mostly learned and dropped (78%); 80% of the stuck tokens are in the free-form reasoning, mostly common words such as "the" and "and".

![Figure 4](/assets/img/blog/plateau-dropping/fig_token_analysis.png)

Figure 4. Token-level trace of 300 training trajectories over 10-epoch runs (training tokens, one run per method). (a) Where tokens end up under Plateau Dropping at the end of training, by the probability the model gave them at their first visit. (b) Change in loss since the first visit on the same tokens, for plain SFT (red) and Plateau Dropping (green): medium (solid), hard (dashed) and very hard (dotted) tokens.
{: .caption}

### 2.3 Setup

- **Model.** A 67B mixture-of-experts model with about 2B parameters active per token: [open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21](https://huggingface.co/open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21). Section 3.4 also uses a stronger checkpoint of the same model: [open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02](https://huggingface.co/open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02).
- **Terminal data.** 6,301 successful trajectories (45M tokens, 1,610 tasks): 5,324 from the public [allenai/tmax-sft](https://huggingface.co/datasets/allenai/tmax-sft) dataset, written by a 27B teacher, and 977 of the base model's own successes on synthetic terminal tasks. Context 32k, learning rate 1e-4.
- **SWE data.** 9,229 resolved trajectories (7,005 tasks) from the public [nvidia/Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) dataset, restricted to one shell command per step to match our agent (319M tokens). Context 64k.
- **Compared models.** SFT for 3 epochs (66 steps) and 10 epochs (220), and Plateau Dropping for 10 and 20 epochs (220 and 440). On the SWE data, SFT and Plateau Dropping for 20 epochs (1,520 steps).
- **Harness.** We use the [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) harness everywhere: all training trajectories are in its format, and all evaluations and RL rollouts run in it. At each step the model writes its reasoning and one shell command, then sees the command's output.

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

We use all 89 tasks of Terminal-Bench 2.0, TB-Lite (100 command-line tasks), and a fixed random 100 tasks of SWE-bench Verified. Every model runs in mini-swe-agent at temperature 1, with three times the default time limit. Each seed is one independent attempt at every task; we use 8 seeds per model, and 3 for the ablations, the RL checkpoints and three of the TailSFT variants (the "seeds" column of each table). A task is solved only with full credit. An attempt that hits the time limit is scored by the tests on whatever the agent left, so in rare cases (6 of about 400 timed-out attempts) it still counts as solved; infrastructure crashes are discarded. All numbers are averaged over seeds: pass@k is the unbiased estimate from the seeds' attempts, and each column uses the tasks with at least k valid attempts for every model in its comparison. One comparison per benchmark covers Tables 3 to 5, and on SWE-bench Verified also Table 7, so it includes the stronger-base models of Section 3.4; Table 7's terminal rows are matched among their own three models, and on TB-Lite the base model of Tables 3 and 4 is scored on its own task set.

One change to the SWE-bench Verified grader: before running the tests, the standard Harbor task script deletes every file that git does not track, which also deletes new source files the agent created as part of its fix, so a correct fix that adds a file is graded as a failure. Our test script deletes only the untracked files that the benchmark's own test patch touches, and is otherwise unchanged. Every model in this post is graded the same way, so the comparisons between them are unaffected, but our absolute SWE-bench Verified numbers can be higher than those from the stock Harbor tasks.

## 3. Results

In all figures, grey is a model without fine-tuning, red is plain SFT (darker red for 10 epochs) and green is Plateau Dropping.

### 3.1 Terminal benchmarks

![Figure 5](/assets/img/blog/plateau-dropping/fig3_tb2.png)

Figure 5. Terminal-Bench 2.0, pass@1, pass@2, pass@4 and pass@8 (%), averaged over 8 seeds.
{: .caption}

![Figure 6](/assets/img/blog/plateau-dropping/fig4_tblite.png)

Figure 6. TB-Lite, pass@1, pass@2, pass@4 and pass@8 (%), averaged over 8 seeds. The base model is scored on its own task set (98 tasks, 92 at pass@8), not matched to the fine-tuned models.
{: .caption}

**Table 3.** Terminal-Bench 2.0, pass@k in % (88, 88, 88 and 66 tasks per column).

| model | seeds | pass@1 | pass@2 | pass@4 | pass@8 |
|---|---|---|---|---|---|
| base model | 8 | 8.8 | 14.2 | 20.7 | 30.3 |
| + SFT, 3 epochs | 8 | 9.8 | 15.8 | 23.6 | 33.3 |
| + SFT, 10 epochs | 8 | 11.8 | 17.4 | 23.4 | 31.8 |
| + Plateau Dropping, 10 epochs | 8 | 11.7 | 18.7 | 26.1 | 33.3 |
| + Plateau Dropping, 20 epochs | 8 | **14.1** | **20.4** | **26.7** | 31.8 |

**Table 4.** TB-Lite, pass@k in % (98, 98, 98 and 90 tasks per column).

| model | seeds | pass@1 | pass@2 | pass@4 | pass@8 |
|---|---|---|---|---|---|
| base model (own task set: 98, 98, 98 and 92 tasks) | 8 | 16.6 | 26.4 | 38.5 | 52.2 |
| + SFT, 3 epochs | 8 | 21.8 | 32.9 | 44.1 | 54.4 |
| + SFT, 10 epochs | 8 | 21.4 | 32.0 | 43.0 | 55.6 |
| + Plateau Dropping, 10 epochs | 8 | 24.1 | 35.3 | 47.4 | **60.0** |
| + Plateau Dropping, 20 epochs | 8 | **26.6** | **38.8** | **49.6** | 57.8 |

At 20 epochs, Plateau Dropping achieves the highest pass@1 on both benchmarks: 14.1% on Terminal-Bench 2.0 and 26.6% on TB-Lite, 4.3 and 4.8 points above 3-epoch SFT. Against 10-epoch SFT (220 steps), it is 5.2 points ahead on TB-Lite. At exactly matched steps (220), Plateau Dropping is level with SFT on Terminal-Bench 2.0 (11.7% against 11.8%) and ahead on TB-Lite (24.1% against 21.4%). The gain is largest at pass@1 and pass@2 and narrows by pass@8.

### 3.2 SWE-bench Verified

![Figure 7](/assets/img/blog/plateau-dropping/fig5_swebench.png)

Figure 7. SWE-bench Verified (random 100 tasks), pass@1, pass@2, pass@4 and pass@8 (%), averaged over 8 seeds. Both fine-tuned models trained for 20 epochs on the SWE data.
{: .caption}

**Table 5.** SWE-bench Verified, pass@k in %. pass@1, pass@2 and pass@4 use all 100 tasks; pass@8 uses the 86 tasks of the original comparison ("–": too few tasks with 8 valid attempts for every model of this table). Data order 2 repeats the same training with the training data sampled in a different order.

| model | seeds | pass@1 | pass@2 | pass@4 | pass@8 |
|---|---|---|---|---|---|
| base model | 8 | 19.3 | 28.0 | 37.7 | 47.7 |
| + SFT, 3 epochs | 8 | 21.4 | 31.5 | 41.4 | – |
| + SFT, 3 epochs, data order 2 | 8 | 19.4 | 28.9 | 38.9 | – |
| + SFT, 20 epochs | 8 | 19.5 | 28.6 | 37.4 | 47.7 |
| + TailSFT per trajectory, 3 epochs | 8 | 20.6 | 30.7 | 41.5 | – |
| + TailSFT per agent step, 3 epochs | 8 | 20.8 | 30.9 | 40.7 | – |
| + Plateau Dropping, 20 epochs | 8 | 31.1 | 41.7 | 51.6 | **61.6** |
| + Plateau Dropping, 20 epochs, data order 2 | 8 | **31.7** | **43.5** | **54.1** | – |

Plain SFT barely moves the base model at either length: 21.4% and 19.4% pass@1 after 3 epochs (two data orders) and 19.5% after 20. Plateau Dropping on the same data reaches 31.1% and 31.7% in the two data orders, about 10 points above plain SFT at its best length (+9.8 and +10.3 points over 3-epoch SFT with data order 1, paired standard errors 1.7 and 1.9), and adds about 14 points at pass@8 over the base. TailSFT, run for 3 epochs as in Table 1, lands with plain SFT here too (20.6% per trajectory, 20.8% per agent step). All models produce well-formed tool calls (fewer than 2% of steps without one), and 40 epochs of Plateau Dropping did not improve on 20.

**Why it helps on SWE-bench.** We compared the agent trajectories of the two Plateau Dropping runs with those of 3-epoch SFT (both data orders) and the base model, about 800 attempts each. Plateau Dropping does not win by submitting more often (39 to 42% of its attempts end in a submission, against 33 to 39%), but by fixing correctly once it has found the right place: among attempts that edit every file the reference fix edits, 42 to 44% pass, against 28 to 33% for 3-epoch SFT. Alongside this, its reasoning per step is short and stays short (about 1.6 thousand characters, against 2.7 thousand for SFT, whose reasoning grows about threefold over a long attempt), so it fits more steps into the same context (a median of about 245 steps per attempt, against 170 to 185), and it spends them on checking: it runs the repository's tests about 2.4 times as often (19 to 20 runs per attempt against 8) and more often after its last edit (40 to 42% of attempts against 28%). These are correlations from one analysis of the evaluation trajectories, not a causal test. Plateau Dropping also has a failure mode of its own: 5% of its commands repeat the previous command exactly, against 0.4% for SFT.

### 3.3 Ablations

The ablations in Table 6 use the terminal data and 3 seeds. We test whether it matters which tokens are dropped, or only how many. Random dropping removes a random 65% of tokens. Confidence dropping removes every token the model already predicts with probability above 0.9, the snapshot-of-confidence approach that Section 1 argues against. Neither has a built-in start point, so both start at step 65, when Plateau Dropping's drop share has levelled off at about 60%, and all three train for 220 steps. Plateau Dropping itself can only drop a token from its third visit on, so its dropping begins around step 30.

![Figure 8](/assets/img/blog/plateau-dropping/fig6_ablations.png)

Figure 8. pass@1 (%) for each variant, averaged over 3 seeds. The dashed line marks 3-epoch SFT.
{: .caption}

**Table 6.** Ablations, pass@k in % (Terminal-Bench 2.0: 88 tasks at pass@1, 74 at pass@3; TB-Lite: 98 and 85).

| model | seeds | TB2 pass@1 | TB2 pass@3 | TB-Lite pass@1 | TB-Lite pass@3 |
|---|---|---|---|---|---|
| + SFT, 3 epochs | 3 | 9.1 | 17.6 | 22.8 | 42.4 |
| + SFT, 10 epochs | 3 | 11.7 | 23.0 | 22.1 | 47.1 |
| random drop of 65% of tokens | 3 | 11.7 | 25.7 | 22.8 | 45.9 |
| confidence drop (p > 0.9) | 3 | 7.8 | 16.2 | 13.6 | 29.4 |
| + Plateau Dropping, 10 epochs | 3 | 12.3 | 21.6 | 26.9 | 50.6 |
| + Plateau Dropping, 20 epochs | 3 | 15.5 | 27.0 | 26.9 | 50.6 |
| smoothed-history variant, 20 epochs | 3 | 15.5 | 28.4 | 23.1 | 42.4 |
| + Plateau Dropping, 40 epochs | 3 | 14.4 | 33.8 | 27.6 | 50.6 |

- **Random dropping** of 65% of tokens from step 65 matches SFT with the same steps. Since it also starts dropping later than Plateau Dropping, this does not fully separate which tokens are dropped from when dropping starts.
- **Confidence dropping** of every token above probability 0.9 is the worst variant.
- **A smoothed plateau test** (moving-average loss history) keeps the Terminal-Bench 2.0 gain but loses TB-Lite.
- **40 epochs** matches 20.

On the SWE data we also tested two changes to the rule itself, each a single 20-epoch training run evaluated with 8 seeds. Letting a dropped token back in only when its loss rises (instead of on any change) gives 28.5% pass@1, and keeping dropped tokens at a quarter of their weight instead of removing them also gives 28.5%; both are 2.6 points below the rule as described (31.1%, paired standard error 1.9). The gap widens at larger k for the down-weighted variant (45.9% against 51.6% at pass@4). Neither change helps, and removing plateaued tokens completely works at least as well as keeping part of their signal.

### 3.4 A stronger starting model

We repeated the 20-epoch runs from a stronger checkpoint of the same model, which already achieves a pass@1 of 11.9% on Terminal-Bench 2.0, 24.2% on TB-Lite and 33.7% on SWE-bench Verified. On the terminal data we also trained plain SFT for the same 20 epochs as a control.

**Table 7.** Stronger starting model, pass@k in %. Terminal benchmarks: matched across the three models shown (Terminal-Bench 2.0: 88, 88, 88 and 87 tasks; TB-Lite: 98, 98, 98 and 93). SWE-bench Verified: same task sets as Table 5. "–": not run.

| benchmark | model | seeds | pass@1 | pass@2 | pass@4 | pass@8 |
|---|---|---|---|---|---|---|
| Terminal-Bench 2.0 | stronger base | 8 | 11.9 | 18.1 | 25.2 | 33.3 |
| Terminal-Bench 2.0 | + SFT, 20 epochs | 8 | 11.5 | 18.4 | 26.4 | **35.6** |
| Terminal-Bench 2.0 | + Plateau Dropping, 20 epochs | 8 | **15.3** | **22.5** | **29.4** | **35.6** |
| TB-Lite | stronger base | 8 | 24.2 | 35.8 | 47.5 | 59.1 |
| TB-Lite | + SFT, 20 epochs | 8 | **27.3** | **39.8** | 52.3 | **66.7** |
| TB-Lite | + Plateau Dropping, 20 epochs | 8 | 26.9 | 39.4 | **52.5** | 65.6 |
| SWE-bench Verified | stronger base | 8 | **33.7** | 43.6 | 51.7 | 58.1 |
| SWE-bench Verified | + SFT, 20 epochs | – | – | – | – | – |
| SWE-bench Verified | + Plateau Dropping, 20 epochs | 8 | 32.8 | **44.2** | **54.5** | **64.0** |

On Terminal-Bench 2.0, Plateau Dropping raises pass@1 from 11.9% to 15.3%, while plain SFT for the same 20 epochs leaves it at 11.5%; at pass@8 the two fine-tuned models are level (35.6%). On TB-Lite both raise pass@1, to 26.9% with Plateau Dropping and 27.3% with plain SFT, and both raise pass@8 by about 7 points. On SWE-bench Verified Plateau Dropping does not improve pass@1 (33.7% to 32.8%), while pass@8 rises from 58.1% to 64.0%; plain SFT on the SWE data from this checkpoint has not been run.

### 3.5 RL after fine-tuning

We ran the same RL recipe on terminal tasks from 3-epoch SFT and from 20-epoch Plateau Dropping. The second run is still in progress.

**Table 8.** RL from each fine-tuned model, pass@k in % (Terminal-Bench 2.0: 88 tasks at pass@1, 82 at pass@3; TB-Lite: 98 and 93).

| model | seeds | TB2 pass@1 | TB2 pass@3 | TB-Lite pass@1 | TB-Lite pass@3 |
|---|---|---|---|---|---|
| SFT, 3 epochs | 3 | 9.1 | 18.3 | 22.8 | 38.7 |
| + RL, step 25 | 3 | 13.6 | 25.6 | 25.9 | 43.0 |
| + RL, step 107 (its best) | 3 | 15.5 | 28.0 | 26.5 | 47.3 |
| Plateau Dropping, 20 epochs | 3 | 15.5 | 28.0 | 26.9 | 45.2 |
| + RL, step 21 | 3 | **18.2** | **32.9** | **29.3** | **48.4** |

Plateau Dropping alone matches the best RL checkpoint from SFT (step 107), which makes it a better starting point for RL. After 21 RL steps it is the best model in this study, with a pass@1 of 18.2% on Terminal-Bench 2.0 and 29.3% on TB-Lite.

## 4. Related work

**Token selection with a reference model.** Rho-1 ([Lin et al., 2024](https://arxiv.org/abs/2404.07965)) trains only on tokens whose loss is high relative to a reference model. Plateau Dropping needs no reference model, and its selection follows the model's own loss history.

**Example selection by loss.** TailSFT, Selective Backprop ([Jiang et al., 2019](https://arxiv.org/abs/1910.00762)) and loss truncation ([Kang and Hashimoto, 2020](https://arxiv.org/abs/2004.14589)) drop or prioritise whole examples. Plateau Dropping selects individual tokens instead, and drops stuck tokens as well as learned ones.

**Training dynamics.** Forgetting events ([Toneva et al., 2019](https://arxiv.org/abs/1812.05159)) and early-training scores ([Paul et al., 2021](https://arxiv.org/abs/2107.07075)) rank examples by how they behave during training. Plateau Dropping does this online, per token, over two passes.

**Down-weighting easy cases.** Focal loss ([Lin et al., 2017](https://arxiv.org/abs/1708.02002)) down-weights well-classified examples. Its hard-threshold, per-token version, dropping tokens above probability 0.9, is our worst ablation (Section 3.3).

**Minority tokens in RL.** Updating only the 20% highest-entropy tokens matches full updates in RL for reasoning ([Wang et al., 2025](https://arxiv.org/abs/2506.01939)), and DAPO ([Yu et al., 2025](https://arxiv.org/abs/2503.14476)) discards sample groups that carry no learning signal. Plateau Dropping is an SFT counterpart.

## 5. Conclusion

Plateau Dropping is a small change to standard SFT. The trainer remembers each token's loss from its last two visits, and once that loss stops changing, the token is removed from the training loss until it changes again. With the same data, this makes longer training pay off. Pass@1 rises by 4 to 5 points over standard 3-epoch SFT on two terminal benchmarks, and by 12 points over SFT with the same number of steps on SWE-bench Verified. Coverage improves as well: pass@8 rises from 47.7% to 61.6% on SWE-bench Verified and from 54.4% to 57.8% on TB-Lite. This carries over to RL. The Plateau Dropping checkpoint on its own matches the best checkpoint of RL started from plain SFT (15.5% pass@1 on Terminal-Bench 2.0), and after only 21 RL steps from it, pass@1 reaches 18.2%, above every evaluated checkpoint of RL from plain SFT.

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
