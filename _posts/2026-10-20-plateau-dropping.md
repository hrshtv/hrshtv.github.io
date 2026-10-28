---
layout: post
title: "Plateau Dropping: a simple SFT rule for better agentic post-training"
description: "Dropping each token from the SFT loss once its loss stops changing gives gains over vanilla SFT on long-horizon agentic tasks."
authors: <a href="/">Harshit Varma</a>, <a href="https://lukedhlee.github.io/">Donghyun Lee</a>, <a href="https://people.eecs.berkeley.edu/~alexdimakis/">Alex Dimakis</a>
list_description: false
math: true
---

<div class="post-intro" markdown="1">
<div class="post-summary" markdown="1">

**Summary.** 
Supervised fine-tuning (SFT) has become a key component of post-training recipes for almost every major LLM.
However, this is usually not the final step in standard post-training pipelines.
Several existing works show that although prolonged SFT improves pass@1, it usually ends up hurting diversity / coverage (as measured by pass@k, for some "large" k).
This behavior is not conducive to further post-training of the model as it inhibits exploration. For example, GRPO-based RL pipelines, which usually follow SFT in a lot of post-training pipelines, do not work well with a model that is not diverse enough to start with.
TailSFT is a recent method proposed to tackle this problem. However, we find that TailSFT (and some variants/extensions that we tried) does not help in our long-horizon agentic setup.
We instead propose a simple approach which we call *Plateau Dropping*. It is very loosely inspired by GRPO-based methods, which do not learn from saturated (all-pass) or impossible (all-fail) tasks.
During SFT, we track the loss for every token, and drop it from the loss if it has not changed enough in the last few times it was seen. This effectively allows the model to focus on the current "learnable band" of tokens at every step.
If a dropped token's loss moves again in the future, it gets re-added to the loss.
We find that this simple approach significantly improves coverage (as measured by pass@8) for long-horizon agentic tasks without a drop in pass@1, as tested on SWE-bench Verified and OpenThoughts-TBLite for Marin's 67B-A2B model. It also leads to much "softer" models than vanilla SFT, as measured by next-token entropy.

</div>
<div class="post-toc" markdown="1">

<p class="post-toc-title">Contents</p>

- [1. Introduction](#1-introduction)
- [2. Methodology](#2-methodology)
- [3. Experimental Setup](#3-experimental-setup)
- [4. Results](#4-results)
  - [4.1 SWE](#41-swe)
  - [4.2 Terminal](#42-terminal)
  - [4.3 Baselines: TailSFT](#43-baselines-tailsft)
  - [4.4 Learnable tasks](#44-learnable-tasks)
  - [4.5 What is Plateau Dropping helping with?](#45-what-is-plateau-dropping-helping-with)
- [5. Next steps](#5-next-steps)
- [Acknowledgements](#acknowledgements)
- [Appendix A: training/infra footguns](#appendix-a-traininginfra-footguns)
- [Appendix B: dataset sizes in agentic SFT](#appendix-b-dataset-sizes-in-agentic-sft)
- [Appendix C: annotated trajectories](#appendix-c-annotated-trajectories)
- [Appendix D: TailSFT coverage ratio](#appendix-d-tailsft-coverage-ratio)
- [Appendix E: memory overhead of Plateau Dropping](#appendix-e-memory-overhead-of-plateau-dropping)
- [References](#references)

</div>
</div>

---

## 1. Introduction

[Marin](https://marin.community/) is a project aimed at building performant open models (open weights, open data, open recipes, etc.).
They are currently pretraining a 535B-A23B MoE model (view the progress [here](https://mtracker.oa.dev/hero-run-535b)!). Prior to this, they finished pretraining a smaller 67B-A2B model (codenamed "Snowball"). 
This model then went through an SFT stage focused on general capabilities.
Our goal was to spend a few weeks turning this checkpoint into an "agentic expert". We were specifically interested in improving its performance on long-horizon SWE and terminal tasks.
We started out with RL but could not improve the model's performance by more than a few points (within noise for our benchmarks). Our hypothesis was that the base model itself was not performant enough at agentic tasks, and could greatly benefit from an SFT warmup stage.
This pipeline (general SFT → domain-specialized SFT → RL) is not specific to our models: many frontier post-training recipes train domain-specialized experts with SFT and RL and then combine them ([GLM-4.5](https://arxiv.org/abs/2508.06471), [DeepSeek-V3.2](https://arxiv.org/abs/2512.02556), [MiMo-V2-Flash](https://arxiv.org/abs/2601.02780)). The resulting expert can also serve as a teacher in multi-teacher on-policy distillation (MOPD) setups ([MiMo-V2-Flash](https://arxiv.org/abs/2601.02780)).
Since we eventually want to run RL on this warmed-up model (likely with some GRPO variant), we were more interested in preserving and improving the model *coverage* during this specialized SFT phase, rather than solely benchmaxxing pass@1. Indeed, a recent work ([TailSFT](https://arxiv.org/abs/2608.25756)) shows that starting GRPO from models with better initial coverage leads to better post-RL checkpoints (measured by pass@1), even if the initial pass@1 is not optimal.
TailSFT argues that vanilla SFT keeps raising the likelihood of already "fit" / "learned" sequences, and this in turn hurts coverage. They address this by dropping, from each batch, the sequences whose loss has fallen the most relative to the base model. On OLMo-3 7B, they improve pass@16 in 15 of 18 math and code settings. 
However, when we try this method on our long-horizon agentic workload, we do not observe any benefits (Section 4.3).

Inspired by the ideas presented in TailSFT, and by the fact that GRPO doesn't learn anything from all pass / all fail groups, we propose a simple SFT approach called *Plateau Dropping*, where we track a token's loss history during SFT and drop it if its loss hasn't changed enough in the last few times it was seen. If its loss later changes substantially again, it is trained on again. The key intuition is to stop spending updates on tokens whose loss has stopped changing: either the model has already learned them, or further updates on them are not helping. In practice, most of the tokens this rule drops are "learned" ones, which already have a very low loss. Dropping only the small set of "stuck" tokens, whose loss has stopped changing at a high value, recovers all of the pass@1 gain but only about half of the pass@8 gain; the rest of the coverage gain appears only when both kinds are dropped together (Section 4.5). We find that this simple rule works pretty well in agentic settings, yielding a +4.5pp gain at pass@8 on SWE-bench Verified, and smaller gains on terminal benchmarks, with no significant drop in pass@1.

Plateau Dropping is closest to methods that select training data by how it behaves. 
Rho-1 ([Lin et al., 2024](https://arxiv.org/abs/2404.07965)) targets (continual) pre-training and trains only on tokens whose loss is high relative to a reference model. 
In contrast, Plateau Dropping needs no reference model, and its selection criterion follows the current model's own loss history. 
Methods like TailSFT, Selective Backprop ([Jiang et al., 2019](https://arxiv.org/abs/1910.00762)), and loss truncation ([Kang and Hashimoto, 2020](https://arxiv.org/abs/2004.14589)) drop or prioritise whole examples, while Plateau Dropping selects individual tokens instead. 
Focal loss ([Lin et al., 2017](https://arxiv.org/abs/1708.02002)) down-weights well-classified examples, based on a snapshot of the model's confidence; Plateau Dropping instead looks at how a token's loss changes over time.

## 2. Methodology

We use $$\mathcal{D} = \{y_n\}_{n=1}^N$$ to denote our dataset of agentic trajectories.
A trajectory $$y \in \mathcal{D}$$ is a sequence of tokens $$(y_i)_{i=1}^L$$.
We use the standard NLL loss for supervised fine-tuning: $$\ell_i(\theta) = -\log p_\theta (y_i \mid y_{<i})$$ for token $$i$$ of a single trajectory $$y$$.
Since these trajectories are agentic in nature, they also contain tool outputs. We zero out the loss on these tokens, as well as on the prompt.
Thus, the final SFT loss we use is: 

$$
\mathcal{L}_{\mathrm{SFT}}(\theta) = \frac{\mathbb{E}_{y \sim \mathcal{D}} \big[ \sum_{i} w_i \ell_i(\theta) \big]}{\mathbb{E}_{y \sim \mathcal{D}} \big[ \sum_{i} w_i \big]},
$$

where $$w_i=1$$ only for the agent's tokens, and $$=0$$ for the prompt and the tool responses.

**Detecting plateaus.** Let $$\ell_i^{(j)}$$ denote token $$i$$'s loss at its $$j^{\mathrm{th}}$$ visit.
For $$j \ge 2$$, let $$\delta_i(j)$$ be the relative change in this loss since the previous visit:

$$
\delta_i(j) = \frac{|\ell_i^{(j)} - \ell_i^{(j - 1)}|}{\max(\ell_i^{(j - 1)}, \varepsilon)}.
$$

We say token $$i$$ has *plateaued* at visit $$j \ge 3$$ if $$\delta_i(j) < \tau \text{ and } \delta_i(j - 1) < \tau$$.
This can be written as $$\alpha_i(j) = \mathbf{1}\{\delta_i(j) < \tau \,\wedge\, \delta_i(j - 1) < \tau\}$$.
This definition can be extended to consider the past $$k$$ visits as well, but we only consider $$k=2$$ here.
We drop the plateaued tokens from the SFT loss:

$$
\mathcal{L}_{\mathrm{PD}}(\theta) = \frac{\mathbb{E}_{y \sim \mathcal{D}} \big[ \sum_{i} w_i \kappa_i \ell_i(\theta) \big]}{\mathbb{E}_{y \sim \mathcal{D}} \big[ \sum_{i} w_i \kappa_i \big]},
$$

where $$\kappa_i = 1 - \alpha_i(v_i)$$ and $$v_i$$ is token $$i$$'s current visit.

In our experiments, we use $$\tau = 0.1$$ and $$\varepsilon = 0.05$$. 
$$\mathcal{L}_{\mathrm{PD}}$$ reduces to $$\mathcal{L}_{\mathrm{SFT}}$$ for $$\tau=0$$. 
Note that the history is always updated for all tokens, even the ones that get dropped, so that if a dropped token's loss moves again, it is trained again.

Figure 1 shows how the fraction of dropped tokens changes as the training proceeds.

<div class="band-chart" role="img" aria-label="Figure 1: share of trained tokens dropped as learned, dropped as stuck and still learning over training steps">
<p class="band-legend"><span class="band-l"></span>dropped as learned <span class="band-s"></span>dropped as stuck <span class="band-u"></span>still learning</p>
<div class="band-plot">
<div class="band-col"><div class="band-bar"><div class="band-u" style="height:100.0%" title="still learning: 100.0%">no drops yet</div></div><div class="band-x">103</div></div>
<div class="band-col"><div class="band-bar"><div class="band-u" style="height:72.9%" title="still learning: 72.9%">73%</div><div class="band-s" style="height:1.5%" title="dropped as stuck: 1.5%"></div><div class="band-l" style="height:25.5%" title="dropped as learned: 25.5%">26%</div></div><div class="band-x">206</div></div>
<div class="band-col"><div class="band-bar"><div class="band-u" style="height:50.4%" title="still learning: 50.4%">50%</div><div class="band-s" style="height:3.0%" title="dropped as stuck: 3.0%"></div><div class="band-l" style="height:46.6%" title="dropped as learned: 46.6%">47%</div></div><div class="band-x">309</div></div>
<div class="band-col"><div class="band-bar"><div class="band-u" style="height:47.5%" title="still learning: 47.5%">48%</div><div class="band-s" style="height:3.3%" title="dropped as stuck: 3.3%"></div><div class="band-l" style="height:49.2%" title="dropped as learned: 49.2%">49%</div></div><div class="band-x">412</div></div>
<div class="band-col"><div class="band-bar"><div class="band-u" style="height:44.6%" title="still learning: 44.6%">45%</div><div class="band-s" style="height:3.6%" title="dropped as stuck: 3.6%"></div><div class="band-l" style="height:51.9%" title="dropped as learned: 51.9%">52%</div></div><div class="band-x">515</div></div>
<div class="band-col"><div class="band-bar"><div class="band-u" style="height:41.8%" title="still learning: 41.8%">42%</div><div class="band-s" style="height:4.2%" title="dropped as stuck: 4.2%"></div><div class="band-l" style="height:53.9%" title="dropped as learned: 53.9%">54%</div></div><div class="band-x">618</div></div>
<div class="band-col"><div class="band-bar"><div class="band-u" style="height:39.6%" title="still learning: 39.6%">40%</div><div class="band-s" style="height:5.3%" title="dropped as stuck: 5.3%"></div><div class="band-l" style="height:55.1%" title="dropped as learned: 55.1%">55%</div></div><div class="band-x">721</div></div>
<div class="band-col band-last"><div class="band-bar"><div class="band-u" style="height:38.4%" title="still learning: 38.4%">38%</div><div class="band-s" style="height:5.4%" title="dropped as stuck: 5.4%"></div><div class="band-l" style="height:56.2%" title="dropped as learned: 56.2%">56%</div></div><div class="band-x">760</div></div>
</div>
<p class="band-xlabel">training step (the run ends at step 760)</p>
</div>

Figure 1. Plateau Dropping, 10 epochs on the SWE data: the share of trained tokens dropped as learned, dropped as stuck, and still being learned at the training steps shown. Dropped as learned: the loss has stopped moving, below −log 0.9; dropped as stuck: it stopped moving at a higher loss. Bars: the logged per-batch shares, averaged over the 5 steps ending at each step shown. Figure 2 (Appendix C) colours every token of three randomly chosen assistant turns by its final state.
{: .caption}

## 3. Experimental Setup

- **Model.** We use Marin's 67B MoE model with about 2B parameters active per token: [open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21](https://huggingface.co/open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21). This is an earlier version of the Snowball checkpoint released recently. Note that this model has already gone through one general SFT phase after pre-training.
- **Terminal data.** 6,301 total successful trajectories (45M tokens, 1,610 tasks): 5,324 from the public [allenai/tmax-sft](https://huggingface.co/datasets/allenai/tmax-sft) dataset and 977 of the base model's own successes on synthetic terminal tasks from [Endless Terminals](https://arxiv.org/abs/2601.16443) and [Recursive Synthesis of Tasks](https://arxiv.org/abs/2608.05466) (622 and 355 trajectories, from 331 and 219 tasks). Context 32k.
- **SWE data.** 9,229 successful trajectories (7,005 tasks) from the public [nvidia/Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) dataset, restricted to one shell command per step to match our agent (319M tokens). Context 64k.
- **Training.** The learning rates are given in Section 4. We use a cosine schedule over the run that decays to 10% of the peak learning rate, with 3% linear warmup, [AdamH](https://github.com/marin-community/marin/blob/main/lib/levanter/src/levanter/optim/adamh.py), no weight decay, bf16 compute with fp32 master weights and optimizer states, expert capacity factor 2.0 (no dropped token-to-expert assignments), router biases frozen. Vanilla SFT and Plateau Dropping runs of the same length share the data order. We train with [Marin](https://github.com/marin-community/marin) and its JAX training library Levanter.
- **Harness.** We use the [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) harness everywhere: all training trajectories are in its format, and all evaluations run in it.

## 4. Results

We focus only on agentic capabilities in this work, evaluating SWE and terminal tasks.
For SWE, we evaluate on a fixed random subset of 100 tasks from SWE-bench Verified.
For terminal tasks, we use 88 of the 89 tasks in Terminal-Bench 2.0 and 97 of the 100 tasks in [OpenThoughts-TBLite](https://huggingface.co/datasets/open-thoughts/OpenThoughts-TBLite); infrastructure errors kept us from building the sandboxes for the remaining tasks.
Terminal tasks are run with 3x the default time limits, and when an agent hits the time limit, its final state is scored.
Sandbox and API errors are counted as failures.
We report pass@1 and pass@8 for all models. We sample at temperature 1 with no top-p / top-k / penalties.

Note: the SWE-bench Verified tasks served by Harbor's registry (`swebench-verified@1.0`) run `git clean -fd` before the tests, which deletes every file the agent created that is not tracked by git, so a correct fix that adds a new file is marked as a failure. We instead use task scripts that delete only the untracked files touched by the task's test patch, which is what Harbor's SWE-bench adapter now does as well ([PR #1094](https://github.com/harbor-framework/harbor/pull/1094)).

### 4.1 SWE

We use a learning rate of 3e-4 for SFT and 1e-4 for PD. We find these to be optimal for both settings (searched over 1e-4, 3e-4, 5e-4).
We run both SFT and PD for 3 and 10 epochs and find that 10 epochs works much better for both.
Table 1 thus compares pass@1 and pass@8 for SFT and PD at 10 epochs, each with its best learning rate.
All results are averaged over 2 training runs with different data orders.
PD is level with SFT in pass@1 and outperforms it by 4.5 points in pass@8.

Note that it is uncommon in prior work to train for as long as 10 epochs (agentic SFT recipes train for 2 to 7: 2 in [R2E-Gym](https://arxiv.org/abs/2504.07164), [Nemotron-Terminal](https://arxiv.org/abs/2602.21193) and [TMax](https://arxiv.org/abs/2606.23321), up to 3 in [SWE-smith](https://arxiv.org/abs/2504.21798), up to 5 in [SWE-Gym](https://arxiv.org/abs/2412.21139) and 7 in [OpenThoughts-Agent](https://huggingface.co/open-thoughts/OpenThinker-Agent-v1-SFT)), but we find that it can work best, especially when using SFT to specialize in a domain (here, agentic tasks) with limited data.

**Table 1.** SWE-bench Verified (random 100 tasks), 10 epochs (760 steps), pass@1 and pass@8 in %, 8 seeds, averaged over 2 training runs with different data orders.

| model | pass@1 | pass@8 |
|---|---|---|
| base model | 19.3 | 46.0 |
| + SFT | **49.9** | 73.0 |
| + Plateau Dropping | 48.8 | **77.5** |

### 4.2 Terminal

We use a learning rate of 1e-4 for both SFT and PD. For terminal data, unlike SWE data, this outperforms both 3e-4 and 5e-5.
We run both SFT and PD for 3 and 10 epochs and find that 10 epochs works much better for both.
Table 2 thus compares pass@1 and pass@8 for SFT and PD at 10 epochs, with a learning rate of 1e-4.
We observe that the gains obtained by PD over vanilla SFT are much more modest for terminal tasks.
PD outperforms SFT on TB-Lite by 2.2 points in pass@1 and 3.1 points in pass@8. On TB2, PD gives no consistent gains over SFT in either metric.


**Table 2.** TB-Lite (all 97 tasks) and Terminal-Bench 2.0 (all 88 tasks), 10 epochs (220 steps), pass@1 and pass@8 in %, 8 seeds.

<table>
<thead>
<tr><th rowspan="2">model</th><th colspan="2" class="th-group">TB-Lite</th><th colspan="2" class="th-group">Terminal-Bench 2.0</th></tr>
<tr><th>pass@1</th><th>pass@8</th><th>pass@1</th><th>pass@8</th></tr>
</thead>
<tbody>
<tr><td>base model</td><td>16.6</td><td>50.5</td><td>7.1</td><td>28.4</td></tr>
<tr><td>+ SFT</td><td>33.9</td><td>64.9</td><td>16.8</td><td><strong>39.8</strong></td></tr>
<tr><td>+ Plateau Dropping</td><td><strong>36.1</strong></td><td><strong>68.0</strong></td><td><strong>17.5</strong></td><td>38.6</td></tr>
</tbody>
</table>


### 4.3 Baselines: TailSFT

We try out TailSFT and its token-level variant on SWE data and find that neither beats vanilla SFT at pass@8 in any setting.
We train all methods on the same data for 3 epochs with a learning rate of 1e-4. We try out TailSFT with 50% of the trajectories dropped (ramped over the run) and 25% dropped (fixed). These are the settings recommended by the authors for math and code data respectively. We also try a token-level variant of TailSFT which drops tokens instead of entire trajectories. Table 3 shows the results: every variant we tried lags behind vanilla SFT at pass@8.

TailSFT proposes a diagnostic called the coverage ratio (Definition 4.1 in their paper) to predict whether it will help in a given setting: the coverage that vanilla SFT loses relative to the base model divided by the coverage it gains.
For vanilla SFT after 3 epochs, this ratio is above 1 (3.1 at k=8), the regime in which the paper reports gains in 10 of 11 settings, but no TailSFT variant we tried is ahead of vanilla SFT at pass@8.
[Appendix D](#appendix-d-tailsft-coverage-ratio) gives the details of the calculation.

**Table 3.** SWE-bench Verified (random 100 tasks), 3 epochs (228 steps), learning rate 1e-4, pass@1 and pass@8 in %, 8 seeds.

| model | pass@1 | pass@8 |
|---|---|---|
| base model | 19.3 | 46.0 |
| + SFT | 23.4 | 58.0 |
| + TailSFT, 50% dropped (ramped) | 23.4 | 51.0 |
| + TailSFT, 25% dropped (fixed) | 22.5 | 51.0 |
| + TailSFT, token-level | 18.0 | 46.0 |
| + Plateau Dropping | **31.9** | **67.0** |

### 4.4 Learnable tasks

**Table 4.** Number of tasks never solved / solved by some of the 8 seeds / always solved, at 10 epochs, with the models of Tables 1 and 2 (for SWE-bench Verified, the first of the two data orders).

| benchmark | base model | + SFT | + Plateau Dropping |
|---|---|---|---|
| SWE-bench Verified (100 tasks) | 54 / 42 / 4 | 26 / 48 / 26 | 22 / 57 / 21 |
| TB-Lite (97 tasks) | 48 / 48 / 1 | 34 / 60 / 3 | 31 / 63 / 3 |
| Terminal-Bench 2.0 (88 tasks) | 63 / 25 / 0 | 53 / 33 / 2 | 54 / 33 / 1 |

Group-based RL methods like GRPO learn only from tasks the model solves some of the time. 
On SWE-bench Verified, vanilla SFT solves more tasks *reliably* (26 vs 21 always solved), which gives it its higher pass@1, while Plateau Dropping keeps more tasks *within reach* (57 vs 48 in the "learnable band"); on the terminal benchmarks however, the two are about the same.
Note that even vanilla SFT *gains* overall coverage (pass@8) over the base model in every setting we tried. 
The "sharpening" effect is relative here. 
We believe this is expected with relatively weak base models like the one we have used in this study.
We are continuing to test more model families and will update results soon!

### 4.5 What is Plateau Dropping helping with?

Plateau Dropping drops two kinds of tokens: "learned" ones, which make up most of what it drops, and a small set of "stuck" ones.
In our convention, learned tokens are those whose loss plateaus at a value $$< -\log 0.9$$, while "stuck" tokens are all the other plateaued tokens, whose loss is still higher.
To see which kind matters more, we drop only one of them, at the same learning rate (Table 5). 
We find that dropping only the learned tokens does about as well as vanilla SFT, while dropping only the stuck tokens matches Plateau Dropping at pass@1 and recovers about half of its pass@8 gain over vanilla SFT.

**Table 5.** SWE-bench Verified (random 100 tasks), 10 epochs (760 steps), learning rate 1e-4, pass@1 and pass@8 in %, 8 seeds; one training run each, and the two "only" runs use a different data order from the other two; "Dropped" is the share of trained tokens dropped in each batch at the end of training.

| model | dropped | pass@1 | pass@8 |
|---|---|---|---|
| + SFT | 0% | 42.3 | 70.0 |
| + Plateau Dropping, learned tokens only | 56% | 40.8 | 69.0 |
| + Plateau Dropping, stuck tokens only | 5% | **48.9** | 74.0 |
| + Plateau Dropping | 62% | 48.8 | **78.0** |

To see how much each method sharpens the model, we measure the entropy of its next-token distribution (Table 6) at three sets of positions:
- **stuck tokens:** tokens of the SWE training data that Plateau Dropping was dropping as stuck at the end of training, i.e. whose loss had stopped changing while still high;
- **other training tokens:** for each stuck token, a token from the same training trajectory where possible, with the same next token, in the same kind of span (reasoning or action), that Plateau Dropping was not dropping at the end of training;
- **uncertain rollout tokens:** reasoning tokens from successful SWE-bench Verified rollouts of the Table 1 models where the base model's most likely next token has probability below 0.5; every model is scored on the same positions.

At its best learning rate (3e-4), vanilla SFT cuts the base model's entropy by 58-73% across these sets, while Plateau Dropping cuts it by 29-41%, at the same pass@1 and a higher pass@8.
Being less "sharp" is not enough on its own, though: vanilla SFT at learning rate 1e-4 ends up slightly "softer" than Plateau Dropping, but scores 6.6 points lower at pass@1 and 7.5 points lower at pass@8.

**Table 6.** Mean next-token entropy (nats, over the top 100 tokens) after 10 epochs on the SWE data, and SWE-bench Verified pass@1 and pass@8 in %. Vanilla SFT at 3e-4 and Plateau Dropping are averaged over the 2 data orders; vanilla SFT at 1e-4 is a single run.

| model | other training tokens | stuck tokens | uncertain rollout tokens | pass@1 | pass@8 |
|---|---|---|---|---|---|
| base model | 1.24 | 1.48 | 1.85 | 19.3 | 46.0 |
| + SFT, learning rate 3e-4 | 0.33 | 0.63 | 0.72 | **49.9** | 73.0 |
| + SFT, learning rate 1e-4 | 0.83 | 1.17 | 1.31 | 42.3 | 70.0 |
| + Plateau Dropping, learning rate 1e-4 | 0.73 | 1.05 | 1.19 | 48.8 | **77.5** |


**An example.** On `django__django-11477`, the base model and vanilla SFT never succeed (0 of 8 and 0 of 16), and Plateau Dropping succeeds twice in 16 attempts (across 2 data orders). 
In every attempt of both models, the reasoning ties the failure to `None` values in the resolved kwargs (11 of 16 Plateau Dropping and 10 of 16 vanilla SFT attempts name `match.groupdict()` explicitly).
They differ in where they fix it.
Every vanilla SFT attempt that edits the source (9 of 16) patches `translate_url` in `django/urls/base.py`, after setting the alternative aside:

> Let me also check if maybe the fix should be more general — e.g., in `resolve` / `ResolverMatch` — but no, the issue is clearly in `translate_url`'s use of `reverse`.

The hidden tests also check `resolve()` itself, so this fix fails.
Plateau Dropping takes the same route in 7 of its attempts, which fail too, but its two successes fix the resolver:

> Actually, let me reconsider one more time. I keep going back and forth, so let me make a firm decision based on the strongest reasoning: [...] Root cause: `match.groupdict()` includes `None` for non-participating groups. This is a resolver-level concern. Fix: Exclude `None` group values from `kwargs` in the resolver's `match()` methods.

Note: both successes come from one of the two data orders, so this is an illustration rather than a strong statistical result.

Our reading of these measurements is that Plateau Dropping reaches vanilla SFT's accuracy at a lower learning rate, and that this checkpoint is substantially less concentrated than vanilla SFT at its tuned learning rate.
That leaves alternative reasoning paths reachable when sampling, and some of them produce additional solves as illustrated above.

## 5. Next steps

**More model families.** This work was done in collaboration with folks at [Marin](https://marin.community/), with the aim of creating an agentic expert (a domain-specialized model that can also later be used in MOPD setups). This is why all experiments in this work have been performed using Marin's 67B-A2B MoE model. We're currently running more experiments with other popular open-source models to test the generalizability of this approach. 

**RL.** The entire premise of this work has been that starting RL from a model that has more coverage can lead to better post-RL performance. TailSFT indeed demonstrates this for non-agentic tasks. We are now starting RL experiments that compare the post-RL performance of vanilla SFT and Plateau Dropping. 

**Data scale.** The dataset sizes we use for SFT in this work are pretty small by frontier standards and by non-agentic SFT standards (although they are typical for academic agentic SFT; see [Appendix B](#appendix-b-dataset-sizes-in-agentic-sft)). In particular, our terminal data has ~7x fewer tokens than our SWE SFT data. We're currently in the process of scaling the number of trajectories and tasks to see how well our approach works at larger token budgets.

**Evaluation.** We limit our agentic evals to SWE and terminal tasks. However, there are other practically relevant "agentic" domains like web search ([BrowseComp](https://arxiv.org/abs/2504.12516)), tool use ([τ-bench](https://arxiv.org/abs/2406.12045)) and computer use ([OSWorld](https://arxiv.org/abs/2404.07972)), which we are yet to evaluate on.

## Acknowledgements

We thank [Bespoke Labs](https://bespokelabs.ai/) for supporting my coding agent plan, Alex Dimakis for his advice, and the [Jülich Supercomputing Centre (JSC)](https://www.fz-juelich.de/en/jsc) and the [Texas Advanced Computing Center (TACC)](https://tacc.utexas.edu/) for providing compute for this work.

## Appendix A: training/infra footguns

We ran into several issues while training.

**1. The master weights were bf16.** 
The script that imported the base checkpoint in Levanter wrote the weights in bf16, and the trainer kept whatever precision it loaded, so the optimizer was initially applying its fp32 updates to bf16 weights and rounded the result back to bf16 at every step! 
A bf16 number has an 8-bit mantissa, so any update smaller than about 0.4% of a weight's magnitude rounds to zero, which is most Adam updates at a learning rate of 1e-4. 
Even after 1520 steps, 92-95% of the weights were *bit-identical* to the base model because of this; and the training loss stopped at 0.55. 
The fix was to just keep fp32 master weights (the standard mixed-precision recipe: bf16 compute, fp32 weights and optimizer state).
In the fixed runs 65% of the weights changed after just 228 steps, 83% after 760 and 89% after 1520. 
Every one of our initial experiments was trained this way: vanilla SFT, Plateau Dropping, TailSFT and its variants, the ablations, etc.

**2. Dropping token-to-expert assignments.** 
We were initially training the model with an expert capacity factor of 1.0, so whenever a batch routed more tokens to an expert than its even share, the excess assignments were just dropped.
On this data 14-19% of assignments were dropped, which cost 0.067 to 0.082 nats of loss.
With the capacity factor raised to 2.0 the overflow is 0.16% at the start of a run and 0 by the end, and the trainer's loss matches an independent inference implementation to 0.0002.

**3. Epochs are overstated.** We first counted passes from token totals and ignored row padding, so what we called "10 epochs" on the terminal data are actually 8.6 "real" passes and "10 epochs" on the SWE data are 7.4. 
This post keeps the nominal names: our packer fills fixed-length rows (64k tokens on the SWE data, 32k on the terminal data) with whole trajectories, which leaves about 26% of each SWE row and 16% of each terminal row as padding, and the data loader shuffles the packed rows. One real pass over the SWE data is 103 steps, so 228 and 760 steps ("3 and 10 epochs") are 2.2 and 7.4 passes; one pass over the terminal data is about 26 steps, so 66 and 220 steps are 2.6 and 8.6 passes. The plateau test counts visits, not epochs, so is unaffected by this.

**4. Gradient clipping was silently penalising Plateau Dropping.** The trainer clips the global gradient norm at 1.0 (the usual setting) and the loss Plateau Dropping optimises is a mean over the kept tokens only.
Once it has dropped half the tokens its gradient norm is 2-3x vanilla SFT's, so the clip, which touches 3-5% of vanilla SFT's steps, hit 50% of the steps of the 10-epoch Plateau Dropping run on the SWE data and 82-100% of the steps of the terminal runs. 
Nothing in the training logs flagged it: the loss curves of the clipped and unclipped runs are indistinguishable, and only the evaluations differ, by 2-6 points at 10 epochs. 
The TailSFT paper also clips at 1.0 with the loss averaged over the tokens of the kept sequences ([Malladi et al., 2026](https://arxiv.org/abs/2608.25756), Appendix D.1), but in our sequence-level TailSFT runs the clip engaged no more often than for vanilla SFT (see below), so we do not expect its results to be affected in the same way.
At 10 epochs on the SWE data the two settings compare as follows (8 seeds, all 100 tasks; the loss curves of the two Plateau Dropping runs are indistinguishable, 0.347 against 0.346 over all tokens at the end):

| 10 epochs (760 steps) | pass@1 | pass@8 |
|---|---|---|
| Plateau Dropping, clip 1.0 | 46.6 | 72.0 |
| Plateau Dropping, clip 4.0 | **48.8** | **78.0** |

Clip 4.0 beats clip 1.0 by 2.1 points at pass@1 and 6.0 points at pass@8, paired over tasks. 
The TailSFT runs were clipped no more than vanilla SFT was and stand as they are; the drop-share 0.25 and per-token TailSFT runs were trained at clip 4.0. 
The 10-epoch Plateau Dropping runs were redone at clip 4.0, and only those appear in Tables 1 and 2; the 3-epoch Plateau Dropping run in Table 3 never reached the clip at 1.0.

## Appendix B: dataset sizes in agentic SFT

We compare our dataset sizes to a few publicly available datasets.
In trajectory count, they are close to the sets several SWE agents were fine-tuned on (SWE-smith, R2E-Gym, SWE-Dev) and to the terminal set we draw from, and one to two orders of magnitude below the largest public releases (Open-SWE-Traces, Nemotron-Terminal-Corpus).

**Table 7.** Size of public agentic SFT datasets, as stated in each paper or dataset card ("–": not reported).

| dataset | domain | tasks | trajectories | tokens |
|---|---|---|---|---|
| [SWE-Gym](https://arxiv.org/abs/2412.21139) | SWE | 2,438 | 491 used for SFT | – |
| [R2E-Gym](https://arxiv.org/abs/2504.07164) | SWE | 2,048 environments | 3,321 | – |
| [SWE-smith](https://arxiv.org/abs/2504.21798) | SWE | 50k instances (128 repos) | 5,016 used for SFT | – |
| [SWE-Dev](https://arxiv.org/abs/2506.07636) | SWE | 38k instances | 574 to 16,639 across runs | – |
| [SWE-rebench OpenHands trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) | SWE | 3,792 issues resolved | 67,074 (32,161 successful) | – |
| [nvidia/Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces) v1.2 | SWE | – | 235,396 | – |
| [allenai/tmax-sft](https://huggingface.co/datasets/allenai/tmax-sft) | terminal | about 2.2k environments | 10,726 (5,795 successful) | – |
| [OpenThoughts-Agent-v1-SFT](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-SFT) | terminal | – | 15,209 | – |
| [Nemotron-Terminal-Corpus](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus) | terminal | – | 366,154 | – |
| **ours, terminal data** | terminal | 1,610 | 6,301 | 45M (22.5M trained) |
| **ours, SWE data** | SWE | 7,005 | 9,229 | 319M |

## Appendix C: annotated trajectories

We annotate some example trajectories using the still learning / dropped as learned / dropped as stuck categories of Figure 1.

{% raw %}
<div class="tok-legend"><span class="tok-l">dropped: learned</span><span class="tok-s">dropped: stuck</span><span class="tok-u">still learning</span><span class="tok-c">grey: context (tool output), not trained</span></div>
<p class="tok-excerpt-title">Excerpt 1: 40% dropped: learned,  5% dropped: stuck,  55% still learning</p>
<pre class="tok-excerpt"><span class="tok-c">&lt;hdr&gt;tool&lt;/hdr&gt;
&lt;tool_response name=&quot;bash&quot;&gt;(no output)

(exit_code=0)&lt;/tool_response&gt;&lt;eot&gt;
&lt;hdr&gt;assistant&lt;/hdr&gt;
</span><span class="tok-l">&lt;think&gt;</span><span class="tok-u">Now I</span><span class="tok-s"> can</span><span class="tok-u"> see the `registerKeep</span><span class="tok-l">AliveHook` function</span><span class="tok-u">. Let</span><span class="tok-l"> me</span><span class="tok-u"> understand the</span><span class="tok-s"> issue</span><span class="tok-u">.

In the `register</span><span class="tok-l">KeepAliveHook</span><span class="tok-u">`</span><span class="tok-l"> function</span><span class="tok-s">,</span><span class="tok-u"> when</span><span class="tok-s"> a</span><span class="tok-u"> hook is registered with `onActivated` or</span><span class="tok-l"> `onDeactivated</span><span class="tok-u">`, it</span><span class="tok-s"> creates</span><span class="tok-u"> a `wrappedHook`</span><span class="tok-s"> (</span><span class="tok-u">with `__wd</span><span class="tok-l">c`</span><span class="tok-u"> = with deactivation</span><span class="tok-l"> check</span><span class="tok-u">) that:
</span><span class="tok-l">1.</span><span class="tok-u"> Checks if the target instance is in a deactivated branch
</span><span class="tok-l">2.</span><span class="tok-u"> If not, calls `hook()` directly

</span><span class="tok-s">Then</span><span class="tok-u"> it calls `inject</span><span class="tok-l">Hook</span><span class="tok-u">(type</span><span class="tok-l">,</span><span class="tok-u"> wrappedHook, target</span><span class="tok-s">)`</span><span class="tok-u">.

</span><span class="tok-s">The</span><span class="tok-u"> `inject</span><span class="tok-l">Hook`</span><span class="tok-u"> function wraps</span><span class="tok-l"> the</span><span class="tok-u"> hook with error</span><span class="tok-l"> handling</span><span class="tok-u"> (`__</span><span class="tok-l">weh</span><span class="tok-u">`</span><span class="tok-s">),</span><span class="tok-u"> so `call</span><span class="tok-l">WithAsyncErrorHandling</span><span class="tok-u">` is used. So</span><span class="tok-s"> the</span><span class="tok-u"> flow is:

1</span><span class="tok-l">.</span><span class="tok-u"> `inject</span><span class="tok-l">Hook</span><span class="tok-u">(type</span><span class="tok-l">,</span><span class="tok-u"> wrapped</span><span class="tok-l">Hook, target)`</span><span class="tok-s"> creates</span><span class="tok-u"> `wrapped</span><span class="tok-l">Hook</span><span class="tok-u">.__</span><span class="tok-l">weh`</span><span class="tok-u"> which calls</span><span class="tok-l"> `callWithAsyncErrorHandling</span><span class="tok-u">(w</span><span class="tok-l">rappedHook,</span><span class="tok-u"> target</span><span class="tok-l">, type, args</span><span class="tok-u">)</span><span class="tok-l">`.
2.</span><span class="tok-u"> When the hook</span><span class="tok-s"> fires</span><span class="tok-u">, `wrapped</span><span class="tok-l">Hook</span><span class="tok-u">.__</span><span class="tok-l">weh</span><span class="tok-u">` is</span><span class="tok-l"> called</span><span class="tok-s"> (</span><span class="tok-u">via the array of hooks in `target</span><span class="tok-l">[type]`</span><span class="tok-u">).
</span><span class="tok-l">3.</span><span class="tok-u"> `call</span><span class="tok-l">WithAsyncErrorHandling</span><span class="tok-s">(w</span><span class="tok-l">rappedHook</span><span class="tok-u">, ...</span><span class="tok-l">)`</span><span class="tok-u"> calls `wrapped</span><span class="tok-l">Hook</span><span class="tok-u">()`,</span><span class="tok-l"> which</span><span class="tok-u"> checks the de</span><span class="tok-l">activation</span><span class="tok-u"> branch, then</span><span class="tok-l"> calls</span><span class="tok-u"> the original</span><span class="tok-l"> `hook</span><span class="tok-u">()`.

Now, the issue is about async functions. Let me look</span><span class="tok-l"> at</span><span class="tok-s"> the</span><span class="tok-u"> `call</span><span class="tok-l">WithAsyncErrorHandling`</span><span class="tok-u"> function in</span><span class="tok-s"> `</span><span class="tok-l">errorHandling.ts</span><span class="tok-u">`.</span><span class="tok-l">&lt;/think&gt;</span><span class="tok-u">Now let</span><span class="tok-l"> me</span><span class="tok-u"> look</span><span class="tok-l"> at the</span><span class="tok-u"> error handling file which is central</span><span class="tok-l"> to</span><span class="tok-u"> this</span><span class="tok-l"> issue.&lt;tool_call&gt;
{&quot;name&quot;: &quot;bash&quot;, &quot;arguments&quot;: {&quot;command&quot;: &quot;cat /testbed/packages/runtime-core/src/errorHandling.ts&quot;}}
&lt;/tool_call&gt;&lt;eot&gt;</span></pre>
<p class="tok-excerpt-title">Excerpt 2: 69% dropped: learned,  3% dropped: stuck,  28% still learning</p>
<pre class="tok-excerpt"><span class="tok-c">JSON dict: &#x27;{&quot;a&quot;: true, &quot;b&quot;: null}&#x27;
JSON list: &#x27;[&quot;1&quot;, true, null]&#x27;
Plain str: &#x27;my param&#x27;
Int infer: 1234
Disabled: &#x27;{&quot;a&quot;: true}&#x27;


(exit_code=0)&lt;/tool_response&gt;&lt;eot&gt;
&lt;hdr&gt;assistant&lt;/hdr&gt;
</span><span class="tok-l">&lt;think&gt;</span><span class="tok-u">Confirmed. Currently</span><span class="tok-s">:
</span><span class="tok-l">-</span><span class="tok-u"> Python dict works:</span><span class="tok-l"> `{&#x27;a&#x27;: True, &#x27;b&#x27;: None}`
- JSON dict</span><span class="tok-u"> fails:</span><span class="tok-s"> returns</span><span class="tok-u"> `&#x27;</span><span class="tok-l">{&quot;a&quot;: true, &quot;b&quot;: null}&#x27;</span><span class="tok-u">`</span><span class="tok-l"> (</span><span class="tok-u">string</span><span class="tok-l">)
- JSON list fails: returns</span><span class="tok-u"> `&#x27;</span><span class="tok-l">[&quot;1&quot;, true, null]&#x27;` (string</span><span class="tok-u">)

</span><span class="tok-s">Alright</span><span class="tok-l">, let</span><span class="tok-u">&#x27;s implement</span><span class="tok-l"> the fix.</span><span class="tok-u"> Add</span><span class="tok-l"> `import json</span><span class="tok-u">` and</span><span class="tok-l"> modify `_infer_param_type</span><span class="tok-u">` to fall</span><span class="tok-l"> back to</span><span class="tok-u"> `</span><span class="tok-l">json.loads</span><span class="tok-s">`.

</span><span class="tok-u">First let&#x27;s check the</span><span class="tok-s"> import</span><span class="tok-u"> section.</span><span class="tok-l">&lt;/think&gt;</span><span class="tok-u">Confirmed the</span><span class="tok-l"> issue.</span><span class="tok-u"> Now</span><span class="tok-l"> let me</span><span class="tok-u"> implement</span><span class="tok-l"> the fix</span><span class="tok-u">. I&#x27;ll add the</span><span class="tok-l"> `json` import and</span><span class="tok-u"> modify `_</span><span class="tok-l">infer_param_type` to</span><span class="tok-u"> fall</span><span class="tok-l"> back to</span><span class="tok-u"> `</span><span class="tok-l">json.loads`</span><span class="tok-u"> when</span><span class="tok-l"> `literal_eval` fails.</span><span class="tok-u">&lt;tool_call&gt;</span><span class="tok-l">
{&quot;name&quot;: &quot;bash&quot;, &quot;arguments&quot;: {&quot;command&quot;: &quot;cd /testbed &amp;&amp;</span><span class="tok-u"> sed</span><span class="tok-l"> -</span><span class="tok-u">n</span><span class="tok-l"> &#x27;</span><span class="tok-u">19</span><span class="tok-l">,</span><span class="tok-u">26</span><span class="tok-l">p&#x27; toolium/utils/dataset.py</span><span class="tok-u">&quot;}}
</span><span class="tok-l">&lt;/tool_call&gt;&lt;eot&gt;</span></pre>
<p class="tok-excerpt-title">Excerpt 3: 42% dropped: learned,  7% dropped: stuck,  51% still learning</p>
<pre class="tok-excerpt"><span class="tok-c">&lt;hdr&gt;tool&lt;/hdr&gt;
&lt;tool_response name=&quot;bash&quot;&gt;(no output)

(exit_code=0)&lt;/tool_response&gt;&lt;eot&gt;
&lt;hdr&gt;assistant&lt;/hdr&gt;
</span><span class="tok-l">&lt;think&gt;</span><span class="tok-u">Interesting — the</span><span class="tok-s"> one</span><span class="tok-u">-argument form of</span><span class="tok-l"> `</span><span class="tok-u">where` in `ops</span><span class="tok-l">.py`</span><span class="tok-u"> already</span><span class="tok-s"> implements</span><span class="tok-u"> a non-zero-like behavior using `j</span><span class="tok-l">np</span><span class="tok-u">.where(condition</span><span class="tok-l">.array</span><span class="tok-u">, size=new</span><span class="tok-l">_axis</span><span class="tok-u">.size,</span><span class="tok-l"> fill_value=fill</span><span class="tok-u">_value</span><span class="tok-s">)</span><span class="tok-u">`. This is a</span><span class="tok-s"> great</span><span class="tok-u"> pattern to follow for `</span><span class="tok-l">nonzero</span><span class="tok-u">`.

Let&#x27;s take</span><span class="tok-l"> a</span><span class="tok-u"> look</span><span class="tok-l"> at</span><span class="tok-u"> the entire</span><span class="tok-l"> `</span><span class="tok-u">unique` function and other related parts to understand the</span><span class="tok-s"> pattern being</span><span class="tok-l"> used</span><span class="tok-u">.</span><span class="tok-s"> I</span><span class="tok-l">&#x27;ll</span><span class="tok-s"> check</span><span class="tok-u"> the</span><span class="tok-s"> el</span><span class="tok-l">ided</span><span class="tok-u"> middle portion of ops</span><span class="tok-l">.py.&lt;/think&gt;</span><span class="tok-u">The `where</span><span class="tok-l">`</span><span class="tok-u"> function already shows the pattern for</span><span class="tok-s"> `</span><span class="tok-u">j</span><span class="tok-l">np.where</span><span class="tok-u">(condition,</span><span class="tok-l"> size</span><span class="tok-u">=...,</span><span class="tok-l"> fill_value=...)</span><span class="tok-u">`.</span><span class="tok-l"> Let me</span><span class="tok-u"> see</span><span class="tok-l"> the</span><span class="tok-u"> el</span><span class="tok-l">ided</span><span class="tok-u"> middle portion</span><span class="tok-l"> of</span><span class="tok-u"> `ops</span><span class="tok-l">.py</span><span class="tok-u">` to fully</span><span class="tok-l"> understand</span><span class="tok-u"> the `unique</span><span class="tok-l">`</span><span class="tok-u"> function and</span><span class="tok-s"> its patterns</span><span class="tok-l">.&lt;tool_call&gt;
{&quot;name&quot;: &quot;bash&quot;, &quot;arguments&quot;: {&quot;command&quot;: &quot;cd /testbed &amp;&amp; sed -n &#x27;</span><span class="tok-u">140</span><span class="tok-l">,</span><span class="tok-u">330</span><span class="tok-l">p&#x27; src/haliax/ops.py&quot;}}
&lt;/tool_call&gt;&lt;eot&gt;</span></pre>
{% endraw %}

Figure 2. Three assistant turns from the SWE training data, every token coloured by its state at the end of the 10-epoch Plateau Dropping run: dropped as learned (green), dropped as stuck (amber), still learning (blue); grey text is untrained context (tool output). Turns were drawn at random (seed 0) among those with 120 to 320 trained tokens that contain both reasoning and a tool call; they are not hand-picked. Special tokens (&lt;think&gt;, &lt;tool_call&gt;, &lt;eot&gt;) are shown by name.
{: .caption}

## Appendix D: TailSFT coverage ratio

We compute the coverage ratio of Section 4.3 for vanilla SFT after 3 epochs.
Over the 21 tasks the base model solves on only 1-2 of its 8 seeds, vanilla SFT *loses* more pass@8 coverage than it gains.
Note that the ratio is computed only on these tasks, which the base model can already solve some of the time; overall, vanilla SFT still improves pass@8 by 12 points.
Its coverage ratio (at k=8) is therefore 3.1, and 6 of the 42 tasks that the base model solves on some but not all of its seeds fall to 0/8 after SFT.
With the paper's own protocol (k=16, estimated from pass@1, over the 11 tasks the base model solves on 1 of its 8 seeds), the ratio comes out to be 12.8.
The ratio falls below 1 only after longer training (0.06 for vanilla SFT at 10 epochs and learning rate 3e-4), a setting in which we did not run TailSFT.
One caveat: the paper uses k=16, while we have only 8 samples per task, so our k=16 ratio is extrapolated from pass@1, and with only 11 to 21 tasks in the set, both ratios are noisy, though both stay above 1.

## Appendix E: memory overhead of Plateau Dropping

Unlike vanilla SFT, Plateau Dropping needs to remember the loss history of every trained token.
For each token, we store the last two losses it was seen with (float32), its visit count (int16) and whether it was dropped at its last visit (bool), i.e. 11 bytes per token.
We key this history by (trajectory, position in the trajectory) rather than by position in the packed row, so it follows the trajectory wherever it ends up in a packed row.
For simplicity, we store it as a dense table with a slot for every position up to the maximum context length of every trajectory, so on our SWE data (9,229 trajectories, 64k context) the table has 605M slots and takes 6.65 GB, of which only 30% (the trained tokens) are ever written.
For comparison, the model and optimizer state (fp32 master weights and optimizer moments) take about 805 GB, so the history adds less than 1% to the checkpoint.

**Implementation.** We keep the table in host RAM as numpy arrays, outside JAX, with a full copy on every training process.
Before each step, every process looks up the history of the tokens in the batch and passes it to the jitted train step as extra inputs, alongside the batch itself.
Inside the step, the drop decision uses the per-token losses from the same forward pass that computes the training loss, so there is no extra forward pass.
After the step, the per-token losses are gathered back to the host and every process updates its copy of the table, for the dropped tokens as well.
The table is saved as a separate file next to the model checkpoint.
All of this costs about 1.3% in step time (9.20 s per step against 9.08 s for vanilla SFT).

**Scaling.** Our implementation is far from optimal: the table grows with the number of trajectories times the maximum context length, and is replicated on every process.
Storing only the trained positions would cut it to about 2 GB on our SWE data, storing the losses in fp16 and the count in uint8 would bring it down to 6 bytes per token, and sharding the table across processes would remove the replication.
None of these were necessary at our data scale, but they would be at larger token budgets.

## References

1. Malladi, S. et al. (2026). TailSFT: Filtered Fine-Tuning Improves Post-Training Performance. [arXiv:2608.25756](https://arxiv.org/abs/2608.25756).
2. Dang, X., Baek, C., Wen, K., Kolter, Z. and Raghunathan, A. (2025). Weight Ensembling Improves Reasoning in Language Models. [arXiv:2504.10478](https://arxiv.org/abs/2504.10478).
3. Chen, F., Raventos, A., Cheng, N., Ganguli, S. and Druckmann, S. (2025). Rethinking Fine-Tuning when Scaling Test-Time Compute: Limiting Confidence Improves Mathematical Reasoning. [arXiv:2502.07154](https://arxiv.org/abs/2502.07154).
4. GLM-4.5 Team (2025). GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models. [arXiv:2508.06471](https://arxiv.org/abs/2508.06471).
5. DeepSeek-AI (2025). DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models. [arXiv:2512.02556](https://arxiv.org/abs/2512.02556).
6. Xiaomi LLM-Core Team (2026). MiMo-V2-Flash Technical Report. [arXiv:2601.02780](https://arxiv.org/abs/2601.02780).
7. Lin, Z. et al. (2024). Rho-1: Not All Tokens Are What You Need. [arXiv:2404.07965](https://arxiv.org/abs/2404.07965).
8. Jiang, A. H. et al. (2019). Accelerating Deep Learning by Focusing on the Biggest Losers. [arXiv:1910.00762](https://arxiv.org/abs/1910.00762).
9. Kang, D. and Hashimoto, T. (2020). Improved Natural Language Generation via Loss Truncation. ACL 2020. [arXiv:2004.14589](https://arxiv.org/abs/2004.14589).
10. Lin, T.-Y. et al. (2017). Focal Loss for Dense Object Detection. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002).
11. Pan, J. et al. (2025). Training Software Engineering Agents and Verifiers with SWE-Gym. [arXiv:2412.21139](https://arxiv.org/abs/2412.21139).
12. Jain, N. et al. (2025). R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weights SWE Agents. [arXiv:2504.07164](https://arxiv.org/abs/2504.07164).
13. Yang, J. et al. (2025). SWE-smith: Scaling Data for Software Engineering Agents. [arXiv:2504.21798](https://arxiv.org/abs/2504.21798).
14. Wang, H. et al. (2025). SWE-Dev: Building Software Engineering Agents with Training and Inference Scaling. [arXiv:2506.07636](https://arxiv.org/abs/2506.07636).
15. Nebius (2025). SWE-rebench OpenHands trajectories. [Hugging Face](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories).
16. NVIDIA (2026). Nemotron-Terminal-Corpus. [Hugging Face](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus); [arXiv:2602.21193](https://arxiv.org/abs/2602.21193).
17. OpenThoughts (2025). OpenThoughts-Agent-v1-SFT. [Hugging Face](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-SFT).
18. Ivison, H. et al. (2026). Tmax: A Simple Recipe for Terminal Agents. [arXiv:2606.23321](https://arxiv.org/abs/2606.23321).
19. Gandhi, K., Garg, S., Goodman, N. and Papailiopoulos, D. (2026). Endless Terminals: Scaling RL Environments for Terminal Agents. [arXiv:2601.16443](https://arxiv.org/abs/2601.16443).
20. Li, Z. et al. (2026). Recursive Synthesis for Long-Horizon Terminal Tasks. [arXiv:2608.05466](https://arxiv.org/abs/2608.05466).
21. Wei, J. et al. (2025). BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents. [arXiv:2504.12516](https://arxiv.org/abs/2504.12516).
22. Yao, S. et al. (2024). τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. [arXiv:2406.12045](https://arxiv.org/abs/2406.12045).
23. Xie, T. et al. (2024). OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments. [arXiv:2404.07972](https://arxiv.org/abs/2404.07972).

## Citation

If you find this post useful, please cite it as:

```bibtex
@misc{varma2026plateaudropping,
  title  = {Plateau Dropping: a simple SFT rule for better agentic post-training},
  author = {Varma, Harshit and Lee, Donghyun and Dimakis, Alexandros G.},
  year   = {2026},
  month  = oct,
  url    = {https://hrshtv.github.io/blog/plateau-dropping/}
}
```
