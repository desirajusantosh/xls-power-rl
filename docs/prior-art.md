# Prior art

Searched 2026-10-27. Venues, years and summaries come from OpenAlex/arXiv metadata and abstracts; full texts were not read, and author names are given only where well known. **Coverage caveat:** general web search was unavailable in this session, dblp blocked automated queries, and Semantic Scholar rate-limited most requests. The search ran over OpenAlex (about 450 records across 38 queries, with abstracts) and the arXiv API. IEEE Xplore, the ACM DL and Google Scholar were not queried directly, though OpenAlex indexes most of their venues. Unpublished industry practice (in-house power viruses at CPU and GPU vendors) is by nature not covered. Treat "not found" below as "not found in these indexes", not "does not exist".

## Summary

1. **Power-virus and maximum-power search is an old, active field, and it is dominated by genetic algorithms**, with Bayesian optimization, gradient descent, ant colony optimization and constraint solving as alternatives. This holds from gate-level peak-power estimation in the 1990s to CPU and GPU stressmarks in 2024–2025.
2. **RL in hardware stimulus generation is used almost entirely for coverage closure**, not for maximizing power. No paper applying RL to power-virus or maximum-power stimulus generation was found.
3. **LLMs** are used for coverage-driven testbench stimulus, and in one 2024 paper for system-level test programs that target non-functional properties. No LLM power-virus generator for RTL blocks was found.
4. **ML power models** (PRIMAL, APOLLO, MasterRTL, PowerScope and others) predict average or per-cycle power. None found fits leakage, intercept and slope terms and tracks them from RTL through synthesis to post-route.

## 1. Maximum-power / power-virus search: methods used

### Gate-level peak power (sequential circuits), 1997–2015

| Work | Method | Notes |
|---|---|---|
| Hsiao, Rudnick, Patel, *Effects of delay models on peak power estimation*, ICCAD 1997 | GA | Peak power is sensitive to the delay model. That bears on our zero-delay limitation. |
| Hsiao et al., *Peak power estimation using genetic spot optimization*, DATE 1999 / VLSI Design 2001 | GA | Bounds 70.7% tighter than prior work on ISCAS89. |
| Hsiao et al., *Peak power estimation of VLSI circuits: new peak power measures*, TVLSI 2000 | GA | Generates a **"functional vector loop for near-worst case power"**, which is essentially our periodic-pattern search space. |
| *Maximum power estimation for CMOS sequential circuits by GA*, 2001 (and GSAA, 2005; hybrid GA, ICNC 2007) | GA, GA plus simulated annealing | Lower bounds on ISCAS benchmarks. |
| *ALPS*, GLSVLSI 2003 | Search with state reachability | Shows that ignoring reachable states underestimates peak power. |
| *Power virus generation using behavioral models*, VTS 2007 | Word-level integer constraint solving | Works on RTL-like behavioral models rather than gates. |
| *Controllability-driven power virus generation*, VLSID 2007 | D-algorithm guided by controllability | |
| *ACO-based pattern generation for peak power*, ISQED 2009 | Ant colony optimization | |
| *Efficient peak power estimation using probabilistic cost-benefit analysis*, VLSID 2015 | Greedy/probabilistic | |

### Processor and GPU stressmarks, 2008–2026

| Work | Method | Reported result |
|---|---|---|
| Joshi et al., *Automated microprocessor stressmark generation*, HPCA 2008 | Search over about 40 knobs of a synthetic benchmark | Replaces hand-tuned stressmarks. |
| Ganesan et al., *SYMPO*, PACT 2010 | GA over an abstract workload | 14–41% more power than MPrime. |
| *MAMPO*, multicore, 2011 | GA | Multithreaded power viruses. |
| Kim et al., *Automated di/dt stressmark generation*, 2011; IEEE Micro 2013 | GA | More than 40% more voltage droop than hand-coded di/dt stressmarks. |
| *Harnessing CPU EM emanations for voltage-noise characterization*, IEEE TC 2020 | GA (EM-sensed fitness) | |
| Hadjilambrou et al., *GeST*, ISPASS 2019 | GA | Power, thermal, IPC and dI/dt targets; beats hand-written tests. |
| *MicroGrad*, ISPASS 2021 | Gradient-descent tuning of code-generator knobs | |
| *Guser*, HPCA 2024 | **Bayesian optimization** (Optuna) | First systematic GPGPU stressmark generator; 109.3 W on Tesla T4 and 238.7 W on A10 (the comparison baseline was not checked). |
| *SAGA*, ISPASS 2025 | **Surrogate-assisted GA** | Surrogate fitness to cut the number of real power measurements. |
| *Exploration of LLM workload reliability (di/dt)*, HPCA 2026 | Stressmark framework with controllable oscillation | Targets LLM-inference power oscillations on GPUs. |

**Takeaway.** Where the fitness is a measured power number and the candidate is a fixed-length genome, the field converged on GA and, more recently, BO and surrogate-assisted GA. That is the closest prior setting to ours, and it is the reason GA is a strong baseline here, not a strawman.

## 2. RL for hardware stimulus generation

| Work | Objective | Compared against |
|---|---|---|
| *A RL approach to directed test generation for shared memory verification*, DATE 2020 | Coherence-protocol coverage | Constrained-random |
| *Automated test generation for hardware Trojan detection using RL*, ASP-DAC 2021 | Rare-trigger activation | Prior Trojan-test methods |
| *VeRLPy*, 2021 | Coverage (open-source library) | Random |
| *RL made affordable for hardware verification engineers*, Micromachines 2022 | Coverage | Random |
| *Efficient stimuli generation using RL*, SMACD 2024 | Code coverage | Random |
| *Advanced RL algorithms to optimize design verification*, DAC 2024 | **Maximize average FIFO depth** in a MESI controller (DDPG-PER) | DDPG with uniform replay |
| *Multicore environment state representation for agent-directed test generation*, DAC 2025 | Multicore coverage | |
| *StimulusRL* (DQN, DVSBench), 2026; *CA-PPO-AXI* (PPO), 2026; *RLConcolic*, DATE 2026 | Coverage, bugs, branches | Constrained-random, concolic |
| *CovR*, *CHORUS*, *TB or Not TB* (DPO), 2025–2026 | RL **post-training of LLMs** for high-coverage testbenches | SFT baselines |

**Takeaways.**

- The objective is almost always coverage. The closest analogue to maximizing power is the DAC 2024 paper that maximizes average FIFO depth, a scalar "stress" metric, but it compares RL variants only with each other.
- **Baselines are usually constrained-random**, rarely GA or BO. A result of the form "RL beats random" is common. "RL beats a tuned GA at equal simulation budget" is rare. That is the comparison this project makes, so the literature gives little reason to expect RL to win by default.
- RL fits naturally where there is sequential state to exploit, such as protocol states, coherence and deep pipelines. The power-virus literature mostly treats the stimulus as a static genome.

## 3. LLMs for stimulus / stress generation

- *LLM4DV* (FCCM 2025), *VerilogReader* (2024), *ChatTest* (DATE 2026): LLM-generated stimuli for coverage.
- **Closest to our LLM-as-policy baseline:** *Large Language Models to Generate System-Level Test Programs Targeting Non-functional Properties* (arXiv 2403.10086, 2024). An LLM writes system-level test programs aimed at non-functional properties of the device under test, at system level rather than RTL. Only the abstract was read; which properties and baselines it reports is to be checked in the full text.
- *POET* (2026), *RTLScout* (arXiv 2606.06530, 2026), *RTL-OPT* benchmark (arXiv 2601.01765, 2026): LLMs *optimize* RTL for power, performance and area (POET specifically power). This is the inverse direction: lowering power rather than finding its worst case. They are useful as related work and as a possible downstream use of our worst-case vectors.

## 4. Power modeling and proxies

- ML power estimators: *PRIMAL* (DAC 2019), *APOLLO* (MICRO 2021; runtime power proxies from a small set of RTL signals), *MasterRTL* (ICCAD 2023), *PACE* (2024), *ML-Power* (ISLPED 2025), *PowerScope* (2026, intra-cycle), *AtomPower* (2026). They predict average or per-cycle power from activity features. **None found decomposes power into leakage, intercept and slope, or reports how those terms move from RTL to synthesis to post-route.**
- *DiffPower* (arXiv 2608.03778, 2026): differentiable switching-power analysis on GPU. This is a possible *gradient-based* competitor for worst-case search over netlist inputs and worth tracking.
- *Stochastic Activity Prediction for Wallace-tree multipliers* (arXiv 2607.08002, 2026): operand Hamming weight predicts multiplier switching. This is relevant to why input toggles mislead for the bf16 multiplier (see `docs/decisions.md`).
- HLS-side ML (Pyramid, HLSFactory, cross-FPGA transfer) targets FPGA resources and timing, with little on ASIC power.

## 4b. Patents (Google Patents, 2026-10-28; a quick search, not freedom-to-operate)

- **US20260004151A1** (Intel; priority 2024-12-23; pending): reorders weights at compile time to minimize switching activity in NPU MAC arrays, using a Hamming-distance metric and a greedy pivot ordering, with channel order kept consistent across layers. It directly overlaps an operand-reordering study we considered and set aside. It is related work for this project's worst-case search and power-term decomposition, but does not overlap them.

## 5. What this means for our claims

**Not novel (cite, don't claim):**
- searching for maximum-power stimulus with GA
- periodic vector loops as the search space (Hsiao et al., 2000)
- BO and surrogate-assisted stressmarks
- ML power modeling in general

**Apparently open, within the coverage caveat above:**
1. **RL for power-virus generation at RTL/gate level**, specifically for HLS-generated ML-accelerator datapath blocks, compared at **equal proxy budget** with GA, random and hand-crafted patterns, and with an LLM-as-policy baseline still to come.
2. **Fail-closed validity gates with a logged reward-hacking record** for power-virus search. Prior stressmark work constrains the generator (legal instructions) but does not report rejected high-reward attempts.
3. **Proxy fidelity for worst-case ranking**: how well a cheap RTL toggle proxy ranks candidates against post-route power, including how much power is lost by trusting the proxy's choice. SAGA uses a surrogate but optimizes measurement cost, not fidelity to post-route.
4. **Leakage, intercept and slope tracked from RTL through synthesis to post-route** for HLS blocks, with the clock tree's effect on the intercept quantified per block and per pipeline depth.

**Suggested claim wording for the README:** *"To our knowledge, the first open comparison of RL, GA, random and LLM-policy search for worst-case activity in HLS-generated accelerator blocks under fail-closed validity gates, with proxy fidelity measured against post-route power."* Keep "to our knowledge", and cite the GA lineage explicitly.

## 6. What the literature suggests for the RL side

- **Add the baselines the field actually uses:** BO (Optuna/TPE, as in Guser) and a surrogate-assisted GA (SAGA-style), as well as GA and random. Beating random alone would not be credible.
- **Report sample efficiency** (best-so-far against evaluations), not just the final best. GA and BO papers are judged on measurement count.
- **Where RL should help:** blocks with sequential state, where the order of inputs matters (the MAC's accumulator and `last` timing). **Where it may not:** stateless datapaths, where the problem reduces to choosing a short set of operand pairs and GA crossover keeps the useful bit patterns intact.
