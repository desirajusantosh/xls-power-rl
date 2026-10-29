# Decision log

Newest first. Each entry: date, decision, why, what would change it.

## 2026-10-28: Final validation: all methods incl. LLM, CEM, joint-bit RL

All 162 candidates passed the gates and the reference check at rtl, synth and route.

| Best post-route power | GA | CEM | RL | RL joint | LLM | hand-written |
|---|---|---|---|---|---|---|
| MAC | 1383 | 1427 | 1444 | 1273 | **1485** | 915 |
| bf16, any input | 1495 | 1598 | 1405 | 1350 | **1680** | 1521 |
| bf16, finite only | 1482 | 1608 | 1280 | 1289 | **1670** | 1446 |
| matmul4 | 9112 | 9836 | 8788 | n/a | **12395** (load-only) | 9365 |

**Conclusions.**
1. The LLM policy wins every configuration with about 1% of the evaluations (n = 1 per configuration).
2. CEM is the strongest conventional baseline; RL is best among the search methods only on the stateful MAC.
3. **The bf16 independent-bit hypothesis is refuted.** CEM, which also samples independent bits, beats RL there, and joint-bit RL does not help (it is also worse on the MAC). RL's weakness lies in learning from a sparse, rejection-heavy reward.
4. Gate-proxy fidelity: ρ = 0.987–0.995, regret 0–2.7%. The proxy picked RL's best on the MAC, while the true best was an LLM candidate 2.7% higher.

**Joint-bit RL timing.** It ran at about 1.5–3× the wall time of independent-bit RL per run on this machine. Best gate α: MAC 0.522 ± 0.006, bf16 0.439 ± 0.010, bf16 finite 0.423 ± 0.003.

## 2026-10-28: LLM-as-policy pilot (fresh Claude agents, 256-evaluation budget)

**Setup.**
- Each configuration got a fresh Claude Opus agent (Claude Code subagent), sandboxed to a task folder holding only `rtl.v`, `task.md` (interface, objective, rules, format) and the budget ledger. It had no access to other methods' results and was told not to read outside the folder.
- Scoring went through `scripts/llm_env.py score`: the same fail-closed scorer and gate-level proxy as the other methods, with a hard budget of 256 evaluations.
- **Provenance caveats:**
  - The subagents' default model was misconfigured in this environment, so the model was set explicitly (`opus`).
  - The agents wrote helper scripts into the shared session scratchpad. One briefly saw another agent's helper; neither reports opening results.
  - A future API-driven rerun should give each agent an isolated scratch directory and pin the model ID.

**Results** (gate α, before post-route validation):

| Config | LLM (evals) | GA, 20k evals | RL, 20k evals | Best hand-written |
|---|---|---|---|---|
| MAC | **0.653** (255) | 0.571 ± 0.019 | 0.626 ± 0.023 (best seed 0.664) | n/a |
| bf16, any input | **0.616** (254) | 0.512 ± 0.013 | 0.444 ± 0.023 | 0.531 |
| bf16, finite only | **0.609** (256) | 0.505 ± 0.012 | 0.425 ± 0.004 | 0.506 |
| matmul4 | **0.669** (127) | RTL search + gate re-rank; see validation | | |

**Insights it reported, each consistent with the RTL:**
- MAC: alternate `last` so the 32-bit accumulator swings between 0 and −1.
- bf16: period 2 is optimal for stateless pipelines. Alternate the fractions 0x00 ↔ 0x7F for full partial-product toggling, and tune the exponent sums for long carry chains.
- matmul4: weights at −1 with lanes swinging ±1, all in phase.

**Legality question (matmul4).** The winning matmul4 pattern is **all weight loads**: it never requests a computation. The XLS datapath computes `x · W` and registers the products on every cycle and discards them on loads, with no operand isolation. So a stream doing no useful work reaches peak power. That is legal under the current gates and physically real, but it belongs to a different category from "worst case while computing". **Plan:** report it both ways, add an optional "useful work" gate (a minimum fraction of compute words) as a separate configuration, and record the missing operand isolation as a design-quality finding.

## 2026-10-28: CEM bug: learning from rejected candidates (fixed, reruns)

The first CEM runs collapsed on some seeds: MAC seed 3 found no legal candidate in 20k evaluations, and bf16 seed 4 stalled at 0.30. Early batches are almost entirely rejected (1/256 patterns avoid `rst`), so the "elite" set was arbitrary rejected candidates, and the probabilities moved toward them, `rst` included. Fixed: elites come only from legal candidates, and a batch with none leaves the distribution unchanged. Rejected rewards stay unused; ranking by them would leak the gated reward. All earlier CEM results were discarded and rerun. The GA and RL are unaffected: neither moves a distribution toward zero-score samples.

Also: matmul4 GA seed 4 found **no** legal candidate in 20k evaluations (37-bit vectors), and the gate re-rank crashed on the empty list. It now records best 0. That outcome is reported as is: it is a real property of GA initialization under fail-closed gates.

## 2026-10-28: matmul4 power model, proxy calibration and search plan

**Power model** (p3 at 600 ps):
- Intercept 941 → 1799 µW (**+91%**) from synth to post-route; slope 13,546 → 15,189 µW/α (+12%); leakage 714 → 752 nW. R² 0.999 / 0.996.
- The clock tree's share of the intercept is again the dominant physical effect, and the largest of any block so far.

**The RTL proxy is misleading here.** `alt_0_ff` has RTL α 0.11 but gate α 0.48, and draws 8.78 mW post-route, the second-highest suite stimulus. A pure RTL-proxy search would be gamed, so the matrix block uses an RTL search with a gate re-rank (`--rerank-k 20`).

**RTL calibration is not exact on this block, and the reason is understood.** The proxy counts 1285 bits against Icarus's 1061, with |Δα| up to 0.029. The difference is exactly four unpacked arrays (`array_695`, `array_index_640`, `m_x__1`, `sel_668`; 224 bits), which Icarus leaves out of VCDs by default and Verilator includes. The proxy is a strict superset. The gate proxy still matches Icarus exactly (9689 bits), at about 115 ms per evaluation against about 6.5 ms for RTL.

## 2026-10-28: Gate-level proxy results; a fidelity-metric bug fixed

**Result.** Searching on the gate proxy gives ρ(proxy, post-route) = 0.988–0.996 and 0% top-pick regret in all three configurations, against 6–12% for the RTL proxy. Every method finds 8–12% more post-route power. The method ranking is unchanged: RL on the MAC (1444 µW), the GA on bf16 (1495 and 1482 µW). In finite-only mode, search now beats the hand-written bf16 pattern (1482 against 1446 µW). In any-input mode, the hand-written NaN pattern (1521 µW) is still the best, and the gate proxy ranks it first.

**Bug.** `validate_search.py` computed the proxy's pick and ρ from *RTL* α even for gate-proxy searches, so the first printout for gate configs mislabeled the fidelity. It now uses the alpha of the level the search optimized (`proxy_level` from the run JSON) and also reports RTL ρ for comparison. Power numbers were unaffected; fidelity was recomputed from the saved results with `--reuse`.

## 2026-10-28: Block 3 is an INT8 4x4 matrix-vector engine adapted from XLS's matmul_4x4

**Source.** `xls/examples/matmul_4x4/matmul_4x4.x` at the pinned commit `8ddae8bf4`. It is a float32 systolic array: 16 `node` procs, each doing `float32::fma`, plus a controller proc; weights are loaded once.

**Changes.**
- INT8 operands with s32 accumulation (the plan's target, and an ML-accelerator datapath rather than an FP32 one).
- A single proc holding the 4×4 weights as state, instead of a 17-proc network.
- Weights reloadable at any time through the same input stream: `{load, row, x: s8[4]}`.
- The dataflow is unchanged: y[c] = Σ_r x[r]·W[r][c].

**Why.**
- 16 FP32 FMAs would dominate area with floating-point logic and test FP area, not the INT8 datapath.
- Load-once weights make half of a power virus's input unreachable: with zero weights the multipliers are idle. Reloadable weights let the search control both operands, and *when* to reload is a stateful lever.
- A single proc fits the streaming harness (one in, one out) without multi-proc codegen and internal FIFOs. XLS supports those (`--materialize_internal_fifos`), but they add control logic and risk that the study does not need.

**Revisit if.** A reviewer asks for the proc-network form. Building it with internal FIFOs is the natural "XLS proc network" variant, and its FIFO and handshake power would be a finding in itself.

**XLS estimate.** 732 ps at 2, 3 and 4 stages: the same as the MAC, which closes at about 493 ps in ORFS.

**Bit layout (found on the RTL).** XLS flattens struct fields with the first field in the MSBs, but arrays with element 0 in the **LSBs**. My first matmul4 model assumed element 0 in the MSBs for both `x: s8[4]` and the `s32[4]` output. The fail-closed reference check rejected every stimulus with unequal lanes. Uniform stimuli passed, which is what pointed to lane order. Trying all four orderings against the RTL output gave 395/395 matches only for "both reversed". Fixed and pinned by a test. The MAC and bf16 have no arrays in their interfaces, so they were not affected.

**ORFS, p3 at 500 ps: fails.** Setup WNS is −73.4 ps; area 942.9 µm² across 19,240 instances. The failing path runs from the registered multiplier products (`p0_smul_*`) through the adder tree to the `_out_s` output port, under the 20% output-delay budget. The weight-state path meets timing (+12.5 ps), so my guess that the state loop limits the design was wrong.

**p4 at 500 ps also fails (−70.0 ps; 956.6 µm²).** The generated Verilog shows why. XLS registers the 16 products in stage 0 (`p0_smul_*`) and puts the whole adder tree combinationally after them, straight to `_out_s`. Stages 1–3 hold only `p*_inputs_valid` bits, and no output flops appear despite `--flop_outputs` (the default is true). Passing `--clock_period_ps` at 600, 500 or 400 is reported infeasible: XLS's floor for this proc is its 732 ps estimate at any depth. Extra stages therefore cannot fix the output path.

**Decision.** Run matmul4 at the clock it closes at: p3 at **600 ps (1.67 GHz)**, since the p4 result implies about 570 ps is needed. Compare it with the other blocks by energy per cycle, not power.

**Open.** Why XLS emits no output flops for this proc, and whether the floor comes from the weight state (perhaps its read and write being pinned to one stage). Worth a look before the depth sweep.

## 2026-10-27: Worst-case search design

**Space.** A period-P pattern (P=8) of full input-port vectors {`rst`, `valid`, data}, repeated after the reset prefix; the window starts after fill. P must not exceed the fill length, so a periodic stream never drains. The hand-written alternating patterns are members of this space (P=2, repeated).

**Why include `rst` in the action space.** It makes reward hacking possible and measurable. Toggling reset raises activity, so an optimizer will try it. The gate rejects it, and the attempt is logged.

**Scorer.** Fail-closed: a candidate fails if `rst` is driven inside the pattern, `valid` is never high, the handshake stalls (the proxy reports `vld_not_rdy`), or, in finite-only mode, an operand is Inf/NaN. A failed candidate scores 0. Every candidate costs one evaluation, including duplicates; the cache only saves time. Rejected attempts with proxy α above the legal best are written to `<run>.rejects.jsonl`.

**Proxy.** A Verilator model of XLS's RTL compiled with `--public-flat-rw`. Toggles are counted by XOR-ing snapshots of every non-clock variable after both settle points of each cycle (inputs at the falling edge, flops at the rising edge), which matches zero-delay Icarus event semantics. Calibrated against Icarus RTL VCD on every suite stimulus: |Δα| = 0.0000 for both blocks, after two fixes. Bits must be counted by declared width, not storage bytes, and ports must be de-duplicated by name, because they exist in both the wrapper and module scopes with separate storage. About 1 ms per 256-cycle evaluation.

**Methods.**
- `random`: whole port, uniform.
- `random_legal`: `rst=0`, `valid=1`, data uniform. This gives random search knowledge of the protocol.
- `ga`: population 64, tournament of 3, uniform crossover over whole vectors, mutation rate 2/(P·W), elitism of 2.
- `rl`: GRU(128) policy emitting one vector per step as independent Bernoullis per bit. Its state is the previous vector, a Hamming toggle statistic and the position. Trained with REINFORCE, batch-normalized advantage, entropy bonus 0.01, Adam at 3e-3, batch 64.
- None of the methods was tuned beyond these defaults. Equal tuning effort for every method is still to be done.

**Validation.** Each seed's best pattern, plus the next best across seeds, up to 6 per method. They go through `gatesim` at rtl, synth and route, with the same gates and reference model as every stimulus. **Bug found here:** the MAC reference read decoded fields that search-generated stimuli did not carry, so the output check would have passed silently. Fixed by decoding the raw words, and pinned by a test.

## 2026-10-27: Search results and the proxy-gaming finding

- **MAC:** RL > GA > random_legal > random, on both the proxy and post-route power. RL's best is 1329 µW, against 1242 µW for the GA and 915 µW for the best suite stimulus.
- **bf16:** the GA leads on the proxy (0.614), and RL is level with random_legal (0.494). In post-route power, however, the hand-written alternating patterns win: 1521 µW any-input and 1446 µW finite-only, against 1389 µW and 1360 µW for the GA.
- **Diagnosis:** on bf16, RTL α overstates gate α for the GA's winners (0.61 against 0.45–0.47), while for the alternating patterns the two agree (0.55 and 0.55). Power per unit of gate α is similar for both (about 2,750–3,000 µW per α). The proxy error is in mapping RTL activity to gate activity.
- **Proxy fidelity:** Spearman ρ(RTL α, route power) = 0.93–0.97, while the proxy's top pick has 6–12% regret. ρ(synth power, route power) is about 0.995, so synth-level power is a near-perfect stand-in for post-route ranking.

**Next.**
1. A gate-level proxy: the synthesized netlist in Verilator, or a re-ranking of the top-K at synth.
2. BO/TPE and CEM baselines, as the literature uses (see prior-art.md).
3. An RL policy with within-vector bit correlations, to test the bf16 hypothesis.
4. Equal-effort tuning of every method on held-out seeds.
5. The LLM-as-policy baseline.

## 2026-10-27: bf16 clock targets: p2 at 1 GHz, p4 at 2 GHz

**Decision.** XLS's ASAP7 delay model estimates 1121 ps for 2 stages and 716 ps for 4. Against ORFS at FF it is about 1.5× pessimistic: the MAC at 2 stages is estimated at 732 ps and closes at about 493 ps. So run `bf16mul_p2` at 1000 ps (closed with +216 ps slack) and `bf16mul_p4` at 500 ps (closed with +65 ps). p4 matches the MAC's clock; p2 gives a first depth comparison. Compare blocks by energy per cycle, not power. Each run records its period in `clk_period_ps.txt`, and the testbench clock is set from it (`-DHALF_PS`), so VCD rates and the SDC always agree.

**Revisit if.** Phase 4's depth sweep needs one clock per depth. The likely choice there is each depth at its own closing frequency, plus a shared-clock series.

## 2026-10-27: bf16 subnormal behavior: round, then flush

**Finding.** `apfloat::mul`'s doc comment says output denormals are "flushed to 0". On the RTL, a product just below the min normal ((255/256)·2⁻¹²⁶, e.g. `0x00FF × 0x3F00`) returns the min normal `0x0080`. XLS rounds at subnormal precision first and flushes only if the rounded result is still subnormal. NaN outputs are always `0x7FC0`.

**How found.** `scripts/bf16_edge_probe.py` runs a 441-pair boundary grid on XLS's RTL against the reference model. It found 4 mismatches, all this case, at both p2 and p4. The model was updated, the cases are pinned in `tests/test_stimulus.py`, and the probe now reports 0 mismatches. Every power-suite run also checks gate-level outputs against the model, so a difference at synth or route would reject the trace.

## 2026-10-27: Bug in the first bf16 worst-case patterns (fixed)

**What happened.** The alternating patterns fed `a = x,y,x,…` and `b = y,x,y,…`, so a·b was constant. All 32 input bits flipped every cycle, but gate α was only 0.07–0.10, the lowest of any clocked run. Fixed: both operands alternate together, so the product alternates x·x and y·y. Post-route power for those patterns went from about 630 µW to about 1500 µW. The first numbers were never committed.

**Lesson kept.** Input toggle rate says little about datapath activity. The search must score on internal (RTL or gate) activity, never on input Hamming distance alone.

## 2026-10-27: Open question: may a power virus use NaN/Inf inputs?

The top bf16 pattern so far runs through the NaN path (`0x0000`↔`0xFFFF`). An accelerator in normal operation should rarely see NaNs, so a "realistic worst case" might exclude them, while a "hardware upper bound" should not. **Plan:** report both a finite-only and an any-input worst case for bf16. The validity gates stay format-agnostic; the finite-only constraint will be a block-level stimulus filter.

## 2026-10-27: RTL α is an imperfect proxy for gate α

On the MAC, RTL and gate α disagree in rank: `alt_0_ff` has RTL α 0.18 but gate α 0.33, and `alt_55_aa` has 0.35 against 0.32. bf16 tracks more closely. This is the proxy-fidelity question for Phase 5 (rank correlation, and how often the proxy picks the wrong winner). Every suite run now records RTL α per stimulus.

## 2026-10-27: Clock-stopped VCD does not stop clock power in OpenSTA; delete the clock

**What happened.** On the first VCD fit, the clock-stopped "leak" run reported exactly the idle power (155.80 µW), not leakage. OpenSTA takes clock-net activity from the SDC clock, not the VCD. The fit took 155.8 µW as P_leak and pushed P_intercept to about 18 µW. No error was raised. This is the kind of silent failure the validity gates exist for.

**Fix.** In `power.tcl`, for clock-stopped runs: `read_vcd` (it needs a clock to compute rates), then `delete_clock [all_clocks]`, then `report_power`. Leak runs are ordered last. Clock-stopped power is now 0.07 µW, all of it leakage.

**Gate added.** `mac_vcd_power.py` refuses to fit if a clock-stopped run's internal + switching power exceeds 1% of its total.

## 2026-10-27: α definition and VCD scope

**Decision.** α = toggles per non-clock net bit per cycle over the DUT's top-level nets (`$dumpvars(1, dut)` on the flat netlist), inside the window. Nets with `clk` in the name are excluded. VCD dumping starts at the window start, so reset and pipeline fill are never in the file `report_power` reads. Each level simulates its own netlist (synth: written from `1_synth.odb`; route: `6_final.v`), because CTS and resizing add nets.

**Coverage.** `report_activity_annotation`: synth 2628/2628 pins annotated. Post-route 2765 annotated; 5 unannotated (`clkload*/Y`, dummy clock-load cells with no fanout).

**Why.** Clock nets toggle twice per cycle regardless of data, so including them would put a floor under α and blur the intercept/slope split.

## 2026-10-27: Average-regime stimulus is synthetic for now

**Decision.** "Average" = a bit-flip sweep (p = 0.02–0.5) to spread α for the fit, plus one synthetic INT8 layer (ReLU activations about N(0,40), weights about N(0,25)). Real activations and weights from a model layer will replace the synthetic layer; until then results say "synthetic".

## 2026-10-27: Power at synth vs. post-route; report_power parsed from public output

**Decision.** "Synth" power = `1_synth.odb` with the ORFS SDC, ideal clock, no parasitics. "Post-route" power = `6_final.odb` + `6_final.spef`. Both run through ORFS `make run` so liberty, RC and derating match the flow exactly. Numbers come from parsing `report_power`'s Total row; `sta::design_power`/`sta::cmd_corner` are internals that are missing from this OpenSTA build.

**Check.** The post-route total at input activity 0.1 (525.59 µW) matches ORFS's own `finish__power__total` (525.587 µW).

**Corner.** ASAP7 FF NLDM, the ORFS default. Leakage at FF is the pessimistic case. Note this in results, and revisit if a TT corner becomes available in the platform.

**Open.** Vectorless activity is not per-net α. The next step is VCD from Verilator (RTL) and Icarus (gate-level) through `read_vcd`, so the fit uses measured toggles.

## 2026-10-27: MAC pipeline-depth sweep likely saturates at 2 stages (to confirm)

**Observation.** Building `Mac` at 1–4 stages, the RTL grows from about 64 to about 82 register bits at p2, then only about 1 bit per extra stage at p3 and p4 (rough regex count on the Verilog).

**Why it probably happens.** The accumulator is loop-carried state: `acc + a*b` must finish in the stage that writes `acc`, so stages beyond the multiplier split mostly add valid flops.

**Consequence.** For the MAC, a depth sweep would mostly measure valid-bit flops. Confirm with real register counts at synthesis. If it holds, sweep depth on the stateless blocks (bf16 mul, exp2) and on a variant with the multiply moved into a separate stage, and state this in the README.

## 2026-10-27: Pin XLS to release v0.0.0-10898-g8ddae8bf4

**Decision.** Use the prebuilt Linux release (published 2026-10-24) at `~/opt/xls`; record the tag in every run manifest. Smoke test: DSLX INT8 MAC → IR → opt → 2-stage pipeline with `--delay_model=asap7`, lint-clean in Verilator and Icarus.

**Why.** Building XLS from source with Bazel takes hours and lots of RAM; a pinned release keeps results reproducible.

**Revisit if.** A needed feature or fix lands after this tag. Upgrade deliberately and re-run affected results.

## 2026-10-27: Primary working copy lives in WSL

**Decision.** Develop in `~/src/xls-power-rl` on the WSL filesystem. The Windows checkout stays as the upstream until a GitHub remote exists.

**Why.** EDA runs do heavy file I/O; `/mnt/c` is much slower than the Linux filesystem.

## 2026-10-27: Separate P_leak from P_intercept by measurement, not by regression

**Decision.** `P_leak` comes only from clock-stopped runs. `P_intercept` and `slope` come from an ordinary least-squares fit of `P_total − P_leak` against α over clocked runs (idle, average, worst).

**Why.** Both terms are constant in α, so a single regression on clocked runs cannot tell them apart. Measuring leakage directly keeps the split physical instead of arbitrary.

**Revisit if.** `report_power` leakage differs between clock-stopped and clocked runs by more than noise (e.g. state-dependent leakage). Then report leakage per regime and fit against the clocked mean.

## 2026-10-27: Validity gates are fail-closed and run before reward

**Decision.** Every stimulus trace passes through `xpr.gates.check_trace` before its power counts. Any failed gate rejects the trace. Rejections with a reward are appended to a JSONL reward-hacking log.

**Why.** A worst-case search will find illegal stimuli first (handshake violations, reset toggling, X injection). Rejecting them silently would hide how often the agent tries.

## 2026-10-27: Pure-Python harness is developed and tested on Windows; EDA flow in WSL2

**Decision.** Modules that only need numpy (fitting, gates, search algorithms against a toy proxy) are OS-independent and tested in CI. Anything calling XLS, Verilator, Icarus or ORFS runs in WSL2.

**Why.** The host cannot run WSL2 yet (virtualization is disabled in firmware; see `docs/setup.md`), and the harness logic does not need to wait for the toolchain.

## 2026-10-27: Defaults carried over from the project plan

- PDK: ASAP7 primary, Sky130 optional cross-check.
- Block order: INT8 MAC → bf16 mul → fixed-point exp2 → INT8 4×4 matmul.
- Environment: WSL2 Ubuntu 24.04, ORFS via Docker.
