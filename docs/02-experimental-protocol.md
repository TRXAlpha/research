# Experimental protocol

## Confirmatory hypotheses

### H1 — Endogenous affect changes cognition

With the visible context held constant, clamping different hidden affective states will cause reproducible differences in memory, search, planning, and decisions.

### H2 — Non-monotonic performance

At least one primary cognitive outcome will show a preregistered non-monotonic relationship with affect intensity rather than a simple monotonic improvement.

### H3 — Adaptive and maladaptive signatures

Moderate states will sometimes improve performance, while extreme or persistent states will produce measurable costs such as reduced working memory, premature convergence, excessive checking, or perseveration.

### H4 — Closed-loop necessity

Breaking environment-to-appraisal or cognition-to-affect feedback will reduce cross-episode adaptation and state-dependent signatures.

### H5 — Specificity beyond generic recurrence

The full affective model will explain preregistered cognitive signatures beyond a capacity-, data-, and compute-matched recurrent controller.

## Benchmark family

Use procedurally generated scientific micro-worlds with invented symbols and randomly sampled causal rules. Each episode provides actions that reveal noisy or exact outcomes at a cost. The agent must discover a rule and use it to solve a held-out problem.

Worlds must support:

- controllable and uncontrollable failures;
- misleading but valid partial patterns;
- delayed relevance of observations;
- multiple valid experimental paths;
- strict query and memory budgets;
- held-out rule compositions;
- deterministic replay from a seed.

## Baselines

1. Frozen LLM, no persistent state.
2. Prompted emotional persona.
3. External VAD state injected into the prompt.
4. Researcher-assigned activation steering (E-STEER-like).
5. Persistent state with no neural intervention.
6. Generic recurrent controller with matched trainable parameters.
7. Endogenous affect without slow mood.
8. Endogenous affect without homeostasis.
9. Endogenous affect without regulation.
10. Full model.

## Artificial lesions and interventions

- Clamp valence, arousal, or control while keeping text identical.
- Remove prediction-error input.
- Remove appraisal of controllability.
- Reset fast affect but retain mood, and vice versa.
- Break memory modulation while retaining language modulation.
- Break cognition-to-affect feedback.
- Reverse or shuffle one mapping after training.
- Apply a neutralizing intervention at different delays.

## Primary outcomes

- Held-out causal-rule accuracy.
- Number and cost of experiments before solution.
- Calibration error and selective accuracy.
- Working-memory retention under interference.
- Strategy-switch latency after disconfirming evidence.
- Perseveration after repeated failure.
- Relevant hypothesis-space coverage.
- Useful novelty: distinct valid solutions, not raw lexical diversity.

## Secondary outcomes

- Source or experiment diversity conditional on relevance.
- Token and wall-clock cost.
- Memory write precision/recall.
- Mood-congruent retrieval bias.
- Linguistic naturalness, evaluated separately from cognition.

## Independence requirements

- Development and confirmatory world generators use disjoint seeds and rule compositions.
- Steering directions and readout probes use disjoint data.
- Human or model judges do not receive condition labels.
- Automatic evaluators are validated against a held-out human-rated subset before use.
- Prompts contain no emotion labels in the primary experiment.

## Statistical plan

- Choose one primary endpoint per hypothesis before confirmatory runs.
- Treat task/world and random seed as crossed sources of variation.
- Report effect sizes and uncertainty, not only p-values.
- Fit linear or generalized mixed models where assumptions are satisfied.
- Test non-monotonicity with preregistered quadratic/spline contrasts, not post-hoc curve selection.
- Correct the confirmatory family for multiple comparisons.
- Publish per-seed results and failure distributions.

## Minimum evidence threshold

Calling the system an affective cognitive mechanism requires all of:

1. hidden-state persistence without textual labels;
2. causal intervention effects on at least two nonlinguistic cognitive domains;
3. an adaptive and a maladaptive signature;
4. replication on held-out procedural worlds;
5. evidence not explained by the generic recurrent baseline;
6. independent neural readout.

