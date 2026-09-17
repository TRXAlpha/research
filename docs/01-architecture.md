# Architecture specification

## 1. Separation of levels

The architecture separates four processes that are often incorrectly collapsed:

1. **Constitution** — fixed update laws and architectural priors.
2. **Development** — calibration during training.
3. **Elicitation** — automatic online state change after appraisal.
4. **Regulation** — later, limited attempts to reappraise or inhibit a state.

The agent never receives an action such as `activate_curiosity`. Affect is part of the transition function.

## 2. State

The initial reference implementation uses interpretable computational variables rather than neurotransmitter names:

```text
Affect = {
  valence, arousal, dominance,
  stress, frustration, fatigue,
  mood_valence, mood_arousal
}

Homeostasis = {
  certainty, competence, progress,
  resources, safety, affiliation
}
```

All variables are bounded. Fast affect changes quickly; mood integrates affect slowly.

## 3. Appraisal

An event is evaluated relative to goals and internal state:

```text
Appraisal = {
  novelty,
  goal_congruence,
  controllability,
  certainty,
  agency,
  urgency,
  prediction_error
}
```

The same external event may therefore yield different state changes when competence, resources, or prior state differ.

## 4. Automatic transition

The conceptual transition is:

```text
world event
  -> prediction and appraisal
  -> fast affect update
  -> slow mood integration
  -> compulsory cognitive modulation
  -> action and outcome
  -> next appraisal
```

The reference equations in `affect.py` are transparent hypotheses, not a fitted biological model. They exist to make invariants testable before an LLM is introduced.

## 5. Cognitive modulation

The affective substrate emits continuous gains for:

- attention;
- working memory;
- memory consolidation;
- memory retrieval;
- exploration;
- verification;
- planning depth;
- creative divergence;
- plasticity.

The mapping intentionally contains non-monotonic terms. Moderate arousal can increase engagement while excessive arousal, stress, or fatigue can reduce working-memory capacity and flexible search.

## 6. LLM integration plan

The LLM will remain frozen during the first neural-intervention study. Candidate write mechanisms:

1. FiLM-style per-layer gain and bias generated from affect.
2. A low-rank hypernetwork that produces bounded adapter gates.
3. Dynamic mixtures of LoRA experts.
4. Residual activation steering as a prior-art baseline.

Candidate read mechanisms:

1. Held-out linear probes trained on different examples from steering construction.
2. Sparse-feature decoders trained on separate layers.
3. Behavioral signatures defined before intervention.
4. Cross-model and cross-task decoding.

The same vector must never be the sole write direction and measurement axis. Otherwise, for `h' = h + alpha*v`, projection onto `v` necessarily increases by `alpha*||v||^2`, creating circular evidence.

## 7. Multiple intervention sites

The final architecture should allow affect to modulate:

- attention logits or head gains;
- residual/MLP gains;
- memory write priority and retrieval scoring;
- search branching and compute allocation;
- verification thresholds;
- bounded online plasticity in a later phase.

This multi-site coupling is required to distinguish global metacontrol from output-style steering.

## 8. Regulation

Regulation is a separate controller with costs and limits. It may:

- re-evaluate controllability or goal relevance;
- inhibit an action tendency;
- allocate additional verification compute;
- gradually alter affect dynamics.

It may not directly overwrite the affect vector at zero cost. Failed or partial regulation is an intended phenomenon.

