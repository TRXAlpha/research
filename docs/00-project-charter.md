# Project charter

## Working title

**Endogenous Affective Dynamics as Causal Metacontrol in Language-Model Agents**

Suggested subtitle: **Adaptive and maladaptive effects on memory, reasoning, and creative exploration**

## Research question

Do biologically constrained, endogenous affective dynamics produce distinct and reproducible cognitive regimes in a language-model agent beyond those produced by prompting, externally assigned emotion, or a capacity-matched generic recurrent controller?

## Central hypothesis

Under bounded memory, time, and compute, low-dimensional affective state can act as global metacontrol. Because the state is automatically elicited by appraisal and internal regulation rather than selected as an action, it should sometimes improve resource allocation and sometimes create systematic cognitive distortions.

## Necessary properties

The proposed affective state must be:

1. **Endogenous** — elicited by events relative to the agent's goals and internal variables.
2. **Constitutive** — updated automatically; the policy cannot choose `emotion = X`.
3. **Persistent** — maintained outside visible text across steps and turns.
4. **Multi-timescale** — phasic signals, episode-level emotion, and slower mood.
5. **Globally causal** — able to affect more than prose style.
6. **Bidirectionally coupled** — cognition changes action and outcomes; outcomes update affect.
7. **Regulatable but not freely selectable** — reappraisal is a separate, limited process.
8. **Fallible** — excessive states may impair performance.

## Primary contribution sought

The contribution is not a new emotion label, VAD representation, steering vector, or persistent state machine. It is a causal test of whether an endogenous affective physiology supplies a computationally distinct form of metacontrol.

## What would falsify the thesis

The strong thesis should be rejected or narrowed if any of the following holds:

- A parameter- and compute-matched generic recurrent controller reproduces all gains and impairments.
- Effects disappear when linguistic emotion cues and evaluator leakage are removed.
- Independent probes cannot recover state better than chance.
- Benefits occur only on development worlds or one model family.
- The affect mechanism changes output tone but not memory, reasoning, or decisions.
- Readback is explained by projecting onto the same direction used for intervention.

## Scope boundaries

- No claim of subjective feeling, sentience, consciousness, or biological homology.
- No human-subject study in the first stage.
- No simultaneous attempt to model dopamine, serotonin, norepinephrine, cortisol, the amygdala, hippocampus, basal ganglia, and prefrontal cortex anatomically.
- Biological names are used only after a computational variable has a precise definition and testable mapping.

## Deliverables

- A systematic prior-art matrix.
- A preregistered experimental protocol.
- An open implementation with frozen configurations.
- Procedurally generated, contamination-resistant cognitive environments.
- Independent neural measurement and intervention channels.
- Causal ablations and matched baselines.
- A paper reporting positive, null, and maladaptive effects.

