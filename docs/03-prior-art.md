# Prior art and novelty boundary

Status date: 2026-09-17. This is a living scoping review, not yet a systematic review.

## Unsafe novelty claims

The project must not claim to be first to provide:

- artificial emotions;
- appraisal-based affective agents;
- persistent VAD state;
- emotion representations in LLM hidden states;
- representation-level emotion steering;
- emotional effects on LLM reasoning;
- separation between internal appraisal and verbal formulation;
- model-internal emotional signals for agent decisions;
- a symbolic affect/LLM steering feedback loop.

## Closest work

### Controlling Long-Horizon Behavior in Language Model Agents with Explicit State Dynamics (2026, preprint)

Maintains continuous external VAD state with first- and second-order temporal dynamics and injects it into generation. It establishes persistence, inertia, hysteresis, and recovery, but the state is external and the study focuses on dialogue behavior.

URL: https://arxiv.org/abs/2601.16087

### How Emotion Shapes the Behavior of LLMs and Agents: A Mechanistic Study / E-STEER (2026, preprint)

Uses SAE-derived VAD features for direct hidden-state intervention and evaluates objective reasoning, subjective generation, safety, and multi-step agents. It already reports non-monotonic emotion–behavior relationships. Its emotion is experimentally assigned rather than autonomously generated from an agent–environment physiology.

URL: https://arxiv.org/abs/2604.00005

### Feeling First, Speaking Second (2026, workshop paper)

Separates visceral appraisal from strategic formulation and combines dynamic emotional state, long-term memory, personality, and goals. Evaluation centers on simulated narrative scenarios and LLM-as-judge assessments rather than independent neural intervention and broad cognitive benchmarks.

URL: https://aclanthology.org/2026.cas-1.7/

### Emotion2Skill (2026, preprint)

Extracts a 27-dimensional residual-stream emotion signal and uses it for skill routing and skill evolution. It shows that internal emotion-associated signals can inform agent decisions, but does not establish an autonomous affective physiology globally coupled to cognition.

URL: https://arxiv.org/abs/2608.09248

### Synthetic emotions and consciousness: exploring architectural boundaries (2026, peer reviewed)

Provides a biologically motivated controller in which immediate needs and episodic affective memory converge on action selection. It is important prior art for homeostatic and hierarchical emotion-like control, though it is not the same as neural metacontrol inside an LLM.

URL: https://doi.org/10.1007/s00146-026-02896-z

### npc-steering (2026, public implementation)

Implements a six-scalar symbolic state (V/A/D, stress, frustration, fatigue), persona-dependent decay, hand-authored appraisal, residual-stream steering of a frozen LLM, and projection back onto the same affect directions. This is extremely close to the engineering skeleton of persistent affect plus activation steering.

Its limitations define part of the present gap:

- appraisal is substantially hand-authored;
- the state remains a symbolic external controller;
- intervention and readback share directions, creating possible circularity;
- the demonstrated domain emphasizes NPC monologue and behavior;
- it does not compare against a capacity-matched generic recurrent controller on broad cognitive outcomes.

URL: https://github.com/matbeedotcom/npc-steering

## Working gap

The currently defensible gap is:

> A confirmatory causal test of whether biologically constrained, endogenous, persistent affective dynamics create adaptive and maladaptive cognitive regimes in language-model agents beyond external steering and generic recurrence.

The gap requires the conjunction of:

- autonomous elicitation from appraisal and internal variables;
- bidirectional environment–affect–cognition coupling;
- global multi-site modulation;
- independent write/read channels;
- cognitive rather than primarily expressive outcomes;
- matched-controller comparisons;
- lesion, clamp, and dose-response experiments.

## Systematic-review search plan

Databases: Google Scholar, Semantic Scholar, ACL Anthology, IEEE Xplore, ACM DL, Scopus/Web of Science if institutionally available, PubMed for biological mappings, arXiv for current preprints, and GitHub for public implementations.

Core query blocks:

```text
(LLM OR "language model agent") AND
(emotion OR affect OR appraisal OR homeostasis OR neuromodulation) AND
(persistent OR endogenous OR autonomous OR dynamical) AND
(reasoning OR memory OR cognition OR creativity OR planning)
```

```text
(transformer OR LLM) AND
(VAD OR valence-arousal-dominance OR "affective state") AND
(activation steering OR residual stream OR LoRA OR FiLM)
```

Record title, year, venue/status, architecture, source of affect, persistence, intervention site, feedback direction, tasks, baselines, code, limitations, and novelty overlap.

