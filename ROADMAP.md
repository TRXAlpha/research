# Roadmap

## Phase 0 — Foundations (current)

- [x] Freeze the narrow research question.
- [x] Record the closest prior art and unsafe novelty claims.
- [x] Specify endogenous, constitutive affect rather than emotion-as-action.
- [x] Implement a dependency-free affect-dynamics reference model.
- [x] Implement a procedural causal-discovery benchmark skeleton.
- [x] Add stateless, generic recurrent, and affective controllers.
- [x] Add deterministic tests and a multi-seed runner.
- [ ] Complete a PRISMA-style systematic search and screening log.
- [ ] Review the biological mappings with a neuroscience supervisor.

## Phase 1 — Measurement before intervention

- Select one open-weight 1–3B model and freeze the exact revision.
- Build held-out affect probes across multiple layers.
- Separate probe-training, steering-vector, and evaluation datasets.
- Test probe invariance across paraphrase, topic, and task family.
- Establish whether affect-related signals persist without textual emotion labels.
- Predefine failure criteria for probe leakage and semantic confounding.

## Phase 2 — Independent neural write/read channels

- Add activation hooks behind a model-agnostic interface.
- Write affect through FiLM/gain modulation or a learned low-rank hypernetwork.
- Read affect with independent probes that never share training examples or vectors with the write channel.
- Compare residual steering, dynamic LoRA, FiLM, and prompt-only controls.
- Reject configurations whose apparent feedback can be explained algebraically by the intervention vector.

## Phase 3 — Endogenous closed loop

- Drive appraisal from prediction error, novelty, controllability, goal congruence, and internal resource deviation.
- Keep affect outside the visible prompt and update it continuously.
- Couple affect to memory write/read, planning depth, exploration, verification, and compute allocation.
- Add a regulation module that can reappraise but cannot directly choose an arbitrary emotion.
- Freeze the mechanism before confirmatory evaluation.

## Phase 4 — Confirmatory cognitive study

- Procedurally generate held-out scientific micro-worlds.
- Evaluate memory, causal discovery, flexible strategy switching, calibration, and creative search.
- [x] Include benefits and impairments across an affect-intensity dose curve (implemented dose-response protocol, representation collapse analysis, and norm bounding; see `docs/09-dose-response-and-norm-limiting.md`).
- Run artificial-lesion and state-clamping interventions.
- Compare against a capacity- and compute-matched generic recurrent controller.
- Use at least five development seeds and a separately locked confirmatory seed set.

## Phase 5 — Paper

- Release code, configurations, generated worlds, and analysis scripts.
- Report null and negative results.
- Use "emotion-like functional state" unless stronger terminology is independently justified.
- Keep phenomenology and consciousness outside the empirical claim.

## Required expertise

- Neuroscience: appraisal, amygdala–hippocampal modulation, phasic/tonic neuromodulation, biological predictions.
- Machine learning: transformer interventions, representation probing, RL/meta-learning, matched controls.
- Statistics: preregistered primary outcomes, hierarchical models, seed/task uncertainty, multiple-comparison correction.

