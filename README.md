# Endogenous Affective Metacontrol

Research code and protocol for testing whether an autonomous affective substrate can causally reorganize cognition in language-model agents.

The project does **not** claim that an artificial system feels, is conscious, or has biological neurotransmitters. It tests a narrower computational hypothesis:

> Under bounded resources, biologically constrained endogenous affective dynamics can create reproducible cognitive regimes—both adaptive and maladaptive—that are not explained by prompting, external emotion labels, or a capacity-matched generic recurrent controller.

## What is different from an emotional chatbot

- Affect is generated automatically from appraisal, prediction error, goals, and synthetic homeostatic variables.
- The agent cannot select an emotion as a strategic action.
- State persists outside the text context and evolves on fast and slow timescales.
- Affect is intended to modulate attention, memory, planning, exploration, verification, and plasticity—not only wording.
- The design requires causal interventions, artificial lesions, and independent read/write measurements.
- Adaptive and maladaptive effects are both primary outcomes.

## Repository map

- `docs/00-project-charter.md` — scientific question, scope, and success criteria.
- `docs/01-architecture.md` — proposed affective substrate and LLM coupling.
- `docs/02-experimental-protocol.md` — hypotheses, baselines, metrics, and statistics.
- `docs/03-prior-art.md` — current novelty boundary and closest work.
- `docs/04-claims-ethics.md` — permitted claims, safety, and anthropomorphism controls.
- `docs/05-brief-profesor-ro.md` — Romanian briefing and decisions for the neuroscience supervisor.
- `docs/06-mvp-guide.md` — setup and interactive testing.
- `docs/07-mvp-results.md` — checked proof-of-concept results and limitations.
- `ROADMAP.md` — staged implementation plan.
- `literature/prior-art.csv` — structured living evidence table.
- `src/affective_metacontrol/` — executable reference implementation.
- `tests/` — invariants and reproducibility tests.
- `references/references.bib` — initial primary-source bibliography.

## Current milestone

Milestone 0 is implemented: a dependency-free endogenous affect engine, a procedural causal-discovery micro-world, stateless/generic/affective baselines, and a reproducible experiment runner. The MVP additionally runs a frozen local 1.7B model with calibrated neural write/read channels. This is engineering evidence, not confirmation of the scientific hypothesis.

The first 100-episode plumbing run yields equal perfect end accuracy across all three toy controllers. That is expected in the current easy world family and is recorded as a negative/sanity result; future benchmark versions must create controlled ambiguity, interference, and resource pressure before cognitive differences can be meaningfully tested.

For the neural proof of concept, see `docs/06-mvp-guide.md` and run `scripts/run_mvp.ps1`.

Fast checked demonstration:

```powershell
.\scripts\run_mvp.ps1 --demo --max-new-tokens 64
```

On Windows you can also double-click:

- `RUN_MVP.bat` for the interactive comparison;
- `RUN_DEMO.bat` for a deterministic demonstration;
- `RUN_BENCHMARK.bat` for the eight-task closed-loop smoke benchmark.

## Run locally

Requires Python 3.11 or newer.

```powershell
cd C:\Users\chris\Documents\research
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
python -m affective_metacontrol.cli --episodes 100 --budget 6 --seed 7
```

For editable installation:

```powershell
python -m pip install -e .
affective-metacontrol --episodes 100 --budget 6 --seed 7
```

## Scientific status

The novelty claim is a working hypothesis and must be updated through a systematic review. Public code, preprints, workshop papers, and peer-reviewed work are tracked separately. In particular, persistent VAD state, activation steering, affective appraisal, and emotion-dependent reasoning already exist individually and in several close combinations.
