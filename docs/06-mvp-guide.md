# MVP testing guide

## What the MVP demonstrates

The MVP is a local, frozen 1.7B-parameter language model with:

- automatically updated persistent affect outside the prompt;
- observable world events rather than researcher-selected emotion labels;
- disjoint calibration text for neural writing and neural readout;
- a write intervention near 55% of transformer depth;
- an independent readout near 82% of transformer depth;
- baseline and affective generation from the exact same prompt;
- an automatic outcome-to-next-state loop in the benchmark;
- full JSONL/CSV logging.

It demonstrates engineering feasibility. It does not establish genuine emotion, cognitive benefit, or the final novelty claim.

## One-time setup

From PowerShell:

```powershell
cd C:\Users\chris\Documents\research
.\scripts\setup_mvp.ps1
```

The repository already contains the environment and model on the development machine, but the setup script makes the procedure reproducible.

## Interactive test

The easiest option is to double-click `RUN_MVP.bat` in the project folder.

From PowerShell, the equivalent command is:

```powershell
.\scripts\run_mvp.ps1
```

For a deterministic non-interactive demonstration:

```powershell
.\scripts\run_mvp.ps1 --demo --max-new-tokens 64
```

Suggested sequence:

```text
/state
Give three hypotheses for why an experiment produced an unexpected result.
/event uncontrolled_failure
/event threat
/state
Give three hypotheses for why an experiment produced an unexpected result.
/event rest
/event unexpected_success
Propose a creative but testable follow-up experiment.
```

Each ordinary prompt prints:

1. the frozen-model baseline;
2. the response with hidden-state affect intervention;
3. the hidden V/A/D-like write coordinates;
4. separate layer-24 readouts for both responses.

The prompt shown to the model never contains the affective state.

## Automated closed-loop smoke benchmark

```powershell
.\scripts\run_neural_benchmark.ps1
```

The affective agent's correct or incorrect answer automatically generates the next `success` or `controlled_failure` event. Results are written to:

- `runs/neural-benchmark.csv`
- `runs/neural-benchmark-session.jsonl`

The checked MVP run is summarized in `docs/07-mvp-results.md`.

## Strength adjustment

```powershell
.\scripts\run_mvp.ps1 --gain 0.5
.\scripts\run_mvp.ps1 --gain 1.1
```

Very high gains may reduce coherence. That failure is an expected boundary condition and must not be hidden.

## Current methodological limitations

- Contrastive directions are derived from short English calibration statements.
- The read layer and examples are independent, but the readout has not yet been validated cross-model.
- The model is still small and not a suitable basis for broad claims about advanced reasoning.
- The state equations are transparent hypotheses, not learned developmental dynamics.
- The automated benchmark is a smoke test, not a statistically powered experiment.
- Full global modulation of memory, search depth, and plasticity is not yet connected to the LLM.
