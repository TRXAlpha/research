# MVP results — engineering proof of concept

Run date: 2026-09-17

## Configuration

- Backbone: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- Parameters: approximately 1.7B
- Device for checked run: CPU
- Backbone weights: frozen
- Neural write layer: 13 of 24
- Independent read layer: 19 of 24
- Hidden width: 2048
- Steering gain: 0.85
- Write calibration: 36 contrastive statements
- Read calibration: 24 disjoint contrastive statements
- Decoding: greedy, same seed and prompt for both conditions

## Closed-loop smoke benchmark

Eight multiple-choice reasoning tasks were processed sequentially. The affective condition began near baseline. After every answer, correctness automatically produced a `success` or `controlled_failure` event, which updated homeostasis and affect before the next task.

Results:

- Baseline accuracy: 3/8 = 37.5%
- Affective accuracy: 3/8 = 37.5%
- Exact output difference: 4/8 = 50%
- Final valence: -0.473
- Final arousal: 0.563
- Final frustration: 0.802

## Causal changes observed

The hidden-state intervention changed four outputs while prompt, model weights, and decoding seed remained fixed.

Examples:

- Task 4, modus tollens: baseline selected `D) Nothing`; the affective run selected the correct `B) P is false`.
- Task 2, syllogism: baseline selected the correct `B) No`; the affective run changed to the incorrect `A) Yes`.
- Task 7, mislabeled boxes: both were incorrect, but the selected options differed.
- Task 6 retained the correct option while altering output form.

This is exactly the qualitative pattern required for the engineering proof of concept: persistent endogenous state can causally modify internal computation and can produce both apparently adaptive and maladaptive changes. Equal aggregate accuracy prevents claiming a benefit from this tiny run.

## What this proves

- A world-outcome sequence can update affect automatically without emotion labels in the prompt.
- Affect can persist between independent LLM calls.
- The state can be written directly into a frozen model's hidden activations.
- The intervention can alter task answers under an otherwise matched condition.
- The same mechanism can improve one answer and impair another.
- The full run is reproducible and logged.

## What this does not prove

- Statistical reliability.
- General cognitive improvement.
- Biological validity of the current equations.
- Genuine emotion or subjective experience.
- Independence from all generic recurrent controllers.
- Cross-model or cross-task generalization.

## Reproduction

```powershell
cd C:\Users\chris\Documents\research
.\scripts\run_neural_benchmark.ps1 --output runs\neural-benchmark-1.7b.csv
```

Artifacts:

- `runs/neural-benchmark-1.7b.csv`
- `runs/neural-benchmark-session.jsonl`
- `artifacts/calibration-report.json`
- `artifacts/neural-calibration-SmolLM2-1.7B-Instruct.pt`

## Next confirmatory step

Increase the procedural task family, add dose conditions, compare against external VAD and a matched generic recurrent controller, and lock the analysis before confirmatory seeds are generated.

