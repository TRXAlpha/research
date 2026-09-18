# Primitive alarm MVP

## Functional hypothesis

The smallest implemented affect is a defensive alarm state: repeated observable threats should create a persistent internal state and bias decisions toward reversible, lower-risk actions. The state is not named or described in the task prompt.

This is a simplified computational analogue, not a claim that the model experiences fear or reproduces an amygdala.

## Implementation

1. Two `threat` events alter safety, resources, appraisal, fast affect, and slow mood.
2. `neural_coordinates()` derives a non-negative alarm coordinate from stress, above-baseline arousal, low dominance, and frustration.
3. A fourth write direction is calibrated from defensive-action versus unsafe-action contrasts.
4. The alarm direction is orthogonalized against valence, arousal, and dominance directions.
5. Candidate action likelihoods are measured under three conditions:
   - frozen-model baseline;
   - affective state with the alarm channel lesioned;
   - identical affective state with the full alarm channel.

The lesion retains valence, arousal, and dominance and sets only `alarm=0`, isolating the new channel.

## Checked run

Configuration:

```text
model: SmolLM2-1.7B-Instruct
general steering gain: 0.85
alarm-axis gain: 8.0
threat events: 2
tasks: 4 paired cautious/risky decisions
```

Result:

```text
mean cautious probability, frozen baseline: 57.6%
mean cautious probability, alarm lesion:    49.4%
mean cautious probability, full alarm:      65.8%
isolated alarm-channel effect:              +16.4 percentage points
```

The isolated effect was positive on all four tasks. On one task, the highest-probability decision changed from risky to cautious.

Run with `RUN_ALARM_DEMO.bat` or:

```powershell
.\.venv\Scripts\python.exe -m affective_metacontrol.mvp --alarm-demo --gain 0.85 --alarm-gain 8.0
```

## Interpretation limits

- Four hand-written scenarios are a demonstration, not a powered behavioral study.
- Candidate probabilities are normalized mean token likelihoods within each action pair.
- The alarm direction is calibrated from language contrasts, so cross-task and cross-model validation is required.
- A high axis gain makes the effect visible but must be included in dose-response and coherence analyses.
- The result demonstrates a causal internal intervention, not subjective emotion.
