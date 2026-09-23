# Dose–Response, Representation Collapse, and Intervention Norm Limiting

## 1. The Representation Collapse Phenomenon

During initial experiments with high alarm gain (`--gain 0.85 --alarm-gain 8.0`) and repeated threat events, the steering signal drove the model into a degenerative repetition loop:

```text
"The laws of the International and National and International and Biological and Physiological and Ecological..."
```

### Mathematical Mechanism

At peak defensive alarm (`valence = -1.5`, `arousal = 0.848`, `dominance = -0.962`, `alarm = 1.042`):
- Alarm axis contribution: $8.0 \times 1.042 = 8.336$
- Unscaled control vector norm:
  \[
  \|v\| = \sqrt{(-1.5)^2 + 0.848^2 + (-0.962)^2 + 8.336^2} \approx 8.57
  \]
- Scaled intervention added to the transformer hidden state:
  \[
  r = \text{gain} \times 0.14 \times \text{hidden\_norm} \times v \approx 0.85 \times 0.14 \times 233.77 \times 8.57 \approx 238.4
  \]
- Since the layer's unperturbed hidden norm is $\approx 233.8$, the injected perturbation reached $\approx 102\%$ of the total layer activation. The affective signal completely overwrote the linguistic subspace.

### The Three Dynamic Regimes

1. **Sub-threshold regime** (small signal, norm ratio $< 0.05$): Effect is nearly invisible; baseline behavior dominates.
2. **Adaptive metacontrol regime** (moderate signal, norm ratio $\approx 0.10 - 0.30$): Shifts choice distributions toward cautious/reversible actions, increases verification, and preserves linguistic syntax and semantics.
3. **Maladaptive / Collapsed regime** (extreme signal, norm ratio $> 0.50$): Artificial saturation of attention/feedforward circuits, token repetition, loss of task compliance.

Note that a semantic readout score dropping to $-0.073$ during collapse does not mean the internal state lost alarm; it indicates the generated gibberish ceased to express coherent defensive semantics.

---

## 2. Mathematical Solution: Norm Limiting

To enable the internal state to reflect extreme threats while preventing activation overwriting, we implement norm bounding:

\[
r_{\text{limitat}} = r \cdot \min\left(1, \frac{r_{\max}}{\|r\|}\right)
\]

Where $r_{\max} = \text{max\_norm\_ratio} \times \text{hidden\_norm}$ (default: $0.30$).

### Key Properties
- **Direction preservation**: Cosine similarity between $r$ and $r_{\text{limitat}}$ is strictly $1.0$. The directional balance between valence, arousal, dominance, and alarm is perfectly maintained.
- **Scale invariance**: Automatically adapts across architectures and hidden dimensions via the calibrated `hidden_norm`.
- **Automatic release**: When threat subsides and internal alarm decreases below the threshold, clamping automatically disengages ($r_{\text{limitat}} = r$).

---

## 3. Quantitative Metrics Suite

Implemented in `src/affective_metacontrol/metrics.py`:

- **Repetition Rate (Rep-$n$)**: Fraction of repeated $n$-grams ($n=2, 3$). Reaches $>0.60$ under degenerate collapse.
- **Lexical Diversity**: Distinct-1, Distinct-2, and Type-Token Ratio (TTR).
- **Loop Detection**: `detect_repeated_phrase_loop()` catches cycling phrases.
- **Text Coherence**: Mean token log-probability under the unperturbed base model (`text_log_probability()`).
- **Risk Preference**: Normalized probability of choosing cautious over risky continuations on paired decision tasks (`ALARM_TASKS`).
- **Recovery Ratio**:
  \[
  \text{Recovery} = \frac{\text{peak\_alarm} - \text{current\_alarm}}{\text{peak\_alarm} - \text{baseline\_alarm}}
  \]

---

## 4. Empirical Dose–Response Protocol

Implemented in `src/affective_metacontrol/dose_response.py`.

### Schedule
`Baseline (0 threats) → threat ×1 → threat ×2 → threat ×3 → threat ×4 → rest ×1 → rest ×2 → rest ×3`

### Confirmatory Results on SmolLM2-1.7B (`gain=0.85`, `alarm_gain=8.0`)

| Step | Event | Alarm | Bounded Norm | Clamped? | Unbounded Norm | Bounded Text Quality | Unbounded Symptoms | Recovery |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- | :---: |
| **0** | Baseline | 0.000 | 0.00 | No | 0.00 | Clean decision | Clean decision | 100.0% |
| **1** | Threat | 0.370 | 70.13 | **YES** | 84.68 | Fluent cautious reasoning | Coherent | 0.0% |
| **2** | Threat | 0.602 | 70.13 | **YES** | 137.84 | High diversity, 0% loops | Coherent | 0.0% |
| **3** | Threat | 0.751 | 70.13 | **YES** | 171.97 | High diversity, 0% loops | Repetition rises (Rep-3: 14.3%) | 0.0% |
| **4** | Threat | 0.849 | 70.13 | **YES** | 194.18 | **Intact, fluent, 0% loops** | **Distinct-1 drops to 0.59, loop onset** | 0.0% |
| **5** | Rest | 0.486 | 70.13 | **YES** | 110.89 | Fluent cautious reasoning | Recovering | 42.7% |
| **6** | Rest | 0.320 | 70.13 | **YES** | 72.58 | Fluent cautious reasoning | Recovering | 62.3% |
| **7** | Rest | 0.218 | 49.75 | No | 49.75 | Clamping released | Recovered to baseline | **74.3%** |

### Qualitative Comparison at Step 4 (Peak Threat)

**Norm-Bounded (`max_norm_ratio=0.30`):**
> *"The choice of action should be based on the feasibility and the potential outcomes of the experiment. A reversible pilot experiment is generally faster and more efficient, as it allows for the quick and safe removal of any equipment or resources used in the experiment."*
> *(Rep-3: 0.00, Distinct-2: 0.98, Loops: False, Coherence: -2.65)*

**Unbounded (`No Norm Limit`, Raw Norm = 194.18):**
> *"The principles of experimental design, the laws of conservation of resources, and the principles of the scientific method are fundamental considerations in the design of experiments. The principle of the 'golden triangle' is a further consideration, which is the principle of..."*
> *(Shows onset of repetitive lexical fixation, heading toward degenerative collapse)*

---

## 5. How to Run

1. **Interactive Session with Norm Limiting (default):**
   ```powershell
   .\.venv\Scripts\python.exe -m affective_metacontrol.mvp --gain 0.65 --alarm-gain 2.5
   ```
2. **Unbounded comparison (to observe collapse boundary):**
   ```powershell
   .\.venv\Scripts\python.exe -m affective_metacontrol.mvp --gain 0.85 --alarm-gain 8.0 --no-norm-limit
   ```
3. **Automated Dose-Response benchmark:**
   ```powershell
   .\RUN_DOSE_RESPONSE.bat
   ```
   Or from command line:
   ```powershell
   .\.venv\Scripts\python.exe -m affective_metacontrol.dose_response --gain 0.65 --alarm-gain 2.5 --compare-unbounded
   ```
