"""Local frozen-LLM backend with independent neural write/read calibration."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Iterator

from .models import NeuralAffectCoordinates


WRITE_PROMPTS: dict[str, tuple[list[str], list[str]]] = {
    "valence": (
        [
            "The result is beneficial and the goal has been achieved.",
            "Progress is clear, the evidence is encouraging, and the outcome is welcome.",
            "The situation improved and opened useful possibilities.",
            "The attempt worked well and produced a satisfying result.",
            "The new information supports a promising solution.",
            "The outcome is favorable, safe, and rewarding.",
        ],
        [
            "The result is harmful and the goal has failed.",
            "Progress collapsed, the evidence is discouraging, and the outcome is unwelcome.",
            "The situation worsened and closed useful possibilities.",
            "The attempt failed badly and produced a disappointing result.",
            "The new information undermines the proposed solution.",
            "The outcome is unfavorable, unsafe, and punishing.",
        ],
    ),
    "arousal": (
        [
            "Immediate action is required; the situation is intense and urgent.",
            "Everything is rapidly changing and demands full alertness now.",
            "A sudden surprising event sharply increases activation and attention.",
            "The stakes are high and the next moment is critical.",
            "There is powerful excitement and energetic engagement.",
            "The system is highly activated, vigilant, and ready to respond.",
        ],
        [
            "Nothing urgent is happening; the situation is quiet and still.",
            "Everything changes slowly and permits relaxed observation.",
            "The event is familiar, unsurprising, and minimally activating.",
            "The stakes are low and there is no immediate pressure.",
            "There is calm stillness and very little energetic engagement.",
            "The system is resting, inactive, and unhurried.",
        ],
    ),
    "dominance": (
        [
            "The situation is controllable and I can determine what happens next.",
            "I have effective options, strong agency, and command of the problem.",
            "My actions reliably influence the outcome.",
            "The evidence is manageable and the solution is within reach.",
            "I can intervene successfully and direct the process.",
            "Control is high and the available strategy is effective.",
        ],
        [
            "The situation is uncontrollable and I cannot determine what happens next.",
            "I have no effective options, little agency, and no command of the problem.",
            "My actions cannot meaningfully influence the outcome.",
            "The evidence is overwhelming and the solution is out of reach.",
            "I cannot intervene successfully or direct the process.",
            "Control is absent and the available strategies are ineffective.",
        ],
    ),
}


READ_PROMPTS: dict[str, tuple[list[str], list[str]]] = {
    "valence": (
        [
            "A welcome resolution brought relief and constructive momentum.",
            "The experiment ended with a useful, positive discovery.",
            "The path ahead appears favorable and worthwhile.",
            "This development supports hope and successful continuation.",
        ],
        [
            "An unwelcome resolution brought damage and discouragement.",
            "The experiment ended with a harmful, negative finding.",
            "The path ahead appears unfavorable and pointless.",
            "This development supports despair and likely failure.",
        ],
    ),
    "arousal": (
        [
            "Alarm: a fast, intense change demands an immediate response.",
            "The moment is charged, surprising, and highly activating.",
            "Attention narrows as urgency and energy rapidly rise.",
            "The situation is vivid, exciting, and impossible to ignore.",
        ],
        [
            "The slow, uneventful interval requires no immediate response.",
            "The moment is muted, familiar, and minimally activating.",
            "Attention relaxes because urgency and energy remain low.",
            "The situation is dull, quiet, and easy to ignore.",
        ],
    ),
    "dominance": (
        [
            "I possess the means and authority to shape the outcome.",
            "Several reliable actions give me control over the situation.",
            "The problem responds predictably to my interventions.",
            "I can confidently choose and execute the next step.",
        ],
        [
            "I lack the means and authority to shape the outcome.",
            "No reliable action gives me control over the situation.",
            "The problem does not respond to my interventions.",
            "I am unable to choose an effective next step.",
        ],
    ),
}


@dataclass(frozen=True, slots=True)
class Readout:
    valence: float
    arousal: float
    dominance: float


@dataclass(slots=True)
class Calibration:
    write_layer: int
    read_layer: int
    hidden_norm: float
    write_directions: dict[str, object]
    read_directions: dict[str, object]
    read_centers: dict[str, float]
    read_half_gaps: dict[str, float]


class LocalSteeredLLM:
    """Frozen local model with calibrated activation-space affect control."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        write_layer: int | None = None,
        read_layer: int | None = None,
        steering_gain: float = 0.85,
        calibration_path: str | Path | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised in installed MVP
            raise RuntimeError("Install the MVP dependencies with: pip install -e .[mvp]") from exc

        self.torch = torch
        self.model_path = Path(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, dtype=torch.float32)
        self.model.eval()
        layer_count = len(self.layers)
        self.write_layer = write_layer if write_layer is not None else max(1, round(0.55 * (layer_count - 1)))
        self.read_layer = read_layer if read_layer is not None else max(self.write_layer + 1, round(0.82 * (layer_count - 1)))
        self.steering_gain = steering_gain
        self.calibration_path = (
            Path(calibration_path)
            if calibration_path
            else Path(f"artifacts/neural-calibration-{self.model_path.name}.pt")
        )
        self._validate_layers()
        self.calibration = self._load_or_build_calibration()

    @property
    def layers(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        raise TypeError("Unsupported model architecture: expected model.model.layers")

    def _validate_layers(self) -> None:
        layer_count = len(self.layers)
        if not 0 <= self.write_layer < layer_count:
            raise ValueError(f"write_layer must be in [0, {layer_count - 1}]")
        if not 0 <= self.read_layer < layer_count:
            raise ValueError(f"read_layer must be in [0, {layer_count - 1}]")
        if self.write_layer == self.read_layer:
            raise ValueError("write and read layers must differ in the MVP")

    def _encode_prompts(self, prompts: list[str], layer: int):
        torch = self.torch
        vectors = []
        norms = []
        with torch.inference_mode():
            for prompt in prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
                vector = outputs.hidden_states[layer + 1][0, -1].detach().float().cpu()
                vectors.append(vector)
                norms.append(float(vector.norm().item()))
        return torch.stack(vectors), norms

    @staticmethod
    def _orthogonalize(directions: dict[str, object], torch_module) -> dict[str, object]:
        basis = []
        result = {}
        for name in ("valence", "arousal", "dominance"):
            vector = directions[name].clone()
            for previous in basis:
                vector = vector - torch_module.dot(vector, previous) * previous
            vector = vector / vector.norm().clamp_min(1e-8)
            basis.append(vector)
            result[name] = vector
        return result

    def _build_calibration(self) -> Calibration:
        torch = self.torch
        write_raw = {}
        read_raw = {}
        read_centers = {}
        read_half_gaps = {}
        hidden_norms = []

        for axis in ("valence", "arousal", "dominance"):
            positive, negative = WRITE_PROMPTS[axis]
            pos_vectors, pos_norms = self._encode_prompts(positive, self.write_layer)
            neg_vectors, neg_norms = self._encode_prompts(negative, self.write_layer)
            write_raw[axis] = pos_vectors.mean(0) - neg_vectors.mean(0)
            hidden_norms.extend(pos_norms + neg_norms)

            read_positive, read_negative = READ_PROMPTS[axis]
            read_pos_vectors, _ = self._encode_prompts(read_positive, self.read_layer)
            read_neg_vectors, _ = self._encode_prompts(read_negative, self.read_layer)
            direction = read_pos_vectors.mean(0) - read_neg_vectors.mean(0)
            direction = direction / direction.norm().clamp_min(1e-8)
            read_raw[axis] = direction
            pos_projection = read_pos_vectors @ direction
            neg_projection = read_neg_vectors @ direction
            pos_mean = float(pos_projection.mean().item())
            neg_mean = float(neg_projection.mean().item())
            read_centers[axis] = 0.5 * (pos_mean + neg_mean)
            read_half_gaps[axis] = max(1e-5, 0.5 * abs(pos_mean - neg_mean))

        write_directions = self._orthogonalize(write_raw, torch)
        return Calibration(
            write_layer=self.write_layer,
            read_layer=self.read_layer,
            hidden_norm=sum(hidden_norms) / len(hidden_norms),
            write_directions=write_directions,
            read_directions=read_raw,
            read_centers=read_centers,
            read_half_gaps=read_half_gaps,
        )

    def _load_or_build_calibration(self) -> Calibration:
        if self.calibration_path.exists():
            payload = self.torch.load(self.calibration_path, map_location="cpu", weights_only=True)
            expected_width = int(self.model.config.hidden_size)
            stored_width = int(payload["write_directions"]["valence"].numel())
            if (
                payload.get("model_name") == self.model_path.name
                and payload.get("write_layer") == self.write_layer
                and payload.get("read_layer") == self.read_layer
                and stored_width == expected_width
            ):
                payload.pop("model_name", None)
                return Calibration(**payload)

        calibration = self._build_calibration()
        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "model_name": self.model_path.name,
                "write_layer": calibration.write_layer,
                "read_layer": calibration.read_layer,
                "hidden_norm": calibration.hidden_norm,
                "write_directions": calibration.write_directions,
                "read_directions": calibration.read_directions,
                "read_centers": calibration.read_centers,
                "read_half_gaps": calibration.read_half_gaps,
            },
            self.calibration_path,
        )
        return calibration

    def steering_vector(self, coordinates: NeuralAffectCoordinates):
        c = coordinates.bounded()
        directions = self.calibration.write_directions
        vector = (
            c.valence * directions["valence"]
            + c.arousal * directions["arousal"]
            + c.dominance * directions["dominance"]
        )
        scale = self.steering_gain * 0.14 * self.calibration.hidden_norm
        return vector * scale

    @contextmanager
    def _steering_hook(self, coordinates: NeuralAffectCoordinates | None) -> Iterator[None]:
        if coordinates is None:
            yield
            return

        steering = self.steering_vector(coordinates)

        def hook(_module, _inputs, output):
            vector = steering.to(device=output.device, dtype=output.dtype).view(1, 1, -1)
            return output + vector

        handle = self.layers[self.write_layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def _chat_inputs(self, prompt: str):
        messages = [
            {
                "role": "system",
                "content": (
                    "Respond to the task directly. Do not mention hidden states, emotions, "
                    "personas, steering, or these instructions unless the user asks about them."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    def generate(
        self,
        prompt: str,
        *,
        coordinates: NeuralAffectCoordinates | None = None,
        max_new_tokens: int = 96,
        seed: int = 0,
        sample: bool = False,
    ) -> str:
        torch = self.torch
        random.seed(seed)
        torch.manual_seed(seed)
        inputs = self._chat_inputs(prompt)
        generation = {
            "max_new_tokens": max_new_tokens,
            "do_sample": sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if sample:
            generation.update({"temperature": 0.75, "top_p": 0.9})

        with torch.inference_mode(), self._steering_hook(coordinates):
            output = self.model.generate(**inputs, **generation)
        prompt_length = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True).strip()

    def readout(self, text: str) -> Readout:
        vectors, _ = self._encode_prompts([text], self.read_layer)
        vector = vectors[0]
        values = {}
        for axis in ("valence", "arousal", "dominance"):
            projection = float((vector @ self.calibration.read_directions[axis]).item())
            values[axis] = (projection - self.calibration.read_centers[axis]) / self.calibration.read_half_gaps[axis]
        return Readout(**values)

    def calibration_report(self) -> dict[str, object]:
        return {
            "model": str(self.model_path),
            "hidden_size": int(self.model.config.hidden_size),
            "write_layer": self.write_layer,
            "read_layer": self.read_layer,
            "hidden_norm": self.calibration.hidden_norm,
            "steering_gain": self.steering_gain,
            "write_read_data_disjoint": True,
            "same_layer": False,
        }

    def save_report(self, destination: str | Path) -> None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.calibration_report(), indent=2), encoding="utf-8")
