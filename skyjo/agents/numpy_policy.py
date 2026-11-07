"""The policy network of a trained model, computed with NumPy only (no torch, no
Stable-Baselines3): used where those are not available, e.g. in the browser.

The weights file (written by scripts/export_weights.py) holds the policy branch
of a MaskablePPO MlpPolicy: hidden layers (Linear then tanh) and the action
layer (Linear), float32. Layout: MAGIC, header length (uint32 little-endian),
JSON header {"activation", "layers": [{"name", "shape", "offset"}], "source"},
then the arrays, float32 little-endian, at their offsets from the end of the header.

Like MaskablePPO, masked actions get a logit of -1e8 before the softmax; the
deterministic action is the most probable one, the sampled one is drawn with a
NumPy generator.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b"SKYJOPOL1"
MASKED_LOGIT = -1e8  # as sb3_contrib's MaskableCategorical


def save_weights(path: str | Path, layers: list[tuple[np.ndarray, np.ndarray]], source: str = "") -> None:
    """Write ``layers`` [(weight (out, in), bias (out,)), ...], the last one being the action layer."""
    entries, blobs, offset = [], [], 0
    for i, (weight, bias) in enumerate(layers):
        for kind, array in (("weight", weight), ("bias", bias)):
            data = np.ascontiguousarray(array, dtype="<f4").tobytes()
            entries.append({"name": f"{i}.{kind}", "shape": list(array.shape), "offset": offset})
            blobs.append(data)
            offset += len(data)
    header = json.dumps({"activation": "tanh", "layers": entries, "source": source}).encode()
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<I", len(header)) + header)
        for data in blobs:
            f.write(data)


def load_weights(data: bytes) -> list[tuple[np.ndarray, np.ndarray]]:
    if not data.startswith(MAGIC):
        raise ValueError("Not a policy weights file.")
    (size,) = struct.unpack_from("<I", data, len(MAGIC))
    start = len(MAGIC) + 4
    header = json.loads(data[start:start + size])
    if header["activation"] != "tanh":
        raise ValueError(f"Unknown activation {header['activation']!r}.")
    base = start + size
    arrays = {}
    for entry in header["layers"]:
        count = int(np.prod(entry["shape"]))
        arrays[entry["name"]] = np.frombuffer(data, "<f4", count, base + entry["offset"]).reshape(entry["shape"])
    return [(arrays[f"{i}.weight"], arrays[f"{i}.bias"]) for i in range(len(arrays) // 2)]


class NumpyPolicy:
    """Drop-in for the ``predict`` of a MaskablePPO model (see PPOAgent)."""

    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray]], rng: np.random.Generator | None = None) -> None:
        self.layers = layers
        self.rng = rng if rng is not None else np.random.default_rng()

    @classmethod
    def load(cls, path: str | Path, rng: np.random.Generator | None = None) -> NumpyPolicy:
        return cls(load_weights(Path(path).read_bytes()), rng)

    def logits(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        for weight, bias in self.layers[:-1]:
            x = np.tanh(x @ weight.T + bias)
        weight, bias = self.layers[-1]
        return x @ weight.T + bias

    def probabilities(self, obs: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
        logits = np.where(action_mask, self.logits(obs), np.float32(MASKED_LOGIT))
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)

    def predict(self, obs: np.ndarray, action_masks: np.ndarray, deterministic: bool = True) -> tuple[int, None]:
        probs = self.probabilities(obs, np.asarray(action_masks, dtype=bool))
        if deterministic:
            return int(np.argmax(probs)), None
        p = probs.astype(np.float64)
        return int(self.rng.choice(len(p), p=p / p.sum())), None
