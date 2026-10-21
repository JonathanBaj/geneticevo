"""Portable records and atomic, trusted generation-boundary checkpoints."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import tempfile
import functools
import hashlib
import inspect
from typing import Any

import numpy as np


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return repr(value)


def atomic_write(path, data: bytes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, default=json_default, allow_nan=False).encode("utf-8"))


def save_checkpoint(path, state):
    atomic_write(path, pickle.dumps({"schema": 1, "state": state}, protocol=5))


def load_checkpoint(path):
    """Only load checkpoints you trust: pickle can execute code."""
    with open(path, "rb") as file:
        payload = pickle.load(file)
    if payload.get("schema") != 1:
        raise ValueError("Unsupported checkpoint schema")
    return payload["state"]


def software_metadata():
    versions = {}
    for name in ("geneticevo", "numpy", "gymnasium", "torch", "stable-baselines3", "pettingzoo"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"versions": versions, "python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count()}


def describe(value):
    """Stable JSON description of experiment inputs for provenance/resume checks."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {"shape": value.shape, "dtype": str(value.dtype), "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {str(key): describe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [describe(item) for item in value]
    if isinstance(value, functools.partial):
        return {"partial": describe(value.func), "args": describe(value.args), "kwargs": describe(value.keywords)}
    identity = f"{getattr(value, '__module__', type(value).__module__)}.{getattr(value, '__qualname__', type(value).__qualname__)}"
    if inspect.isfunction(value) or inspect.isclass(value) or inspect.isbuiltin(value):
        try:
            source = hashlib.sha256(inspect.getsource(value).encode()).hexdigest()
        except (OSError, TypeError):
            source = None
        return {"callable": identity, "source_sha256": source}
    return {"type": identity, "fields": {key: describe(item) for key, item in vars(value).items() if not key.startswith('_')}}


@dataclass
class Result:
    best_agent: Any
    best_fitness: float
    population: list
    archive: list
    history: list[dict]
    lineage: list[dict]
    metadata: dict
    stop_reason: str

    def evaluate(self, evaluator, scenarios, seed=0):
        """Explicit final testing; results cannot feed back into a completed run."""
        return evaluator.evaluate(self.best_agent.clone(training_state=False), scenarios, seed)


def uncertainty_summary(values):
    """Across independent training runs, not episodes; normal CI is descriptive."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        raise ValueError("At least two independent training runs are needed")
    mean = float(values.mean())
    half = float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
    return {"runs": len(values), "mean": mean, "std": float(values.std(ddof=1)),
            "normal_95_interval": [mean - half, mean + half]}

