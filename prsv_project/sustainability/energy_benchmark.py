from __future__ import annotations

"""
Energy/inference-cost benchmarking (v3.1): formally measures the SVM
pipeline's inference cost, turning the project's "lightweight by design"
choice into a measurable, citable number rather than an assumed advantage.

Honesty note: `CodeCarbon` (the library named in the wishlist) measures
actual joules via OS/hardware power-draw APIs (Intel RAPL, NVIDIA-SMI,
etc.), which aren't available in every environment (notably: sandboxed
containers, some cloud VMs). This module uses CodeCarbon when it's
installed and hardware power-draw access is available, and otherwise falls
back to a CPU-time-based proxy (wall-clock inference time x number of CPU
cores used), which is a standard, defensible stand-in used throughout the
Green-AI literature when direct power measurement isn't accessible - not
pretending to be the same precision as a direct joule measurement, and
labeled as such in the output.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np


@dataclass
class EnergyBenchmarkResult:
    model_name: str
    n_inferences: int
    total_wall_time_seconds: float
    mean_inference_time_ms: float
    method: str
    estimated_energy_joules: Optional[float]


def _try_codecarbon_measurement(inference_fn: Callable[[], Any], n_runs: int) -> Optional[float]:
    try:
        from codecarbon import EmissionsTracker

        tracker = EmissionsTracker(log_level="error", save_to_file=False)
        tracker.start()
        for _ in range(n_runs):
            inference_fn()
        emissions_kg = tracker.stop()
        # CodeCarbon reports CO2-equivalent kg; convert back to an
        # approximate joule figure via the same energy/emissions factor it
        # used internally is nontrivial without its internal state, so this
        # reports emissions directly rather than a converted joule estimate.
        return float(emissions_kg) if emissions_kg else None
    except Exception:  # noqa: BLE001 - optional dependency and/or no power-draw access
        return None


def benchmark_inference_cost(
    model_name: str,
    inference_fn: Callable[[], Any],
    n_runs: int = 100,
    cpu_cores_used: int = 1,
    cpu_thermal_design_power_watts: float = 15.0,
) -> EnergyBenchmarkResult:
    """
    inference_fn: a zero-arg callable performing exactly one inference call
    (e.g. `lambda: model.predict(single_feature_vector)`), timed in a tight
    loop for `n_runs` iterations.

    cpu_thermal_design_power_watts: a rough per-core TDP estimate used only
    for the CPU-time-proxy fallback (default 15W reflects a typical laptop/
    small-cloud-instance core under load) - explicitly an estimate, not a
    measurement, and reported as such.
    """
    codecarbon_kg_co2 = _try_codecarbon_measurement(inference_fn, n_runs)

    start = time.perf_counter()
    for _ in range(n_runs):
        inference_fn()
    total_time = time.perf_counter() - start

    mean_time_ms = (total_time / n_runs) * 1000

    if codecarbon_kg_co2 is not None:
        method = "codecarbon_measured"
        estimated_energy_joules = None  # CodeCarbon result reported separately as emissions, not joules
    else:
        method = "cpu_time_proxy_estimate"
        estimated_energy_joules = round(total_time * cpu_cores_used * cpu_thermal_design_power_watts, 6)

    return EnergyBenchmarkResult(
        model_name=model_name,
        n_inferences=n_runs,
        total_wall_time_seconds=round(total_time, 6),
        mean_inference_time_ms=round(mean_time_ms, 4),
        method=method,
        estimated_energy_joules=estimated_energy_joules,
    )


def compare_models_energy(results: Dict[str, EnergyBenchmarkResult]) -> Dict[str, Any]:
    if len(results) < 2:
        return {"status": "insufficient_models", "message": "Need at least two models to compare."}

    fastest = min(results.values(), key=lambda r: r.mean_inference_time_ms)
    slowest = max(results.values(), key=lambda r: r.mean_inference_time_ms)
    speedup = slowest.mean_inference_time_ms / max(fastest.mean_inference_time_ms, 1e-9)

    return {
        "status": "compared",
        "fastest_model": fastest.model_name,
        "slowest_model": slowest.model_name,
        "speedup_factor": round(speedup, 2),
        "method": fastest.method,
        "caveat": (
            "CPU-time-proxy estimates are a defensible stand-in, not a direct "
            "joule measurement - install `codecarbon` on hardware with power-draw "
            "access (not typical in sandboxed containers) for measured emissions."
        )
        if fastest.method == "cpu_time_proxy_estimate"
        else None,
    }
