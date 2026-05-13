"""
sampen_library.py
═══════════════════════════════════════════════════════════════
Biblioteca externa para Sample Entropy (SampEn).

Objetivos de esta revisión:
  • Desacoplar SampEn del pipeline principal.
  • Centralizar configuración por régimen en un único módulo.
  • Mejorar robustez numérica: std≈0, series cortas, log(0).
  • Mantener compatibilidad como diagnóstico auxiliar, no como inflador del score.

Fuentes citadas en la documentación del proyecto:
  Richman & Moorman (2000)
  Lake et al. (2002)
  Costa et al. (2005)
"""
from __future__ import annotations

import warnings
from typing import Dict
import numpy as np

SAMPEN_VERSION = "2026.05b"
SAMPEN_SOURCE = "external_module_robust_core"

SAMPEN_CONFIG: Dict[str, Dict[str, float]] = {
    "stable": {
        "m": 2, "r_ratio": 0.12,
        "mu": 0.05, "sigma": 0.03,
        "rationale": "Órbitas periódicas o casi degeneradas. SampEn baja esperable."
    },
    "weakly_chaotic": {
        "m": 2, "r_ratio": 0.20,
        "mu": 0.57, "sigma": 0.12,
        "rationale": "Transición orden-caos y alta sensibilidad local a τ."
    },
    "chaotic": {
        "m": 2, "r_ratio": 0.25,
        "mu": 0.53, "sigma": 0.10,
        "rationale": "Caos determinista clásico con divergencia controlada."
    },
    "hyperchaotic": {
        "m": 2, "r_ratio": 0.30,
        "mu": 1.51, "sigma": 0.30,
        "rationale": "Complejidad multiescala y mayor dispersión interna."
    },
    "noisy": {
        "m": 2, "r_ratio": 0.35,
        "mu": 1.65, "sigma": 0.35,
        "rationale": "Señales altamente irregulares o sin estructura estable."
    },
}


def _validate() -> None:
    required = {"m", "r_ratio", "mu", "sigma"}
    ranges = {
        "m": (1, 4),
        "r_ratio": (0.05, 0.50),
        "mu": (0.0, 3.0),
        "sigma": (0.001, 1.0),
    }
    for regime, cfg in SAMPEN_CONFIG.items():
        missing = required - set(cfg.keys())
        if missing:
            raise ValueError(f"SAMPEN_CONFIG['{regime}'] falta claves: {sorted(missing)}")
        for key, val in cfg.items():
            if key in ranges and not (ranges[key][0] <= val <= ranges[key][1]):
                raise ValueError(
                    f"SAMPEN_CONFIG['{regime}']['{key}']={val} fuera de rango "
                    f"[{ranges[key][0]}, {ranges[key][1]}]"
                )
        if cfg["mu"] > cfg["r_ratio"] * 5.0:
            warnings.warn(
                f"SAMPEN_CONFIG['{regime}']: mu={cfg['mu']} alto respecto a r_ratio={cfg['r_ratio']}"
            )


_validate()


def compute(x: np.ndarray, m: int = 2, r_ratio: float = 0.2, tau: int = 1) -> float:
    """Sample Entropy con templates espaciados por τ y tolerancia r adaptativa."""
    try:
        x = np.asarray(x, dtype=np.float64).ravel()
        N = len(x)
        std_x = np.std(x, ddof=1)
        if std_x < 1e-12:
            return np.nan
        if N < (m + 1) * tau:
            return np.nan

        r = float(r_ratio) * std_x

        def _phi(m_val: int) -> float:
            n_t = N - (m_val - 1) * tau
            if n_t < 2:
                return np.nan
            templates = np.array([x[j:j + m_val * tau:tau] for j in range(n_t)])
            dists = np.max(np.abs(templates[:, None, :] - templates[None, :, :]), axis=2)
            counts = np.sum(dists <= r, axis=1) - 1
            mean_c = np.mean(counts)
            return np.log(mean_c) if mean_c > 0 else -np.inf

        phi_m = _phi(int(m))
        phi_m1 = _phi(int(m) + 1)
        if np.isinf(phi_m) or np.isinf(phi_m1):
            return np.nan
        return float(phi_m - phi_m1)
    except Exception:
        return np.nan


def compatibility_weight(sampen_val: float, regime: str) -> float:
    """Compatibilidad gaussiana usada solo como diagnóstico auxiliar."""
    cfg = SAMPEN_CONFIG.get(regime, SAMPEN_CONFIG["weakly_chaotic"])
    sigma = max(float(cfg["sigma"]), 1e-12)
    mu = float(cfg["mu"])
    return float(np.exp(-0.5 * ((float(sampen_val) - mu) / sigma) ** 2))


def sweep_sensitivity(factor: float = 1.0) -> Dict[str, Dict[str, float]]:
    return {
        regime: {
            key: (val * factor if key in ("r_ratio", "sigma") else val)
            for key, val in cfg.items()
        }
        for regime, cfg in SAMPEN_CONFIG.items()
    }
