"""
pipeline.py
═══════════════════════════════════════════════════════════════
MODE / R3 pipeline con:
  • ε dinámico
  • τ semidinamico
  • δ vectorial por métrica y régimen
  • SampEn modularizada en sampen_library.py

Diseño de esta revisión:
  • SampEn se calcula desde un módulo externo, versionado y validado.
  • La compatibilidad SampEn se conserva como diagnóstico, no como empuje del score.
  • R3 mantiene lectura vectorial (R3_min, R3_std, R3_dominant) para no ocultar
    colapsos locales detrás de un promedio único.
  • Embeddings degenerados excluyen SampEn del score de forma explícita.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from typing import Dict, Optional, Tuple
import numpy as np
from scipy.spatial.distance import cdist

# ── SampEn modular ────────────────────────────────────────────
try:
    import sampen_library as sampen_mod
except Exception:
    sampen_mod = None


# ══════════════════════════════════════════════════════════════
# 1. EMBEDDING DE TAKENS
# ══════════════════════════════════════════════════════════════
def embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Reconstrucción del espacio de fase (Takens, 1981)."""
    N = len(x)
    n = N - (m - 1) * tau
    if n <= 0:
        raise ValueError(
            f"Serie demasiado corta para m={m}, τ={tau}. "
            f"Necesitás al menos {(m - 1) * tau + 1} puntos."
        )
    return np.column_stack([x[i * tau:i * tau + n] for i in range(m)])


# ══════════════════════════════════════════════════════════════
# 2. H1 — ε DINÁMICO
# ══════════════════════════════════════════════════════════════
class DynamicEpsilon:
    def __init__(self, k_neighbors: int = 5, scale: float = 0.5):
        self.k = int(k_neighbors)
        self.scale = float(scale)

    def compute_series(self, Y: np.ndarray) -> np.ndarray:
        dists = cdist(Y, Y)
        np.fill_diagonal(dists, np.inf)
        knn = np.sort(dists, axis=1)[:, :self.k]
        return self.scale * np.mean(knn, axis=1)

    def scalar(self, Y: np.ndarray) -> float:
        return float(np.median(self.compute_series(Y)))


# ══════════════════════════════════════════════════════════════
# 3. H2 — τ SEMIDINAMICO
# ══════════════════════════════════════════════════════════════
class SemidynamicTau:
    def __init__(self, max_lag: int = 50, bins: int = 16):
        self.max_lag = int(max_lag)
        self.bins = int(bins)
        self._cache: Dict[str, int] = {}

    def _ami(self, x: np.ndarray, lag: int) -> float:
        x1 = x[:-lag]
        x2 = x[lag:]
        h2d, _, _ = np.histogram2d(x1, x2, bins=self.bins)
        pxy = h2d / (h2d.sum() + 1e-12)
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)
        denom = px * py
        mask = (pxy > 0) & (denom > 0)
        return float(np.sum(pxy[mask] * np.log2(pxy[mask] / denom[mask])))

    def compute(self, x: np.ndarray, regime: str = "unknown") -> int:
        if regime in self._cache:
            return self._cache[regime]

        max_l = max(2, min(self.max_lag, max(3, len(x) // 4)))
        ami_arr = np.array([self._ami(x, lag) for lag in range(1, max_l)])
        tau = 1

        for i in range(1, len(ami_arr) - 1):
            if ami_arr[i] < ami_arr[i - 1] and ami_arr[i] < ami_arr[i + 1]:
                tau = i + 1
                break
        else:
            threshold = ami_arr[0] / np.e if len(ami_arr) else 0.0
            for i, v in enumerate(ami_arr):
                if v < threshold:
                    tau = i + 1
                    break

        self._cache[regime] = int(max(1, tau))
        return self._cache[regime]


# ══════════════════════════════════════════════════════════════
# 4. SAMPEN — ADAPTADOR EXTERNO
# ══════════════════════════════════════════════════════════════
class SampEnBridge:
    """Fuente única de configuración/cálculo para SampEn."""

    FALLBACK_CONFIG = {
        "stable": {"m": 2, "r_ratio": 0.12, "mu": 0.05, "sigma": 0.03},
        "weakly_chaotic": {"m": 2, "r_ratio": 0.20, "mu": 0.57, "sigma": 0.12},
        "chaotic": {"m": 2, "r_ratio": 0.25, "mu": 0.53, "sigma": 0.10},
        "hyperchaotic": {"m": 2, "r_ratio": 0.30, "mu": 1.51, "sigma": 0.30},
        "noisy": {"m": 2, "r_ratio": 0.35, "mu": 1.65, "sigma": 0.35},
    }

    @classmethod
    def get_config(cls, regime: Optional[str]) -> dict:
        regime = regime or "weakly_chaotic"
        if sampen_mod is not None and hasattr(sampen_mod, "SAMPEN_CONFIG"):
            cfg = getattr(sampen_mod, "SAMPEN_CONFIG")
            return dict(cfg.get(regime, cfg.get("weakly_chaotic", cls.FALLBACK_CONFIG["weakly_chaotic"])))
        return dict(cls.FALLBACK_CONFIG.get(regime, cls.FALLBACK_CONFIG["weakly_chaotic"]))

    @classmethod
    def compute(cls, x: np.ndarray, regime: Optional[str], tau: int) -> Tuple[float, dict]:
        cfg = cls.get_config(regime)
        if sampen_mod is not None and hasattr(sampen_mod, "compute"):
            value = float(sampen_mod.compute(x, m=int(cfg["m"]), r_ratio=float(cfg["r_ratio"]), tau=int(tau)))
            meta = {
                "version": getattr(sampen_mod, "SAMPEN_VERSION", "external"),
                "source": getattr(sampen_mod, "SAMPEN_SOURCE", "external"),
            }
        else:
            value = float(Metrics._sample_entropy_fallback(x, m=int(cfg["m"]), r_ratio=float(cfg["r_ratio"]), tau=int(tau)))
            meta = {"version": "fallback", "source": "internal_fallback"}
        meta.update({
            "m": int(cfg["m"]),
            "r_ratio": float(cfg["r_ratio"]),
            "mu": float(cfg.get("mu", np.nan)),
            "sigma": float(cfg.get("sigma", np.nan)),
            "regime": str(regime or "weakly_chaotic"),
        })
        return value, meta

    @classmethod
    def compatibility_weight(cls, sampen_val: float, regime: Optional[str]) -> float:
        regime = regime or "weakly_chaotic"
        if sampen_mod is not None and hasattr(sampen_mod, "compatibility_weight"):
            return float(sampen_mod.compatibility_weight(float(sampen_val), regime))
        cfg = cls.get_config(regime)
        sigma = max(float(cfg.get("sigma", 0.12)), 1e-12)
        mu = float(cfg.get("mu", 0.57))
        return float(np.exp(-0.5 * ((float(sampen_val) - mu) / sigma) ** 2))


# ══════════════════════════════════════════════════════════════
# 5. MÉTRICAS DINÁMICAS
# ══════════════════════════════════════════════════════════════
class Metrics:
    @staticmethod
    def lyapunov(x: np.ndarray, tau: int, m: int = 3,
                 min_tsep: Optional[int] = None, max_iter: int = 300) -> float:
        Y = embed(x, m, tau)
        N = len(Y)
        if min_tsep is None:
            min_tsep = max(1, int(0.1 * N))

        dists = cdist(Y, Y)
        np.fill_diagonal(dists, np.inf)
        divergences = []

        idx = np.arange(N)
        for i in range(N):
            mask = np.abs(idx - i) > min_tsep
            d = dists[i].copy()
            d[~mask] = np.inf
            j = int(np.argmin(d))
            if np.isinf(d[j]) or d[j] == 0:
                continue

            steps = min(max_iter, N - max(i, j) - 1)
            if steps < 3:
                continue

            d_series = np.array([
                np.linalg.norm(Y[min(i + k, N - 1)] - Y[min(j + k, N - 1)])
                for k in range(steps)
            ])
            pos = d_series > 0
            if pos.sum() < 3:
                continue

            t = np.where(pos)[0]
            log_d = np.log(d_series[pos] / d[j])
            if len(t) > 1:
                divergences.append(np.polyfit(t, log_d, 1)[0])

        return float(np.median(divergences)) if divergences else np.nan

    @staticmethod
    def correlation_dimension(Y: np.ndarray, n_r: int = 20) -> float:
        N = len(Y)
        if N > 1500:
            rng = np.random.default_rng(42)
            Y = Y[rng.choice(N, 1500, replace=False)]
            N = 1500

        dists = cdist(Y, Y)
        flat = dists[np.triu_indices(N, k=1)]
        flat = flat[flat > 0]
        if len(flat) == 0:
            return np.nan

        r_min = np.percentile(flat, 5)
        r_max = np.percentile(flat, 45)
        if r_min >= r_max:
            return np.nan

        r_vals = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
        C_r = np.array([np.mean(flat < r) for r in r_vals])
        valid = (C_r > 0.01) & (C_r < 0.99)
        if valid.sum() < 4:
            return np.nan
        return float(np.polyfit(np.log(r_vals[valid]), np.log(C_r[valid]), 1)[0])

    @staticmethod
    def lempel_ziv(x: np.ndarray, Y: Optional[np.ndarray] = None) -> float:
        if Y is not None and Y.shape[0] > 10:
            Yc = Y - Y.mean(axis=0)
            cov = np.cov(Yc.T)
            if np.ndim(cov) == 0:
                seq = Yc[:, 0]
            else:
                eigvals, eigvecs = np.linalg.eigh(cov)
                seq = Yc @ eigvecs[:, -1]
        else:
            seq = x

        binary = "".join("1" if v > np.median(seq) else "0" for v in seq)
        n = len(binary)
        if n < 2:
            return np.nan

        c, l, i, k, k_max = 1, 1, 0, 1, 1
        stop = False
        while not stop:
            if i + k <= n and l + k <= n and binary[i + k - 1] == binary[l + k - 1]:
                k += 1
                if l + k > n:
                    c += 1
                    stop = True
            else:
                k_max = max(k, k_max)
                i += 1
                if i == l:
                    c += 1
                    l += k_max
                    stop = l + 1 > n
                    i, k, k_max = 0, 1, 1
                else:
                    k = 1
        norm = n / (np.log2(n) + 1e-10)
        return float(np.clip(c / norm, 0, 2))

    @staticmethod
    def transfer_entropy(x: np.ndarray, tau: int, bins: int = 8) -> float:
        n = len(x) - tau
        if n < 20:
            return np.nan
        X, Xf = x[:n], x[tau:n + tau]
        h2d, _, _ = np.histogram2d(X, Xf, bins=bins)
        pxy = h2d / (h2d.sum() + 1e-12)
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        te = 0.0
        for i in range(bins):
            for j in range(bins):
                if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    te += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
        return float(max(0.0, te))

    @staticmethod
    def _sample_entropy_fallback(x: np.ndarray, m: int = 2, r_ratio: float = 0.2, tau: int = 1) -> float:
        try:
            x = np.asarray(x, dtype=np.float64)
            N = len(x)
            std_x = np.std(x, ddof=1)
            if std_x < 1e-12:
                return np.nan
            r = r_ratio * std_x
            if N < (m + 1) * tau:
                return np.nan

            def _phi(m_val: int) -> float:
                n_t = N - (m_val - 1) * tau
                if n_t < 2:
                    return np.nan
                templates = np.array([x[j:j + m_val * tau:tau] for j in range(n_t)])
                dists = np.max(np.abs(templates[:, None, :] - templates[None, :, :]), axis=2)
                counts = np.sum(dists <= r, axis=1) - 1
                mean_c = np.mean(counts)
                return np.log(mean_c) if mean_c > 0 else -np.inf

            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)
            if np.isinf(phi_m) or np.isinf(phi_m1):
                return np.nan
            return float(phi_m - phi_m1)
        except Exception:
            return np.nan

    @classmethod
    def compute_all(cls, x: np.ndarray, tau: int, m: int = 3,
                    regime: Optional[str] = None) -> Tuple[dict, dict]:
        Y = embed(x, m, tau)
        sampen_val, sampen_meta = SampEnBridge.compute(x, regime=regime, tau=tau)
        metrics = {
            "lambda": cls.lyapunov(x, tau, m),
            "D2": cls.correlation_dimension(Y),
            "LZ": cls.lempel_ziv(x, Y),
            "TE": cls.transfer_entropy(x, tau),
            "SampEn": sampen_val,
        }
        return metrics, sampen_meta


# ══════════════════════════════════════════════════════════════
# 6. DETECTOR DE RÉGIMEN
# ══════════════════════════════════════════════════════════════
class RegimeDetector:
    DESCRIPTIONS = {
        "stable": "Estable / Periódico",
        "weakly_chaotic": "Caos débil / Cuasiperiódico",
        "chaotic": "Caótico",
        "hyperchaotic": "Hipercaótico / Estructurado",
        "noisy": "Ruido / Sin estructura dinámica",
    }

    def classify(self, lam: float, lz: Optional[float] = None, d2: Optional[float] = None) -> str:
        lam = float(lam) if lam is not None and not np.isnan(lam) else np.nan
        lz = float(lz) if lz is not None and not np.isnan(lz) else np.nan
        d2 = float(d2) if d2 is not None and not np.isnan(d2) else np.nan

        if not np.isnan(lam) and lam < 0:
            return "stable"

        if not np.isnan(d2) and not np.isnan(lz):
            if lz > 0.98 and d2 > 2.30:
                return "noisy"
            if lz > 0.94 and d2 > 2.25:
                return "hyperchaotic"
            if lz > 0.55 and d2 > 1.60:
                return "chaotic"
            if 1.45 <= d2 < 2.00 and not np.isnan(lam):
                if lam >= 0.008:
                    return "chaotic"
                if lam >= 0.002:
                    return "weakly_chaotic"
            if lz < 0.20 and d2 < 1.20 and (np.isnan(lam) or lam < 0.002):
                return "stable"
            if 1.20 <= d2 < 1.45 and not np.isnan(lam) and lam >= 0.002:
                return "weakly_chaotic"
            if lz < 0.35 and d2 < 1.35:
                return "stable"
            return "weakly_chaotic"

        if not np.isnan(lz):
            if lz < 0.20:
                return "stable"
            if lz < 0.55:
                return "weakly_chaotic"
            if lz < 0.94:
                return "chaotic"
            return "hyperchaotic"

        if not np.isnan(lam):
            if lam < 0.002:
                return "stable"
            if lam < 0.008:
                return "weakly_chaotic"
            if lam < 0.50:
                return "chaotic"
            return "hyperchaotic"

        return "weakly_chaotic"


# ══════════════════════════════════════════════════════════════
# 7. BIBLIOTECA δ
# ══════════════════════════════════════════════════════════════
class DeltaLibrary:
    TABLE = {
        "stable": 0.06,
        "weakly_chaotic": 0.07,
        "chaotic": 0.08,
        "hyperchaotic": 0.15,
        "noisy": 0.20,
    }

    def get(self, regime: str) -> dict:
        try:
            from r3_delta_library import DELTA_LIBRARY
            if regime in DELTA_LIBRARY:
                return dict(DELTA_LIBRARY[regime])
        except Exception:
            pass
        scalar = float(self.TABLE.get(regime, 0.10))
        return {k: scalar for k in ("lambda", "D2", "LZ", "TE", "SampEn")}


# ══════════════════════════════════════════════════════════════
# 8. H3 — R3 DESCRIPTOR
# ══════════════════════════════════════════════════════════════
class R3Descriptor:
    COHERENCE_THRESHOLD = 0.60

    def __init__(self):
        self.regime_detector = RegimeDetector()
        self.delta_lib = DeltaLibrary()

    def _gradients(self, x: np.ndarray, tau: int, m: int) -> Tuple[dict, dict, dict]:
        base, base_sampen_meta = Metrics.compute_all(x, tau, m)
        regime = self.regime_detector.classify(base.get("lambda", np.nan), base.get("LZ", np.nan), base.get("D2", np.nan))

        # Recalcular SampEn bajo configuración del régimen detectado.
        base, base_sampen_meta = Metrics.compute_all(x, tau, m, regime=regime)
        base["_regime"] = regime

        tau_p = max(1, tau + 1)
        tau_m = max(1, tau - 1)

        mp, _ = Metrics.compute_all(x, tau_p, m, regime=regime)
        mm, _ = Metrics.compute_all(x, tau_m, m, regime=regime) if tau_m != tau else (base, base_sampen_meta)

        grads = {}
        for k in ("lambda", "D2", "LZ", "TE", "SampEn"):
            v0 = base.get(k, np.nan)
            vp = mp.get(k, np.nan)
            vm = mm.get(k, np.nan)
            try:
                values = np.array([v0, vp, vm], dtype=float)
            except Exception:
                values = np.array([np.nan, np.nan, np.nan], dtype=float)
            if np.any(np.isnan(values)):
                grads[k] = np.nan
            else:
                denom = np.sqrt(np.sum(values ** 2) + 1e-12) / np.sqrt(3)
                grads[k] = float(abs(vp - vm) / denom) if denom > 1e-10 else float(abs(vp - vm))
        return grads, base, base_sampen_meta

    def score(self, x: np.ndarray, tau: int, m: int = 3, epsilon_scalar: Optional[float] = None) -> dict:
        grads, metrics, sampen_meta = self._gradients(x, tau, m)
        regime = metrics.pop("_regime", "weakly_chaotic")
        delta_map = self.delta_lib.get(regime)

        stability_map = {}
        stability_weights = []
        valid_n = 0
        sampen_reliable = not (epsilon_scalar is not None and float(epsilon_scalar) <= 1e-8)

        for k, g in grads.items():
            if np.isnan(g):
                continue

            delta_k = float(delta_map.get(k, 0.10))
            if k == "SampEn" and not sampen_reliable:
                stability_map[k] = {
                    "gradient": float(g),
                    "stable": False,
                    "delta": delta_k,
                    "weight": np.nan,
                    "w_compat": np.nan,
                    "used_in_score": False,
                    "note": "SampEn excluida por embedding degenerado (ε≈0).",
                }
                continue

            valid_n += 1
            is_stable = float(g) < delta_k
            w_stab = max(0.0, 1.0 - (float(g) / delta_k)) if delta_k > 0 else 0.0
            w_compat = 1.0
            if k == "SampEn":
                w_compat = SampEnBridge.compatibility_weight(metrics.get("SampEn", np.nan), regime)

            stability_weights.append(w_stab)
            stability_map[k] = {
                "gradient": float(g),
                "stable": bool(is_stable),
                "delta": delta_k,
                "weight": float(w_stab),
                "w_compat": float(w_compat),
                "used_in_score": True,
            }

        if valid_n < 2 or not stability_weights:
            r3_score = np.nan
            r3_std = np.nan
            r3_min = np.nan
            r3_dominant = "insuficiente"
        else:
            weights = np.array(stability_weights, dtype=float)
            r3_score = float(np.mean(weights))
            r3_std = float(np.std(weights))
            r3_min = float(np.min(weights))
            score_keys = [k for k, v in stability_map.items() if bool(v.get("used_in_score", False))]
            r3_dominant = min(score_keys, key=lambda key: stability_map[key]["weight"]) if score_keys else "insuficiente"

        coherent = False if np.isnan(r3_score) else bool(
            r3_score >= self.COHERENCE_THRESHOLD and
            r3_min >= self.COHERENCE_THRESHOLD / 2 and
            regime != "noisy"
        )

        return {
            "R3_score": r3_score,
            "R3_std": r3_std,
            "R3_min": r3_min,
            "R3_dominant": r3_dominant,
            "R3_vector": {
                k: (round(float(v["weight"]), 8) if np.isfinite(v.get("weight", np.nan)) else np.nan)
                for k, v in stability_map.items()
            },
            "coherent": coherent,
            "regime": regime,
            "regime_desc": RegimeDetector.DESCRIPTIONS.get(regime, regime),
            "delta": delta_map,
            "delta_mean": float(np.mean(list(delta_map.values()))) if delta_map else np.nan,
            "metrics": metrics,
            "gradients": grads,
            "stability_map": stability_map,
            "n_valid": valid_n,
            "sampen_meta": {
                **sampen_meta,
                "compat_weight": float(SampEnBridge.compatibility_weight(metrics.get("SampEn", np.nan), regime)) if not np.isnan(metrics.get("SampEn", np.nan)) else np.nan,
                "used_in_score": bool(sampen_reliable),
            },
        }


# ══════════════════════════════════════════════════════════════
# 9. PIPELINE INTEGRADO
# ══════════════════════════════════════════════════════════════
class AttractorPipeline:
    def __init__(self, m: int = 3, max_tau: int = 50, verbose: bool = True):
        self.m = int(m)
        self.verbose = bool(verbose)
        self._eps = DynamicEpsilon()
        self._tau = SemidynamicTau(max_lag=max_tau)
        self._r3 = R3Descriptor()
        self.results: Dict[str, dict] = {}

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def run(self, x: np.ndarray, label: str = "serie") -> dict:
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            x = x.ravel()
        if len(x) < max(50, self.m * 4):
            raise ValueError("Serie demasiado corta para el pipeline.")

        x = (x - x.mean()) / (x.std() + 1e-12)
        self._tau._cache.clear()

        self._log(f"\n{'═' * 52}")
        self._log(f" PIPELINE · {label} · N={len(x)}")
        self._log(f"{'═' * 52}")

        tau0 = self._tau.compute(x, regime="unknown")
        self._log(f" τ₀ (AMI, sin régimen): {tau0}")

        metrics0, sampen0 = Metrics.compute_all(x, tau0, self.m)
        regime0 = RegimeDetector().classify(metrics0["lambda"], metrics0["LZ"], metrics0["D2"])
        self._log(f" Régimen detectado: {regime0}")

        tau = self._tau.compute(x, regime=regime0)
        self._log(f" τ refinado (régimen): {tau}" if tau != tau0 else f" τ confirmado: {tau}")

        Y = embed(x, self.m, tau)
        eps_series = self._eps.compute_series(Y)
        eps_scalar = float(np.median(eps_series))
        self._log(f" Embedding shape: {Y.shape}")
        self._log(f" ε (mediana): {eps_scalar:.5f}")

        metrics, sampen_meta = Metrics.compute_all(x, tau, self.m, regime=regime0)
        self._log("\n Métricas:")
        self._log(f"  λ  (Lyapunov)  = {metrics['lambda']}")
        self._log(f"  D₂ (Corr. dim) = {metrics['D2']}")
        self._log(f"  LZ (Compl.)    = {metrics['LZ']}")
        self._log(f"  TE (Trans.ent) = {metrics['TE']}")
        self._log(f"  SE (Muestra)   = {metrics['SampEn']}")

        r3 = self._r3.score(x, tau, self.m, epsilon_scalar=eps_scalar)
        self._log("\n R³ descriptor:")
        self._log(f"  Score      = {r3['R3_score']}")
        self._log(f"  R3_min     = {r3['R3_min']}")
        self._log(f"  Coherente  = {r3['coherent']}")
        self._log(f"  Régimen    = {r3['regime_desc']}")
        self._log(f"  δ activo   = {r3['delta']}")
        self._log(f"  SampEn cfg = m={r3['sampen_meta'].get('m')} r={r3['sampen_meta'].get('r_ratio')}")
        for k, v in r3["stability_map"].items():
            sym = "✔" if v.get("stable", False) else "✘"
            self._log(
                f"  {sym} {k:<8} grad={v.get('gradient')} weight={v.get('weight')} δ={v.get('delta')}"
            )
        self._log(f"{'═' * 52}\n")

        result = {
            "label": label,
            "x_normalized": x,
            "tau": tau,
            "tau_initial": tau0,
            "epsilon": eps_scalar,
            "epsilon_series": eps_series,
            "embedding": Y,
            "metrics": metrics,
            "sampen_meta": {**sampen_meta, **r3.get("sampen_meta", {})},
            "regime": r3["regime"],
            "regime_desc": r3["regime_desc"],
            "R3": r3,
        }
        self.results[label] = result
        return result
