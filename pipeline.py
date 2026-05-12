"""
pipeline.py
═══════════════════════════════════════════════════════════════
Atractor con ε dinámico, τ semidinamico y R³ como descriptor
de co-estabilización observacional.
REVISIÓN 2026-05b: Calibración final desde datos empíricos
δ weakly_chaotic: 0.05 → 0.07 (Rössler empírico)
RegimeDetector: umbral hyperchaotic CLZ 0.85 → 0.92
R3Descriptor: umbral coherencia 0.60 → 0.57
Lorenz demo: 2000 pasos warmup descartados
Arquitectura:
H1 — ε dinámico : adapta resolución al sistema
H2 — τ semidinamico : estabiliza reconstrucción por régimen
H3 — R³ descriptor : revela cuándo H1 + H2 lograron coherencia
δ calibrado desde literatura empírica + datos observados:
Peng et al. (1995) : HRV / fisiológico
Grassberger & Procaccia (1983) : atractores clásicos
Mantegna & Stanley (1999) : series financieras
Schreiber (2000) : transferencia de entropía
Duarte (2026) : calibración empírica Rössler/Lorenz/Logístico
═══════════════════════════════════════════════════════════════
"""
import numpy as np
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════
# 1. EMBEDDING DE TAKENS
# ═══════════════════════════════════════════
def embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Reconstrucción del espacio de fase (Takens, 1981)."""
    N = len(x)
    n = N - (m - 1) * tau
    if n <= 0:
        raise ValueError(f"Serie demasiado corta para m={m}, τ={tau}. Necesitás al menos {(m-1)*tau + 1} puntos.")
    Y = np.column_stack([x[i * tau: i * tau + n] for i in range(m)])
    return Y

# ═══════════════════════════════════════════
# 2. H1 — ε DINÁMICO
# ═══════════════════════════════════════════
class DynamicEpsilon:
    def __init__(self, k_neighbors: int = 5, scale: float = 0.5):
        self.k = k_neighbors
        self.scale = scale

    def compute_series(self, Y: np.ndarray) -> np.ndarray:
        dists = cdist(Y, Y)
        np.fill_diagonal(dists, np.inf)
        knn = np.sort(dists, axis=1)[:, :self.k]
        return self.scale * np.mean(knn, axis=1)

    def scalar(self, Y: np.ndarray) -> float:
        return float(np.median(self.compute_series(Y)))

# ═══════════════════════════════════════════
# 3. H2 — τ SEMIDINAMICO
# ═══════════════════════════════════════════
class SemidynamicTau:
    def __init__(self, max_lag: int = 50, bins: int = 16):
        self.max_lag = max_lag
        self.bins = bins
        self._cache: dict = {}

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

    def compute(self, x: np.ndarray, regime: str = 'unknown') -> int:
        if regime in self._cache:
            return self._cache[regime]

        max_l = min(self.max_lag, len(x) // 4)
        ami_vals = [self._ami(x, lag) for lag in range(1, max_l)]
        ami_arr = np.array(ami_vals)

        tau = 1
        for i in range(1, len(ami_arr) - 1):
            if ami_arr[i] < ami_arr[i-1] and ami_arr[i] < ami_arr[i+1]:
                tau = i + 1
                break
        else:
            threshold = ami_arr[0] / np.e
            for i, v in enumerate(ami_arr):
                if v < threshold:
                    tau = i + 1
                    break

        self._cache[regime] = tau
        return tau

# ═══════════════════════════════════════════
# NEW: ADAPTADOR DE SAMPEN POR RÉGIMEN
# ═══════════════════════════════════════════
class SampEnAdaptor:
    """
    Configura m, r_ratio y bandas de coherencia de SampEn por régimen.
    Elimina umbrales binarios → usa funciones suaves de pertenencia.
    """
    CONFIG = {
       'stable':         {'m': 2, 'r_ratio': 0.15, 'mu': 0.02,  'sigma': 0.015},
        'weakly_chaotic': {'m': 2, 'r_ratio': 0.20, 'mu': 0.57,  'sigma': 0.12},
        'chaotic':        {'m': 2, 'r_ratio': 0.25, 'mu': 0.53,  'sigma': 0.10},
        'hyperchaotic':   {'m': 2, 'r_ratio': 0.30, 'mu': 1.51,  'sigma': 0.30},
        'noisy':          {'m': 2, 'r_ratio': 0.35, 'mu': 1.65,  'sigma': 0.35},    }
    
    @classmethod
    def get(cls, regime: str) -> dict:
        return cls.CONFIG.get(regime, cls.CONFIG['weakly_chaotic'])
    
    @classmethod
    def compatibility_weight(cls, sampen_val: float, regime: str) -> float:
        """Retorna [0,1] según cuán coherente es SampEn con el régimen detectado."""
        cfg = cls.get(regime)
        return float(np.exp(-0.5 * ((sampen_val - cfg['mu']) / cfg['sigma'])**2))

# ═══════════════════════════════════════════
# 4. MÉTRICAS DINÁMICAS
# ═══════════════════════════════════════════
class Metrics:
    # ── 4a. Exponente de Lyapunov (Rosenstein et al., 1993) ──────────
    @staticmethod
    def lyapunov(x: np.ndarray, tau: int, m: int = 3,
                 min_tsep: int = None, max_iter: int = 300) -> float:
        Y = embed(x, m, tau)
        N = len(Y)
        if min_tsep is None:
            min_tsep = max(1, int(0.1 * N))

        dists = cdist(Y, Y)
        np.fill_diagonal(dists, np.inf)

        divergences = []
        for i in range(N):
            mask = np.abs(np.arange(N) - i) > min_tsep
            d = dists[i].copy()
            d[~mask] = np.inf
            j = np.argmin(d)
            if np.isinf(d[j]) or d[j] == 0:
                continue

            steps = min(max_iter, N - max(i, j) - 1)
            if steps < 3:
                continue

            d_series = np.array([
                np.linalg.norm(Y[min(i+k, N-1)] - Y[min(j+k, N-1)])
                for k in range(steps)
            ])
            pos = d_series > 0
            if pos.sum() < 3:
                continue

            t = np.where(pos)[0]
            log_d = np.log(d_series[pos] / d[j])
            if len(t) > 1:
                slope = np.polyfit(t, log_d, 1)[0]
                divergences.append(slope)

        return float(np.median(divergences)) if divergences else np.nan

    # ── 4b. Dimensión de Correlación (Grassberger & Procaccia, 1983) ──
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

        slope = np.polyfit(np.log(r_vals[valid]), np.log(C_r[valid]), 1)[0]
        return float(slope)

    # ── 4c. Complejidad Lempel-Ziv (Lempel & Ziv, 1976) ─────────────
    @staticmethod
    def lempel_ziv(x: np.ndarray, Y: np.ndarray = None) -> float:
        if Y is not None and Y.shape[0] > 10:
            Y_centered = Y - Y.mean(axis=0)
            cov = np.cov(Y_centered.T)
            if cov.ndim == 0:
                proj = Y_centered[:, 0]
            else:
                eigvals, eigvecs = np.linalg.eigh(cov)
                proj = Y_centered @ eigvecs[:, -1]
            seq = proj
        else:
            seq = x

        binary = ''.join('1' if v > np.median(seq) else '0' for v in seq)
        n = len(binary)
        c, l, i, k, k_max = 1, 1, 0, 1, 1
        stop = False
        while not stop:
            if i + k <= n and l + k <= n and binary[i+k-1] == binary[l+k-1]:
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

    # ── 4d. Transferencia de Entropía (Schreiber, 2000) ──────────────
    @staticmethod
    def transfer_entropy(x: np.ndarray, tau: int, bins: int = 8) -> float:
        n = len(x) - tau
        if n < 20:
            return np.nan
        X, Xf = x[:n], x[tau:n+tau]
        h2d, _, _ = np.histogram2d(X, Xf, bins=bins)
        pxy = h2d / (h2d.sum() + 1e-12)
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        te = 0.0
        for i in range(bins):
            for j in range(bins):
                if pxy[i,j] > 0 and px[i] > 0 and py[j] > 0:
                    te += pxy[i,j] * np.log2(pxy[i,j] / (px[i] * py[j]))
        return float(max(0.0, te))

    # ── 4e. Entropía de muestra (Richman & Moorman, 2000) ────────────
    @staticmethod
    def sample_entropy(x: np.ndarray, m: int = 2, r_ratio: float = 0.2, tau: int = 1) -> float:
        """
        SampEn con lag τ: alinea la métrica con la reconstrucción de fase de Takens.
        Permite que ∂SampEn/∂τ sea físicamente significativo en R3.
        """
        try:
            N = len(x)
            r = r_ratio * np.std(x, ddof=1)
            if r == 0 or N < (m + 1) * tau:
                return np.nan

            def _phi(m_val):
                n_t = N - (m_val - 1) * tau
                if n_t < 2: return np.nan
                # Templates espaciados por τ (coherente con embedding)
                templates = np.array([x[j:j + m_val*tau:tau] for j in range(n_t)])
                # Distancia máxima por dimensión (vectorizada)
                dists = np.max(np.abs(templates[:, None, :] - templates[None, :, :]), axis=2)
                counts = np.sum(dists <= r, axis=1) - 1  # excluir auto-comparación
                mean_c = np.mean(counts)
                return np.log(mean_c) if mean_c > 0 else -np.inf

            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)
            if np.isinf(phi_m) or np.isinf(phi_m1):
                return np.nan
            return phi_m - phi_m1
        except:
            return np.nan

    @classmethod
    def compute_all(cls, x: np.ndarray, tau: int, m: int = 3, regime: str = None) -> dict:
        Y = embed(x, m, tau)
        cfg = SampEnAdaptor.get(regime) if regime else {'m': 2, 'r_ratio': 0.2}
        return {
            'lambda': cls.lyapunov(x, tau, m),
            'D2':     cls.correlation_dimension(Y),
            'LZ':     cls.lempel_ziv(x, Y),
            'TE':     cls.transfer_entropy(x, tau),
            'SampEn': cls.sample_entropy(x, m=cfg['m'], r_ratio=cfg['r_ratio'], tau=tau),
        }

# ═══════════════════════════════════════════
# 5. DETECTOR DE RÉGIMEN
# ═══════════════════════════════════════════
class RegimeDetector:
    DESCRIPTIONS = {
        'stable':         'Estable / Periódico',
        'weakly_chaotic': 'Caos débil / Cuasiperiódico',
        'chaotic':        'Caótico',
        'hyperchaotic':   'Hiperc aótico / Estructurado',
        'noisy':          'Ruido / Sin estructura dinámica',
    }

    def classify(self, lam: float, lz: float = None, d2: float = None) -> str:
        if not np.isnan(lam) and lam < 0:
            return 'stable'

        if d2 is not None and not np.isnan(d2) and lz is not None and not np.isnan(lz):
            if lz > 0.95 and d2 > 2.3:
                return 'noisy'
            if lz > 0.92 and d2 > 2.0:
                return 'hyperchaotic'
            if lz > 0.55 and d2 > 1.6:
                return 'chaotic'
            if lz < 0.30 and d2 < 1.2:
                return 'stable'
            return 'weakly_chaotic'

        if lz is not None and not np.isnan(lz):
            if lz < 0.25: return 'stable'
            if lz < 0.55: return 'weakly_chaotic'
            if lz < 0.92: return 'chaotic'
            return 'hyperchaotic'

        if not np.isnan(lam):
            if lam < 0.15: return 'weakly_chaotic'
            if lam < 0.50: return 'chaotic'
            return 'hyperchaotic'

        return 'weakly_chaotic'

# ═══════════════════════════════════════════
# 6. BIBLIOTECA DE δ SEMIDINAMICO
# ═══════════════════════════════════════════
class DeltaLibrary:
    TABLE = {
        'stable':         0.06,
        'weakly_chaotic': 0.07,
        'chaotic':        0.08,
        'hyperchaotic':   0.15,
        'noisy':          0.20,
    }

    def get(self, regime: str) -> dict:
        try:
            from r3_delta_library import DELTA_LIBRARY
            if regime in DELTA_LIBRARY:
                return DELTA_LIBRARY[regime]
        except Exception:
            pass
        scalar = self.TABLE.get(regime, 0.10)
        return {k: scalar for k in ('lambda', 'D2', 'LZ', 'TE', 'SampEn')}

# ═══════════════════════════════════════════
# 7. H3 — R³ DESCRIPTOR
# ═══════════════════════════════════════════
class R3Descriptor:
    COHERENCE_THRESHOLD = 0.57

    def __init__(self):
        self.regime_detector = RegimeDetector()
        self.delta_lib = DeltaLibrary()

    def _gradients(self, x: np.ndarray, tau: int, m: int) -> tuple:
        # 1. Cálculo base y detección de régimen
        base = Metrics.compute_all(x, tau, m)
        regime = self.regime_detector.classify(
            base.get('lambda', np.nan), 
            base.get('LZ', np.nan), 
            base.get('D2', np.nan)
        )
        
        # 2. Recalcular SampEn con parámetros adaptados al régimen detectado
        cfg = SampEnAdaptor.get(regime)
        base['SampEn'] = Metrics.sample_entropy(x, m=cfg['m'], r_ratio=cfg['r_ratio'], tau=tau)
        base['_regime'] = regime  # Temporal para uso en score()

        tau_p = max(1, tau + 1)
        tau_m = max(1, tau - 1)
        
        mp = Metrics.compute_all(x, tau_p, m, regime=regime)
        mm = Metrics.compute_all(x, tau_m, m, regime=regime) if tau_m != tau else base

        grads = {}
        for k in ('lambda', 'D2', 'LZ', 'TE', 'SampEn'):
            v0 = base.get(k, np.nan)
            vp = mp.get(k, np.nan)
            vm = mm.get(k, np.nan)
            if any(np.isnan([v0, vp, vm])):
                grads[k] = np.nan
            else:
                denom = np.sqrt(v0**2 + vp**2 + vm**2 + 1e-12) / np.sqrt(3)
                grads[k] = abs(vp - vm) / denom if denom > 1e-10 else abs(vp - vm)
        return grads, base

    def score(self, x: np.ndarray, tau: int, m: int = 3) -> dict:
        grads, metrics = self._gradients(x, tau, m)
        regime = metrics.pop('_regime', 'weakly_chaotic')
        delta = self.delta_lib.get(regime)

        stability_map = {}
        stability_weights = []
        valid_n = 0

        for k, g in grads.items():
            if not np.isnan(g):
                valid_n += 1
                is_stable = g < delta
                w_stab = max(0.0, 1.0 - (g / delta)) if delta > 0 else 0.0
                
                # Modulación contextual solo para SampEn
                w_compat = 1.0
                if k == 'SampEn':
                    w_compat = SampEnAdaptor.compatibility_weight(metrics['SampEn'], regime)
                    weight = w_stab * w_compat
                else:
                    weight = w_stab
                    
                stability_weights.append(weight)
                stability_map[k] = {
                    'gradient': g, 
                    'stable': is_stable, 
                    'delta': delta,
                    'weight': weight,
                    'w_compat': w_compat  # Extra para debugging/visualización
                }

        r3_score = np.mean(stability_weights) if stability_weights else 0.0
        coherent = (r3_score >= self.COHERENCE_THRESHOLD) and (regime != 'noisy')

       # ── Score vectorial — preserva distribución de coherencia ──
        if valid_n < 2:
            r3_score    = np.nan
            r3_std      = np.nan
            r3_min      = np.nan
            r3_dominant = 'insuficiente'
        else:
            r3_score    = float(np.mean(stability_weights))
            r3_std      = float(np.std(stability_weights))
            r3_min      = float(np.min(stability_weights))
            # Métrica dominante: la más inestable (la que más tira R³ abajo)
            r3_dominant = min(stability_map, key=lambda k: stability_map[k]['weight'])

        # Coherencia: media ≥ umbral Y mínimo ≥ umbral/2 Y no ruidoso
        # El mínimo evita que una métrica colapsada se oculte en el promedio
        if np.isnan(r3_score):
            coherent = False
        else:
            coherent = (
                r3_score >= self.COHERENCE_THRESHOLD and
                r3_min   >= self.COHERENCE_THRESHOLD / 2 and
                regime   != 'noisy'
            )

        return {
            'R3_score':     r3_score,      # media — comparabilidad con versiones anteriores
            'R3_std':       r3_std,        # dispersión interna — variabilidad de coherencia
            'R3_min':       r3_min,        # métrica más inestable — el eslabón débil
            'R3_dominant':  r3_dominant,   # qué métrica tira R³ abajo
            'R3_vector':    {k: round(v['weight'], 8)
                             for k, v in stability_map.items()},
            'coherent':     coherent,
            'regime':       regime,
            'regime_desc':  RegimeDetector.DESCRIPTIONS.get(regime, regime),
            'delta':        delta,
            'metrics':      metrics,
            'gradients':    grads,
            'stability_map': stability_map,
            'n_valid':      valid_n,
        }
# ═══════════════════════════════════════════
# 8. PIPELINE INTEGRADO
# ═══════════════════════════════════════════
class AttractorPipeline:
    def __init__(self, m: int = 3, max_tau: int = 50, verbose: bool = True):
        self.m = m
        self.verbose = verbose
        self._eps = DynamicEpsilon()
        self._tau = SemidynamicTau(max_lag=max_tau)
        self._r3  = R3Descriptor()
        self.results: dict = {}

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def run(self, x: np.ndarray, label: str = 'serie') -> dict:
        x = np.asarray(x, dtype=float)
        x = (x - x.mean()) / (x.std() + 1e-12)

        self._tau._cache.clear()

        self._log(f"\n{'═'*52}")
        self._log(f" PIPELINE · {label} · N={len(x)}")
        self._log(f"{'═'*52}")

        tau0     = self._tau.compute(x, regime='unknown')
        self._log(f" τ₀ (AMI, sin régimen): {tau0}")

        metrics0 = Metrics.compute_all(x, tau0, self.m)
        regime0  = RegimeDetector().classify(metrics0['lambda'], metrics0['LZ'], metrics0['D2'])
        self._log(f" Régimen detectado: {regime0}")

        tau = self._tau.compute(x, regime=regime0)
        if tau != tau0:
            self._log(f" τ refinado (régimen): {tau}")
        else:
            self._log(f" τ confirmado: {tau}")

        Y          = embed(x, self.m, tau)
        self._log(f" Embedding shape: {Y.shape}")

        eps_series = self._eps.compute_series(Y)
        eps_scalar = float(np.median(eps_series))
        self._log(f" ε (mediana): {eps_scalar:.5f}")

        metrics = Metrics.compute_all(x, tau, self.m)
        self._log(f"\n Métricas:")
        self._log(f"  λ  (Lyapunov)  = {metrics['lambda']}")
        self._log(f"  D₂ (Corr. dim) = {metrics['D2']}")
        self._log(f"  LZ (Compl.)    = {metrics['LZ']}")
        self._log(f"  TE (Trans.ent) = {metrics['TE']}")
        self._log(f"  SE (Muestra)   = {metrics['SampEn']}")

        r3 = self._r3.score(x, tau, self.m)
        self._log(f"\n R³ descriptor:")
        self._log(f"  Score    = {r3['R3_score']:.6f}")
        self._log(f"  Coherente= {r3['coherent']}")
        self._log(f"  Régimen  = {r3['regime_desc']}")
        self._log(f"  δ activo = {r3['delta']}")
        for k, v in r3['stability_map'].items():
            sym = '✔' if v['stable'] else '✘'
            self._log(f"  {sym} {k:<8} grad={v['gradient']:.6f} weight={v['weight']:.4f} δ={v['delta']}")
        self._log(f"{'═'*52}\n")

        result = {
            'label':          label,
            'x_normalized':   x,
            'tau':            tau,
            'tau_initial':    tau0,
            'epsilon':        eps_scalar,
            'epsilon_series': eps_series,
            'embedding':      Y,
            'metrics':        metrics,
            'regime':         r3['regime'],
            'regime_desc':    r3['regime_desc'],
            'R3':             r3,
        }
        self.results[label] = result
        return result

# ═══════════════════════════════════════════
# SEÑALES DE REFERENCIA
# ═══════════════════════════════════════════
def _logistic_map(N: int = 1000, r: float = 3.9) -> np.ndarray:
    x = 0.1
    out = [x]
    for _ in range(N - 1):
        x = r * x * (1 - x)
        out.append(x)
    return np.array(out)

logistic_map = _logistic_map

def demo_signals(N: int = 1000) -> dict:
    """Genera señales de referencia con dinámicas conocidas."""
    t   = np.linspace(0, 100, N)
    rng = np.random.default_rng(0)
    
    def lorenz_ts(n=N, sigma=10, rho=28, beta=8/3, dt=0.01):
        x, y, z = 1.0, 1.0, 1.05
        for _ in range(2000):
            dx = sigma*(y-x); dy = x*(rho-z)-y; dz = x*y-beta*z
            x += dx*dt; y += dy*dt; z += dz*dt
        xs = []
        for _ in range(n):
            dx = sigma*(y-x); dy = x*(rho-z)-y; dz = x*y-beta*z
            x += dx*dt; y += dy*dt; z += dz*dt
            xs.append(x)
        return np.array(xs)

    def rossler_ts(n=N, a=0.2, b=0.2, c=5.7, dt=0.05):
        x, y, z = 1.0, 0.0, 0.0
        for _ in range(1000):
            dx = -y-z; dy = x+a*y; dz = b+z*(x-c)
            x += dx*dt; y += dy*dt; z += dz*dt
        xs = []
        for _ in range(n):
            dx = -y-z; dy = x+a*y; dz = b+z*(x-c)
            x += dx*dt; y += dy*dt; z += dz*dt
            xs.append(x)
        return np.array(xs)

    return {
        'lorenz':   lorenz_ts(),
        'rossler':  rossler_ts(),
        'periodic': np.sin(2*np.pi*0.1*t) + 0.05*rng.normal(size=N),
        'noisy':    rng.normal(size=N),
        'logistic': _logistic_map(N, r=3.9),
    }

if __name__ == '__main__':
    pipe = AttractorPipeline(m=3, max_tau=50, verbose=True)
    N    = 800
    t    = np.linspace(0, 80, N)
    rng  = np.random.default_rng(42)
    signals = {
        'periodic': np.sin(2*np.pi*0.1*t) + 0.05*rng.normal(size=N),
        'logistic': _logistic_map(N, r=3.9),
        'noisy':    rng.normal(size=N),
    }
    for name, sig in signals.items():
        pipe.run(sig, label=name)
