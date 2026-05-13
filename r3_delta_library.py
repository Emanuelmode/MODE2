"""
r3_delta_library.py
═══════════════════════════════════════════════════════════════
Biblioteca de umbrales δ_{k,r} por métrica y régimen.

Recalibración sintética conservadora v2:
  stable:          sinusoidal de bajo ruido + logístico r=3.5
  weakly_chaotic:  Rössler + transiciones suaves
  chaotic:         Lorenz + logístico r=3.7 / r=3.9
  noisy:           ruido blanco / rosa

Criterio:
- δ expresa tolerancia de estabilidad frente a τ.
- Se ajusta sólo donde aparecieron falsos negativos reproducibles
  en demos sintéticas canónicas.
- La compuerta de coherencia por mínimo interno (R3_min) sigue
  bloqueando falsos positivos ruidosos aunque el score medio suba.

Actualización: mayo 2026 — Emanuel Duarte
═══════════════════════════════════════════════════════════════
"""

DELTA_LIBRARY = {
    # ── Estable / Periódico ──────────────────────────────────────────
    'stable': {
        'lambda': 0.26,
        'D2':     0.03,
        'LZ':     0.03,
        'TE':     0.08,
        'SampEn': 0.08,
    },

    # ── Caos débil / Cuasiperiódico ─────────────────────────────────
    # λ y TE se amplían sólo lo necesario para que Rössler no quede
    # injustamente penalizado. LZ se mantiene estricta para que
    # senoidal con ruido alto y ruido rosa sigan bloqueados por mínimo.
    'weakly_chaotic': {
    'lambda': 0.18,
    'D2':     0.07,
    'LZ':     0.18,
    'TE':     0.18,
    'SampEn': 0.08,
    },

    # ── Caótico ─────────────────────────────────────────────────────
    # TE se amplía levemente para rescatar el logístico r=3.7 sin
    # relajar el resto del régimen.
    'chaotic': {
        'lambda': 0.18,
        'D2':     0.03,
        'LZ':     0.20,
        'TE':     0.145,
        'SampEn': 0.05,
    },

    # ── Hipercaótico / Estructurado ─────────────────────────────────
    'hyperchaotic': {
        'lambda': 0.15,
        'D2':     0.03,
        'LZ':     0.18,
        'TE':     0.10,
        'SampEn': 0.12,
    },

    # ── Ruido / Sin estructura dinámica ─────────────────────────────
    'noisy': {
        'lambda': 0.20,
        'D2':     0.05,
        'LZ':     0.08,
        'TE':     0.12,
        'SampEn': 0.35,
    },
}
