"""
r3_delta_library.py
═══════════════════════════════════════════════════════════════
Biblioteca de umbrales δ_{k,r} por métrica y régimen.

Recalibración honesta para demos sintéticas de referencia:
  stable:          sinusoidal de bajo ruido + logístico r=3.5
  weakly_chaotic:  Rössler + transiciones suaves
  chaotic:         Lorenz + logístico r=3.9 / r=3.7
  hyperchaotic:    reservado, sin demo sintética primaria actual
  noisy:           ruido blanco / rosa

Objetivo:
- Corregir la discriminación de régimen sin empujar artificialmente R3.
- δ actúa como tolerancia de estabilidad en τ, no como truco para inflar score.
- Los regímenes con referencias heterogéneas usan envolventes conservadoras.

Actualización: mayo 2026 — Emanuel Duarte
═══════════════════════════════════════════════════════════════
"""

DELTA_LIBRARY = {
    # ── Estable / Periódico ──────────────────────────────────────────
    # Más estricto que caos débil, pero no tan estrecho como para
    # penalizar sinusoides con leve ruido observacional.
    'stable': {
        'lambda': 0.08,
        'D2':     0.03,
        'LZ':     0.03,
        'TE':     0.08,
        'SampEn': 0.08,
    },

    # ── Caos débil / Cuasiperiódico ─────────────────────────────────
    # Envelope conservadora basada en Rössler y transiciones suaves.
    'weakly_chaotic': {
        'lambda': 0.06,
        'D2':     0.03,
        'LZ':     0.04,
        'TE':     0.05,
        'SampEn': 0.08,
    },

    # ── Caótico ─────────────────────────────────────────────────────
    # Incluye atractores clásicos (Lorenz) y mapas caóticos 1D.
    # Se amplía SampEn respecto a la versión previa para no arrastrar
    # artificialmente sistemas con complejidad real pero embedding distinto.
    'chaotic': {
        'lambda': 0.18,
        'D2':     0.03,
        'LZ':     0.13,
        'TE':     0.08,
        'SampEn': 0.05,
    },

    # ── Hipercaótico / Estructurado ─────────────────────────────────
    # Se mantiene amplio porque aún no hay demo primaria bien cerrada.
    'hyperchaotic': {
        'lambda': 0.15,
        'D2':     0.03,
        'LZ':     0.18,
        'TE':     0.10,
        'SampEn': 0.12,
    },

    # ── Ruido / Sin estructura dinámica ─────────────────────────────
    # Umbrales permisivos. coherent=False sigue dependiendo del régimen.
    'noisy': {
        'lambda': 0.20,
        'D2':     0.05,
        'LZ':     0.08,
        'TE':     0.12,
        'SampEn': 0.35,
    },
}
