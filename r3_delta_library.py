"""
r3_delta_library.py
═══════════════════════════════════════════════════════════════
Biblioteca de umbrales δ_{k,r} por métrica y régimen.

Recalibración sintética conservadora:
  stable:          sinusoidal de bajo ruido + logístico r=3.5
  weakly_chaotic:  Rössler + transiciones suaves
  chaotic:         Lorenz + logístico r=3.7 / r=3.9
  noisy:           ruido blanco / rosa

Criterio:
- δ expresa tolerancia de estabilidad frente a τ.
- No se usa para “dibujar” coherencia, sino para evitar falsos negativos
  evidentes en demos sintéticas ya conocidas.
- SampEn queda deliberadamente acotada y su compatibilidad se reporta
  aparte del score principal.

Actualización: mayo 2026 — Emanuel Duarte
═══════════════════════════════════════════════════════════════
"""

DELTA_LIBRARY = {
    # ── Estable / Periódico ──────────────────────────────────────────
    # λ se amplía para no penalizar sinusoides con ruido bajo por
    # inestabilidad numérica del Lyapunov local.
    'stable': {
        'lambda': 0.26,
        'D2':     0.03,
        'LZ':     0.03,
        'TE':     0.08,
        'SampEn': 0.08,
    },

    # ── Caos débil / Cuasiperiódico ─────────────────────────────────
    # Se relaja λ y TE para no castigar en exceso a Rössler;
    # LZ permanece estricta para no confundir ruido coloreado con coherencia.
    'weakly_chaotic': {
        'lambda': 0.12,
        'D2':     0.03,
        'LZ':     0.04,
        'TE':     0.12,
        'SampEn': 0.08,
    },

    # ── Caótico ─────────────────────────────────────────────────────
    # LZ y TE se amplían suavemente para acomodar Lorenz y mapas
    # logísticos caóticos sin bajar la vara frente al ruido.
    'chaotic': {
        'lambda': 0.18,
        'D2':     0.03,
        'LZ':     0.20,
        'TE':     0.11,
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
