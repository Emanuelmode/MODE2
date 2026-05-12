"""
r3_delta_library.py
═══════════════════════════════════════════════════════════════
Biblioteca de umbrales δ_{k,r} por métrica y régimen.

Calibración: percentil 80 de gradientes RMS observados en señales
CONOCIDAMENTE COHERENTES dentro de cada régimen.

Señales de referencia:
  stable:        Lorenz (dt=0.01) + Logístico r=3.5
  chaotic:       Logístico r=3.7
  hyperchaotic:  Logístico r=3.9
  weakly_chaotic: sin señales coherentes → defaults conservadores
  noisy:         sin señales coherentes → coherente=False por definición

SampEn opera sobre proyección PCA-1 del embedding (tau-sensible).
Para embeddings degenerados (std_proj < 1e-8), SampEn → NaN → excluido.

Fuentes primarias:
  Peng et al. (1995)           — HRV fisiológico
  Grassberger & Procaccia (1983) — atractores clásicos
  Mantegna & Stanley (1999)    — series financieras
  Schreiber (2000)             — transferencia de entropía
  Richman & Moorman (2000)     — entropía de muestra

Actualización: mayo 2026 — Emanuel Duarte
═══════════════════════════════════════════════════════════════
"""

DELTA_LIBRARY = {

    # ── Estable / Periódico ──────────────────────────────────────────
    # Calibrado: Lorenz (λ=0.0024, D2=0.0066, LZ=0.0044, TE=0.0002, SE=0.1807)
    #            Logístico r=3.5 (λ=0.0068, D2=0.0853, LZ=0.0043, TE=0.0244, SE=0.6034)
    # Nota: SE alta en logístico r=3.5 refleja embedding casi-degenerado (epsilon~0)
    #       — sensibilidad real, no artefacto.
    'stable': {
        'lambda': 0.02,   # p80=0.006 → floor 0.02 (mínimo operacional)
        'D2':     0.10,   # p80=0.074 → redondeado a 0.10 (acomoda orbita periódica)
        'LZ':     0.02,   # p80=0.004 → floor 0.02
        'TE':     0.03,   # p80=0.020 → ceil a 0.03
        'SampEn': 0.65,   # p80=0.519 → ceil a 0.65 (acomoda logístico r=3.5)
    },

    # ── Caos débil / Cuasiperiódico ─────────────────────────────────
    # Sin señales coherentes de referencia en este régimen.
    # Valores conservadores derivados de la estructura de la métrica.
    'weakly_chaotic': {
        'lambda': 0.08,   # λ variable en caos débil
        'D2':     0.05,   # D2 estable globalmente
        'LZ':     0.07,   # LZ moderadamente sensible
        'TE':     0.12,   # TE más variable en caos débil
        'SampEn': 0.35,   # SampEn moderado
    },

    # ── Caótico ─────────────────────────────────────────────────────
    # Calibrado: Logístico r=3.7 (λ=0.1734, D2=0.0033, LZ=0.1686, TE=0.0539, SE=0.0124)
    'chaotic': {
        'lambda': 0.19,   # p80=0.1734 → ceil a 0.18 + margen → 0.19
        'D2':     0.02,   # p80=0.003 → floor 0.02
        'LZ':     0.18,   # p80=0.1686 → ceil a 0.18
        'TE':     0.07,   # p80=0.054 → ceil a 0.07
        'SampEn': 0.02,   # p80=0.012 → ceil a 0.02
    },

    # ── Hipercaótico / Estructurado ──────────────────────────────────
    # Calibrado: Logístico r=3.9 (λ=0.1142, D2=0.0019, LZ=0.0043, TE=0.0473, SE=0.1540)
    'hyperchaotic': {
        'lambda': 0.13,   # p80=0.114 → ceil a 0.13
        'D2':     0.02,   # p80=0.002 → floor 0.02
        'LZ':     0.02,   # p80=0.004 → floor 0.02
        'TE':     0.06,   # p80=0.047 → ceil a 0.06
        'SampEn': 0.17,   # p80=0.154 → ceil a 0.17
    },

    # ── Ruido / Sin estructura dinámica ──────────────────────────────
    # Sin señales coherentes (ruido es incoherente por definición).
    # Umbrales permisivos — el flag coherent=False viene del régimen, no de δ.
    'noisy': {
        'lambda': 0.20,
        'D2':     0.05,
        'LZ':     0.07,
        'TE':     0.12,
        'SampEn': 0.35,
    },
}
