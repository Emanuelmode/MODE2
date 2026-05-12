"""
mitbih_analysis.py
═══════════════════════════════════════════════════════════════
Análisis de señales ECG MIT-BIH con el pipeline MODE
Investigador: Emanuel Duarte — Pergamino, Argentina — 2026

Uso:
    python mitbih_analysis.py

Requiere:
    - Carpeta 'mitbih/' con los archivos descomprimidos
    - pipeline.py en el mismo directorio
    - numpy, scipy, matplotlib
═══════════════════════════════════════════════════════════════
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

sys.path.insert(0, '.')
from pipeline import AttractorPipeline

# ═══════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════

BASE_PATH   = 'mitbih/mit-bih-arrhythmia-database-1.0.0/'
OUTPUT_DIR  = 'output_mitbih'
AUTHOR      = 'Investigador: Emanuel Duarte'

# Registros a analizar
RECORDS = {
    '100': 'Ritmo sinusal normal (referencia)',
    '208': 'Arritmias ventriculares frecuentes',
    '214': 'Bloqueo y ritmo irregular',
}

# Parámetros de ventana
WINDOW_SIZE  = 1000   # muestras por ventana (~2.8 seg a 360 Hz)
WINDOW_STEP  = 500    # paso entre ventanas (50% overlap)
MAX_WINDOWS  = 60     # máximo de ventanas por registro (no correr todo)

# Parámetros del pipeline
M       = 3
MAX_TAU = 40

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════
# LECTOR MIT-BIH (formato 212)
# ═══════════════════════════════════════════

def read_mitbih(record_id):
    """
    Lee canal MLII de un registro MIT-BIH en formato 212.
    Retorna: (señal_mV, frecuencia_muestreo, n_muestras)
    """
    hea_path = BASE_PATH + str(record_id) + '.hea'
    dat_path = BASE_PATH + str(record_id) + '.dat'

    if not os.path.exists(hea_path):
        raise FileNotFoundError(f"No se encontró {hea_path}")

    # Leer header
    with open(hea_path) as f:
        lines = f.readlines()

    header   = lines[0].strip().split()
    fs       = int(header[2])
    n_samp   = int(header[3])
    sig_info = lines[1].strip().split()
    gain     = float(sig_info[2])   # ADC gain (adu/mV)
    baseline = int(sig_info[4])     # ADC baseline

    # Leer datos binarios formato 212
    # (3 bytes por 2 muestras, intercaladas por canal)
    with open(dat_path, 'rb') as f:
        raw = f.read()

    samples = []
    i = 0
    while i + 2 < len(raw):
        b0, b1, b2 = raw[i], raw[i+1], raw[i+2]
        # Muestra 1: byte0 + nibble bajo de byte1
        s1 = b0 | ((b1 & 0x0F) << 8)
        if s1 >= 2048: s1 -= 4096
        # Muestra 2: byte2 + nibble alto de byte1
        s2 = b2 | ((b1 & 0xF0) << 4)
        if s2 >= 2048: s2 -= 4096
        samples.extend([s1, s2])
        i += 3

    # Canal MLII: muestras con índice par (canal 0)
    mlii = np.array(samples[::2][:n_samp])
    signal_mv = (mlii - baseline) / gain
    return signal_mv, fs, n_samp


# ═══════════════════════════════════════════
# ANÁLISIS POR VENTANAS
# ═══════════════════════════════════════════

def analyze_record(record_id, description):
    """
    Carga un registro, segmenta en ventanas y corre el pipeline.
    Retorna DataFrame con resultados por ventana.
    """
    print(f"\n{'═'*60}")
    print(f"  Registro {record_id}: {description}")
    print(f"{'═'*60}")

    # Cargar señal
    signal, fs, n_total = read_mitbih(record_id)
    print(f"  Cargado: {n_total} muestras  |  {fs} Hz  |  {n_total/fs/60:.1f} min")

    # Inicializar pipeline (nuevo por registro para limpiar cache)
    pipe = AttractorPipeline(m=M, max_tau=MAX_TAU, verbose=False)

    results = []
    n_windows = 0
    t_start   = 0

    while t_start + WINDOW_SIZE <= len(signal) and n_windows < MAX_WINDOWS:
        window = signal[t_start : t_start + WINDOW_SIZE]

        # Saltar ventanas completamente planas (artefacto)
        if window.std() < 1e-6:
            t_start += WINDOW_STEP
            continue

        t_sec = t_start / fs

        try:
            r = pipe.run(window, label=f"rec{record_id}_t{t_sec:.0f}s")
            r3 = r['R3']
            m  = r['metrics']

            results.append({
                'record':    record_id,
                'ventana':   n_windows,
                't_inicio_s': round(t_sec, 2),
                't_fin_s':   round((t_start + WINDOW_SIZE) / fs, 2),
                'tau':       r['tau'],
                'epsilon':   round(r['epsilon'], 5),
                'R3_score':  r3['R3_score'],
                'coherente': r3['coherent'],
                'regimen':   r3['regime'],
                'regimen_desc': r3['regime_desc'],
                'delta':     r3['delta'],
                'lambda':    round(m['lambda'], 5) if m['lambda'] else None,
                'D2':        round(m['D2'], 4)     if m['D2']     else None,
                'LZ':        round(m['LZ'], 4)     if m['LZ']     else None,
                'TE':        round(m['TE'], 4)     if m['TE']     else None,
            })

            coh = '✔' if r3['coherent'] else '✘'
            print(f"  t={t_sec:6.1f}s  τ={r['tau']:2d}  "
                  f"R³={r3['R3_score']:.3f} {coh}  "
                  f"régimen={r3['regime']:<15}  "
                  f"δ={r3['delta']}")

        except Exception as e:
            print(f"  t={t_sec:6.1f}s  ERROR: {e}")

        t_start  += WINDOW_STEP
        n_windows += 1

    df = pd.DataFrame(results)
    print(f"\n  Total ventanas analizadas: {len(df)}")
    if len(df) > 0:
        coh_pct = df['coherente'].mean() * 100
        print(f"  Ventanas coherentes:       {coh_pct:.1f}%")
        print(f"  R³ promedio:               {df['R3_score'].mean():.3f}")
        print(f"  Regímenes detectados:      {df['regimen'].value_counts().to_dict()}")

    return df


# ═══════════════════════════════════════════
# VISUALIZACIONES
# ═══════════════════════════════════════════

PALETTE = {
    'bg': '#0d1117', 'surface': '#161b22', 'text': '#e6edf3',
    'accent': '#58a6ff', 'green': '#3fb950', 'red': '#f85149',
    'orange': '#d29922', 'purple': '#bc8cff', 'border': '#30363d'
}

plt.rcParams.update({
    'figure.facecolor': PALETTE['bg'],  'axes.facecolor':  PALETTE['surface'],
    'axes.edgecolor':   PALETTE['border'], 'axes.labelcolor': PALETTE['text'],
    'xtick.color':      PALETTE['border'], 'ytick.color':     PALETTE['border'],
    'text.color':       PALETTE['text'],   'grid.color':      PALETTE['border'],
    'grid.alpha': 0.35, 'font.family': 'monospace', 'font.size': 9,
})

def add_watermark(fig):
    fig.text(0.5, 0.5, AUTHOR, fontsize=11, color='white', alpha=0.08,
             ha='center', va='center', rotation=30, transform=fig.transFigure)
    fig.text(0.99, 0.01, AUTHOR, fontsize=7, color='white', alpha=0.28,
             ha='right', va='bottom', transform=fig.transFigure)


def plot_r3_timeline(df, record_id, description, signal, fs):
    """R³ score y coherencia a lo largo del tiempo."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), facecolor=PALETTE['bg'])
    fig.suptitle(f'Registro {record_id} — {description}',
                 color=PALETTE['accent'], fontsize=12, fontweight='bold')

    t = df['t_inicio_s'].values
    r3 = df['R3_score'].values
    coh = df['coherente'].values

    # Panel 1: ECG raw (primeros 10 segundos)
    ax1 = axes[0]; ax1.set_facecolor(PALETTE['surface'])
    t_ecg = np.arange(min(3600, len(signal))) / fs
    ax1.plot(t_ecg, signal[:len(t_ecg)], lw=0.5, color=PALETTE['accent'], alpha=0.9)
    ax1.set_ylabel('mV', fontsize=8)
    ax1.set_title('Señal ECG (primeros 10 seg)', fontsize=9, color=PALETTE['text'])
    ax1.grid(True, alpha=0.3)

    # Panel 2: R³ score por ventana
    ax2 = axes[1]; ax2.set_facecolor(PALETTE['surface'])
    colors = [PALETTE['green'] if c else PALETTE['red'] for c in coh]
    ax2.bar(t, r3, width=WINDOW_STEP/fs*0.8, color=colors, alpha=0.8)
    ax2.axhline(0.75, color=PALETTE['orange'], lw=1, ls='--', label='Umbral R³=0.75')
    ax2.set_ylabel('R³ Score', fontsize=8)
    ax2.set_ylim(0, 1.1)
    ax2.set_title('R³ Score por ventana temporal (verde=coherente, rojo=incoherente)',
                  fontsize=9, color=PALETTE['text'])
    ax2.legend(fontsize=7, framealpha=0.2)
    ax2.grid(True, alpha=0.3)

    # Panel 3: métricas individuales
    ax3 = axes[2]; ax3.set_facecolor(PALETTE['surface'])
    if 'D2' in df.columns:
        ax3.plot(t, df['D2'].fillna(0), color=PALETTE['purple'],
                 lw=1, label='D₂', alpha=0.8)
    if 'LZ' in df.columns:
        ax3.plot(t, df['LZ'].fillna(0), color=PALETTE['orange'],
                 lw=1, label='C_LZ', alpha=0.8)
    if 'TE' in df.columns:
        ax3.plot(t, df['TE'].fillna(0)/df['TE'].max() if df['TE'].max()>0 else df['TE'],
                 color=PALETTE['accent'], lw=1, label='TE (norm)', alpha=0.8)
    ax3.set_xlabel('Tiempo (segundos)', fontsize=8)
    ax3.set_ylabel('Valor métrica', fontsize=8)
    ax3.set_title('Métricas individuales por ventana', fontsize=9, color=PALETTE['text'])
    ax3.legend(fontsize=7, framealpha=0.2)
    ax3.grid(True, alpha=0.3)

    add_watermark(fig)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = f'{OUTPUT_DIR}/timeline_{record_id}.png'
    fig.savefig(path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Guardado: {path}")


def plot_comparison(all_dfs):
    """Compara R³ entre registros."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=PALETTE['bg'])
    fig.suptitle('Comparación R³ entre registros MIT-BIH',
                 color=PALETTE['accent'], fontsize=12, fontweight='bold')

    for i, (rec_id, (df, desc)) in enumerate(all_dfs.items()):
        ax = axes[i]; ax.set_facecolor(PALETTE['surface'])

        # Distribución R³
        r3_vals = df['R3_score'].values
        unique, counts = np.unique(r3_vals, return_counts=True)
        colors = [PALETTE['green'] if v >= 0.75 else PALETTE['red'] for v in unique]
        ax.bar([str(v) for v in unique], counts, color=colors, alpha=0.85)

        coh_pct = df['coherente'].mean() * 100
        ax.set_title(f'Registro {rec_id}\n{desc[:30]}\nCoherente: {coh_pct:.0f}%',
                     fontsize=8, color=PALETTE['text'])
        ax.set_xlabel('R³ Score', fontsize=7)
        ax.set_ylabel('N° ventanas', fontsize=7)
        ax.grid(True, axis='y', alpha=0.3)

    add_watermark(fig)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = f'{OUTPUT_DIR}/comparison_r3.png'
    fig.savefig(path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  Guardado: {path}")


# ═══════════════════════════════════════════
# EJECUCIÓN PRINCIPAL
# ═══════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print(f"  MODE Pipeline — MIT-BIH Arrhythmia Database")
    print(f"  {AUTHOR}")
    print(f"  Ventana: {WINDOW_SIZE} muestras  |  Paso: {WINDOW_STEP}")
    print(f"  Max ventanas por registro: {MAX_WINDOWS}")
    print(f"{'='*60}")

    all_results = {}
    all_dfs     = {}

    for rec_id, desc in RECORDS.items():
        try:
            signal, fs, _ = read_mitbih(rec_id)
            df = analyze_record(rec_id, desc)

            if len(df) > 0:
                # Guardar CSV
                csv_path = f'{OUTPUT_DIR}/resultado_{rec_id}.csv'
                df.to_csv(csv_path, index=False)
                print(f"  CSV guardado: {csv_path}")

                # Gráfico de timeline
                plot_r3_timeline(df, rec_id, desc, signal, fs)

                all_dfs[rec_id] = (df, desc)
                all_results[rec_id] = df

        except Exception as e:
            print(f"  ERROR en registro {rec_id}: {e}")
            import traceback; traceback.print_exc()

    # Gráfico comparativo
    if len(all_dfs) > 1:
        plot_comparison(all_dfs)

    # Tabla resumen final
    print(f"\n{'='*60}")
    print(f"  RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"\n{'Registro':<12} {'Ventanas':>9} {'Coherentes':>12} {'R³ medio':>10} {'Régimen dominante'}")
    print("─" * 65)
    for rec_id, df in all_results.items():
        coh_n   = df['coherente'].sum()
        coh_tot = len(df)
        coh_pct = coh_n / coh_tot * 100 if coh_tot > 0 else 0
        r3_mean = df['R3_score'].mean()
        reg_dom = df['regimen'].mode()[0] if len(df) > 0 else 'n/a'
        print(f"  {rec_id:<10} {coh_tot:>9} {coh_n:>7} ({coh_pct:4.0f}%) "
              f"{r3_mean:>10.3f} {reg_dom}")

    print(f"\n  Resultados en carpeta: {OUTPUT_DIR}/")
    print(f"  {AUTHOR}")

if __name__ == '__main__':
    main()
