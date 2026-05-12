"""
MODE Attractor Pipeline — ECG Fix Edition v2.2
Autor: Emanuel Duarte — Pergamino, Argentina — 2026
Versión: v2.2 ECG_FIX | Corrección carga MIT-BIH con session_state
Streamlit 1.57 compatible · Arrow-safe · inotify-safe
"""

import warnings
warnings.filterwarnings("ignore")
import traceback, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import streamlit as st

# ── Import del pipeline ──────────────────────────────────────────
try:
    from pipeline import AttractorPipeline
except ImportError as _ie:
    st.error(f"❌ No se pudo importar pipeline.py: {_ie}")
    st.stop()

# ── Constantes ───────────────────────────────────────────────────
AUTHOR  = "Investigador Emanuel Duarte"
VERSION = "v2.2 ECG_FIX"

P = dict(
    bg       = "#080c10",
    surface  = "#0f1419",
    card     = "#141b23",
    border   = "#1e2a36",
    text     = "#cdd9e5",
    muted    = "#768390",
    accent   = "#4a9eff",
    green    = "#2dbe6c",
    red      = "#e5534b",
    orange   = "#d4a017",
    purple   = "#a371f7",
    teal     = "#2fb392",
)

plt.rcParams.update({
    "figure.facecolor": P["bg"],       "axes.facecolor": P["surface"],
    "axes.edgecolor":   P["border"],   "axes.labelcolor": P["text"],
    "xtick.color":      P["muted"],    "ytick.color":     P["muted"],
    "text.color":       P["text"],     "grid.color":      P["border"],
    "grid.alpha":        0.5,          "font.family":    "monospace",
    "font.size":         9,
    "axes.spines.top":  False,         "axes.spines.right": False,
})

# ── Señales demo ─────────────────────────────────────────────────
def lorenz_ts(n=1000):
    x, y, z = 1.0, 1.0, 1.05
    # warmup 2000 pasos descartados — necesario para desarrollar atractor
    for _ in range(2000):
        dx = 10*(y-x); dy = x*(28-z)-y; dz = x*y-(8/3)*z
        x += dx*.01; y += dy*.01; z += dz*.01
    o = []
    for _ in range(n):
        dx = 10*(y-x); dy = x*(28-z)-y; dz = x*y-(8/3)*z
        x += dx*.01; y += dy*.01; z += dz*.01
        o.append(x)
    return np.array(o)

def rossler_ts(n=1000):
    x, y, z = 1.0, 0.0, 0.0
    o = []
    for _ in range(n):
        dx = -y-z; dy = x+0.2*y; dz = 0.2+z*(x-5.7)
        x += dx*.05; y += dy*.05; z += dz*.05
        o.append(x)
    return np.array(o)


def _logistic_map(N=1000, r=3.9):
    x = 0.1
    out = [x]
    for _ in range(N - 1):
        x = r * x * (1 - x)
        out.append(x)
    return np.array(out)

DEMOS = {
    "Lorenz (caótico clásico)":           lorenz_ts,
    "Rössler (caos débil)":               rossler_ts,
    "Mapa Logístico r=3.9 (caótico)":     lambda n=1000: _logistic_map(n, r=3.9),
    "Mapa Logístico r=3.5 (periódico)":   lambda n=1000: _logistic_map(n, r=3.5),
    "Mapa Logístico r=3.7 (transición)":  lambda n=1000: _logistic_map(n, r=3.7),
    "Senoidal ruido bajo":                lambda n=1000: np.sin(2*np.pi*0.05*np.arange(n)) + 0.05*np.random.default_rng(0).normal(size=n),
    "Senoidal ruido alto":                lambda n=1000: np.sin(2*np.pi*0.05*np.arange(n)) + 0.5 *np.random.default_rng(1).normal(size=n),
    "Ruido blanco":                       lambda n=1000: np.random.default_rng(2).normal(size=n),
    "Ruido rosa (1/f)":                   lambda n=1000: np.cumsum(np.random.default_rng(3).normal(size=n)),
}

# ── Utilidades ───────────────────────────────────────────────────
def fmt(v, decimals=8):
    if v is None: return "N/A"
    try:
        f = float(v)
        if np.isnan(f): return "NaN"
        return f"{f:.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)

def watermark(fig):
    fig.text(0.5, 0.5, AUTHOR, fontsize=10, color="white", alpha=0.07,
             ha="center", va="center", rotation=28, transform=fig.transFigure)
    fig.text(0.99, 0.01, AUTHOR, fontsize=6.5, color="white", alpha=0.25,
             ha="right", va="bottom", transform=fig.transFigure)

def topng(fig, dpi=115):
    watermark(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0); plt.close(fig)
    return buf.read()

# ── Figuras ──────────────────────────────────────────────────────
def compute_baselines(x, result):
    fv   = np.abs(np.fft.rfft(x - x.mean()))
    psd  = fv**2; psd /= (psd.sum() + 1e-12)
    hspec = float(-np.sum(psd[psd>0] * np.log2(psd[psd>0])) / np.log2(max(len(psd),2)))
    hist, _ = np.histogram(x, bins=32, density=True)
    hn   = hist / (hist.sum() + 1e-12)
    hshan = float(-np.sum(hn[hn>0] * np.log2(hn[hn>0])) / np.log2(32))
    d2raw = result["metrics"].get("D2")
    d2    = float(d2raw) if d2raw is not None and not np.isnan(float(d2raw)) else 0.0
    return hspec, hshan, d2

def fig_signal(x, label):
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8), facecolor=P["bg"])
    ax = axes[0]; ax.set_facecolor(P["surface"])
    ax.plot(x[-600:], lw=0.65, color=P["accent"], alpha=0.92)
    ax.fill_between(range(min(600, len(x))), x[-600:], alpha=0.07, color=P["accent"])
    ax.set_xlabel("t"); ax.set_ylabel("x(t)")
    ax.set_title(str(label)[:45], fontsize=8, color=P["accent"]); ax.grid(True, alpha=0.25)
    ax2 = axes[1]; ax2.set_facecolor(P["surface"])
    fv = np.abs(np.fft.rfft(x - x.mean())); fr = np.fft.rfftfreq(len(x))
    ax2.semilogy(fr[1:], fv[1:], color=P["purple"], lw=0.65, alpha=0.88)
    ax2.fill_between(fr[1:], fv[1:], alpha=0.06, color=P["purple"])
    ax2.set_xlabel("Frecuencia"); ax2.set_ylabel("FFT")
    ax2.set_title("Espectro de potencia", fontsize=8, color=P["accent"]); ax2.grid(True, alpha=0.25)
    fig.tight_layout(); return fig

def fig_epsilon(result):
    eps = result["epsilon_series"]
    fig, ax = plt.subplots(figsize=(10, 2.4), facecolor=P["bg"])
    ax.set_facecolor(P["surface"])
    t = np.arange(len(eps))
    ax.fill_between(t, eps, alpha=0.18, color=P["teal"])
    ax.plot(t, eps, color=P["teal"], lw=0.75, alpha=0.9)
    med = np.median(eps)
    ax.axhline(med, color=P["orange"], lw=1.2, ls="--", label=f"ε={med:.8f}")
    ax.set_xlabel("t (índice embedding)"); ax.set_ylabel("ε(t)")
    ax.set_title("ε dinámico — escala local del sistema", fontsize=9, color=P["accent"])
    ax.legend(fontsize=7, framealpha=0.15); ax.grid(True, alpha=0.25)
    fig.tight_layout(); return fig

def fig_attractor(result):
    Y   = result["embedding"]
    eps = result["epsilon_series"]
    if Y.shape[1] < 3: return None
    norm = Normalize(vmin=eps.min(), vmax=eps.max()); cmap = plt.cm.plasma
    fig  = plt.figure(figsize=(6, 5.2), facecolor=P["bg"])
    ax   = fig.add_subplot(111, projection="3d", facecolor=P["surface"])
    n    = len(Y); step = max(1, n//2000)
    Ys   = Y[::step]; es = eps[n%step::step][:len(Ys)]
    ax.scatter(Ys[:,0], Ys[:,1], Ys[:,2], c=es, cmap=cmap, norm=norm, s=1.0, alpha=0.75)
    ax.set_xlabel("y(t)"); ax.set_ylabel(f"y(t-τ)"); ax.set_zlabel(f"y(t-2τ)")
    ax.set_title(f"τ={result['tau']}  ε={result['epsilon']:.8f}",
                 color=P["accent"], fontsize=9, pad=8)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False; pane.set_edgecolor(P["border"])
    ax.tick_params(colors=P["border"], labelsize=5)
    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.1)
    cb.ax.tick_params(labelsize=5, colors=P["text"]); cb.set_label("ε(t)", color=P["text"], fontsize=6)
    fig.tight_layout(); return fig

def fig_metrics(result):
    r3  = result["R3"]
    sm  = r3["stability_map"]
    delta = r3["delta"]
    all_keys   = ["lambda", "D2", "LZ", "TE", "SampEn"]
    key_labels = {"lambda": "λ Lyapunov", "D2": "D₂ Dim.Corr.",
                  "LZ": "CLZ Compl.", "TE": "TE Trans.Ent.", "SampEn": "SampEn Muestra"}
    present = [k for k in all_keys if k in sm]
    labels  = [key_labels[k] for k in present]
    grads   = [sm[k].get("gradient", 0.0) for k in present]
    stable  = [sm[k].get("stable", False) for k in present]
    colors  = [P["green"] if s else P["red"] for s in stable]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), facecolor=P["bg"])
    ax = axes[0]; ax.set_facecolor(P["surface"])
    bars = ax.barh(labels, grads, color=colors, alpha=0.85, height=0.52)
    ax.axvline(delta, color=P["orange"], lw=1.8, ls="--", label=f"δ={delta} ({r3['regime']})")
    ax.set_xlabel("∂μ/∂τ RMS‑μ (gradiente normalizado)", fontsize=7)
    ax.set_title("Sensibilidad a τ por métrica", fontsize=9, color=P["accent"])
    ax.legend(fontsize=7, framealpha=0.15); ax.grid(True, axis="x", alpha=0.25)
    mx = max(grads) if grads and max(grads) > 0 else 0.01
    for bar, g in zip(bars, grads):
        ax.text(g + mx*0.03, bar.get_y() + bar.get_height()/2,
                f"{g:.7f}", va="center", ha="left", fontsize=6, color=P["text"])

    ax2 = axes[1]; ax2.set_facecolor(P["surface"])
    theta = np.linspace(-np.pi, 0, 300)
    for i in range(len(theta)-1):
        c = plt.cm.RdYlGn(i/(len(theta)-1))
        ax2.fill_between([np.cos(theta[i]), np.cos(theta[i+1])],
                         [np.sin(theta[i])*0.68, np.sin(theta[i+1])*0.68],
                         [np.sin(theta[i])*1.0,  np.sin(theta[i+1])*1.0],
                         color=c, alpha=0.88)
    score = float(r3["R3_score"])
    angle = np.pi * (1 - min(max(score, 0), 1))
    ax2.annotate("", xy=(0.83*np.cos(angle), 0.83*np.sin(angle)), xytext=(0,0),
                 arrowprops=dict(arrowstyle="-|>", color=P["text"], lw=2.8, zorder=5))
    ax2.plot(0, 0, "o", color=P["text"], ms=6, zorder=6)
    cohc = P["green"] if r3["coherent"] else P["red"]
    coht = "COHERENTE" if r3["coherent"] else "NO COHERENTE"
    ax2.text(0, -0.20, f"R3 score: {score:.8f}", ha="center", fontsize=13, fontweight="bold", color=P["accent"])
    ax2.text(0, -0.44, coht,              ha="center", fontsize=10, color=cohc)
    ax2.text(0, -0.64, r3["regime_desc"], ha="center", fontsize=8,  color=P["orange"])
    ax2.text(-1.04, -0.1, "0", ha="center", fontsize=8, color=P["red"])
    ax2.text( 1.04, -0.1, "1", ha="center", fontsize=8, color=P["green"])
    ax2.text(0, 1.07, "0.5", ha="center", fontsize=8, color=P["orange"])
    ax2.set_xlim(-1.28, 1.28); ax2.set_ylim(-0.88, 1.22)
    ax2.set_aspect("equal"); ax2.axis("off")
    ax2.set_title("R3 Score — co-estabilización CONTINUO", fontsize=9, color=P["accent"])
    fig.tight_layout(pad=1.5); return fig

def fig_baselines(x, result):
    hspec, hshan, d2 = compute_baselines(x, result)
    r3score   = float(result["R3"]["R3_score"])
    coherent  = result["R3"]["coherent"]
    fig, ax   = plt.subplots(figsize=(8, 3.2), facecolor=P["bg"])
    ax.set_facecolor(P["surface"])
    methods = ["H Fourier", "H Shannon", "D₂ norm", "R3"]
    vals    = [hspec, hshan, min(d2/3.0, 1.0), r3score]
    barc    = [P["muted"], P["muted"], P["muted"], P["green"] if coherent else P["red"]]
    bars    = ax.bar(methods, vals, color=barc, alpha=0.85, width=0.55)
    ax.axhline(0.60, color=P["orange"], lw=1.2, ls="--", alpha=0.7, label="Umbral R3 ≥ 0.60")
    ax.set_ylim(0, 1.15); ax.set_ylabel("Valor normalizado", fontsize=8)
    ax.set_title("Comparación baselines vs R3", fontsize=9, color=P["accent"])
    ax.legend(fontsize=7, framealpha=0.15); ax.grid(True, axis="y", alpha=0.25)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.7f}",
                ha="center", fontsize=7, color=P["text"])
    fig.tight_layout(); return fig

def fig_windowed(sig, win_results):
    if not win_results: return None
    tv  = [r["ts"]        for r in win_results]
    r3v = [float(r["R3"]) for r in win_results]
    cohv= [bool(r["coherente"]) for r in win_results]
    tauv= [int(r["tau"])  for r in win_results]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), facecolor=P["bg"])
    fig.suptitle("Análisis por ventanas temporales", color=P["accent"], fontsize=11, fontweight="bold")
    ax1 = axes[0]; ax1.set_facecolor(P["surface"])
    ax1.plot(sig[-min(3000, len(sig)):], lw=0.5, color=P["accent"], alpha=0.9)
    ax1.set_ylabel("Amplitud"); ax1.set_title("Señal", fontsize=8, color=P["text"]); ax1.grid(True, alpha=0.2)
    ax2 = axes[1]; ax2.set_facecolor(P["surface"])
    bc  = [P["green"] if c else P["red"] for c in cohv]
    w   = max(tv[1]-tv[0], 0.5)*0.8 if len(tv)>1 else 0.5
    ax2.bar(tv, r3v, width=w, color=bc, alpha=0.85)
    ax2.axhline(0.60, color=P["orange"], lw=1.2, ls="--", alpha=0.8)
    ax2.set_ylim(0, 1.15); ax2.set_ylabel("R3")
    ax2.set_title("R3 por ventana (verde=coherente, rojo=incoherente)", fontsize=8, color=P["text"]); ax2.grid(True, alpha=0.2)
    ax3 = axes[2]; ax3.set_facecolor(P["surface"])
    ax3.plot(tv, tauv, color=P["purple"], lw=1.2, marker="o", ms=3, alpha=0.9)
    ax3.fill_between(tv, tauv, alpha=0.1, color=P["purple"])
    ax3.set_xlabel("t (índice)"); ax3.set_ylabel("τ")
    ax3.set_title("Evolución de τ semidinamico", fontsize=8, color=P["text"]); ax3.grid(True, alpha=0.2)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); return fig

# ── Lector MIT-BIH ───────────────────────────────────────────────
def read_mitbih_bytes(datfile, heafile):
    lines = heafile.read().decode("latin-1").strip().split("\n")
    hdr   = lines[0].split(); fs, n = int(hdr[2]), int(hdr[3])
    si    = lines[1].split(); gain, bl = float(si[2]), int(si[4])
    raw   = datfile.read()
    s     = []
    i     = 0
    while i + 2 < len(raw):
        b0, b1, b2 = raw[i], raw[i+1], raw[i+2]
        s1 = b0 | ((b1 & 0x0F) << 8); s1 = s1 - 4096 if s1 >= 2048 else s1
        s2 = b2 | ((b1 & 0xF0) << 4); s2 = s2 - 4096 if s2 >= 2048 else s2
        s.extend([s1, s2]); i += 3
    return np.array(s[:n], dtype=float) - bl, fs, n

# ── Main ─────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="MODE Attractor Pipeline",
        page_icon="🌀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'Space Grotesk', sans-serif; background: {P["bg"]}; color: {P["text"]}; }}
    [data-testid="stAppViewContainer"] {{ background: {P["bg"]}; }}
    [data-testid="stSidebar"] {{ background: {P["surface"]}; border-right: 1px solid {P["border"]}; }}
    [data-testid="stSidebar"] * {{ color: {P["text"]} !important; }}
    .stButton>button {{ background: linear-gradient(135deg,#1a4a7a,#0d2d52); color: #4a9eff;
        border: 1px solid #1e3a5a; border-radius: 6px; font-family: 'JetBrains Mono',monospace;
        font-weight: 600; letter-spacing: .05em; }}
    h1,h2,h3 {{ font-family: 'Space Grotesk',sans-serif; color: {P["accent"]}; }}
    [data-baseweb="tab"] {{ font-family: 'JetBrains Mono',monospace; color: {P["muted"]}; }}
    [aria-selected="true"] {{ color: {P["accent"]} !important; border-bottom: 2px solid {P["accent"]} !important; }}
    </style>""", unsafe_allow_html=True)

    # Header
    colt, coli = st.columns([3, 1])
    with colt:
        st.markdown("**MODE Pipeline**")
        st.markdown("H1 ε dinámico · H2 τ semidinamico · H3 R3 co-estabilización")
    with coli:
        st.markdown(f"""<div style="text-align:right;padding-top:12px;color:{P["muted"]};
        font-family:'JetBrains Mono',monospace;font-size:.74rem">{AUTHOR}<br>{VERSION}</div>""",
        unsafe_allow_html=True)
    st.divider()

    # ── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("**Configuración**"); st.divider()
        modo = st.radio("Modo", ["Señal sintética", "Cargar CSV", "ECG MIT-BIH", "Ventanas temporales"])
        st.divider()
        st.markdown("**Pipeline**")
        mdim    = st.slider("Dimensión embedding m", 2, 6, 3)
        maxtau  = st.slider("τ máximo AMI", 10, 80, 40, 5)

        xdata = np.array([]); label = "sin datos"; extra = {}

        if modo == "Señal sintética":
            st.divider(); st.markdown("**Señal demo**")
            demoname = st.selectbox("Seleccionar", list(DEMOS.keys()))
            N        = st.slider("N muestras", 300, 3000, 1000, 100)
            xdata    = DEMOS[demoname](N); label = demoname

        elif modo == "Cargar CSV":
            st.divider(); st.markdown("**Archivo**")
            upcsv = st.file_uploader("CSV (una columna)", type=["csv", "txt"])
            if upcsv is not None:
                xdata = pd.read_csv(upcsv, header=None).iloc[:,0].dropna().values.astype(float)
                label = upcsv.name

        elif modo == "ECG MIT-BIH":
            st.divider(); st.markdown("**Archivos ECG**")
            st.caption("Subir el .dat y el .hea del mismo registro")
            # Inicializar session_state para los archivos ECG
            if "ecg_files_loaded" not in st.session_state:
                st.session_state.ecg_files_loaded = False
            if "ecg_sigfull" not in st.session_state:
                st.session_state.ecg_sigfull = None
            if "ecg_fsecg" not in st.session_state:
                st.session_state.ecg_fsecg = None
            if "ecg_nsamples" not in st.session_state:
                st.session_state.ecg_nsamples = None

            updat = st.file_uploader("Archivo .dat (MIT-BIH format)", type=["dat", "bin"], key="ecg_dat_file")
            uphea = st.file_uploader("Archivo .hea (header)", type=["hea", "txt"], key="ecg_hea_file")

            # Leer los archivos cuando ambos están cargados
            if updat is not None and uphea is not None and not st.session_state.ecg_files_loaded:
                try:
                    sigfull, fsecg, nsamples = read_mitbih_bytes(updat, uphea)
                    # Guardar en session_state
                    st.session_state.ecg_sigfull = sigfull
                    st.session_state.ecg_fsecg = fsecg
                    st.session_state.ecg_nsamples = nsamples
                    st.session_state.ecg_files_loaded = True
                    st.success(f"ECG cargado: {nsamples} muestras @ {fsecg} Hz ({nsamples/fsecg:.1f}s)")
                except Exception as e:
                    st.error(f"Error leyendo ECG: {e}")
                    with st.expander("Detalles del error"):
                        st.code(traceback.format_exc())
                    st.session_state.ecg_files_loaded = False

            # Sliders con límites basados en datos cargados
            if st.session_state.ecg_sigfull is not None:
                max_time = int(st.session_state.ecg_nsamples / st.session_state.ecg_fsecg)
                ecgstart = st.slider("Inicio (s)", 0, max(0, max_time-5), 0, 5, key="ecg_start_slider")
                ecgdur = st.slider("Duración (s)", 5, min(60, max_time), 10, 5, key="ecg_dur_slider")

                start_idx = int(ecgstart * st.session_state.ecg_fsecg)
                end_idx = min(int((ecgstart + ecgdur) * st.session_state.ecg_fsecg), len(st.session_state.ecg_sigfull))

                if start_idx < len(st.session_state.ecg_sigfull):
                    xdata = st.session_state.ecg_sigfull[start_idx:end_idx]
                    label = f"ECG {updat.name} t={ecgstart:.0f}-{ecgstart+ecgdur:.0f}s"
                    extra = {"fs": st.session_state.ecg_fsecg}
            else:
                # Deshabilitar sliders si no hay datos
                st.slider("Inicio (s)", 0, 600, 0, 5, disabled=True, key="ecg_start_disabled")
                st.slider("Duración (s)", 5, 60, 10, 5, disabled=True, key="ecg_dur_disabled")
                st.info("Cargar ambos archivos (.dat y .hea) para activar el análisis")

        elif modo == "Ventanas temporales":
            st.divider(); st.markdown("**Fuente**")
            wsrc = st.radio("", ["Demo", "CSV", "ECG MIT-BIH"], key="wsrc")

            if wsrc == "Demo":
                wdemo = st.selectbox("Señal", list(DEMOS.keys()), key="wdm")
                wN    = st.slider("N total", 1000, 5000, 2000, 100, key="wN")
                xdata = DEMOS[wdemo](wN); label = wdemo

            elif wsrc == "CSV":
                wcsv = st.file_uploader("CSV", type=["csv","txt"], key="wcsv")
                if wcsv is not None:
                    xdata = pd.read_csv(wcsv, header=None).iloc[:,0].dropna().values.astype(float)
                    label = wcsv.name

            else:  # ECG MIT-BIH
                st.caption("Subí el .dat y el .hea del mismo registro")
                # Inicializar session_state para ventanas ECG
                if "wecg_loaded" not in st.session_state:
                    st.session_state.wecg_loaded  = False
                    st.session_state.wecg_sig     = None
                    st.session_state.wecg_fs      = None
                    st.session_state.wecg_label   = ""

                wdat = st.file_uploader("Archivo .dat", type=["dat","bin"], key="wecg_dat")
                whea = st.file_uploader("Archivo .hea", type=["hea","txt"], key="wecg_hea")

                if wdat is not None and whea is not None and not st.session_state.wecg_loaded:
                    try:
                        wsig, wfs, wn = read_mitbih_bytes(wdat, whea)
                        st.session_state.wecg_sig    = wsig
                        st.session_state.wecg_fs     = wfs
                        st.session_state.wecg_label  = wdat.name
                        st.session_state.wecg_loaded = True
                        st.success(f"ECG cargado: {wn} muestras @ {wfs} Hz ({wn/wfs/60:.1f} min)")
                    except Exception as e:
                        st.error(f"Error leyendo ECG: {e}")
                        st.session_state.wecg_loaded = False

                if wdat is None or whea is None:
                    st.session_state.wecg_loaded = False
                    st.session_state.wecg_sig    = None

                if st.session_state.wecg_loaded and st.session_state.wecg_sig is not None:
                    xdata = st.session_state.wecg_sig
                    label = f"ECG {st.session_state.wecg_label}"
                    st.info(f"Señal lista: {len(xdata):,} muestras")

            st.divider(); st.markdown("**Ventanas**")
            winsize  = st.slider("Tamaño (muestras)", 200, 2000, 1000, 100)
            winstep  = st.slider("Paso (muestras)",   100, 1000,  500,  50)
            winmax   = st.slider("Máx. ventanas",      5,   50,   20,    5)
            extra    = {"winsize": winsize, "winstep": winstep, "winmax": winmax}

        st.divider()
        run_btn = st.button("⚡ Ejecutar pipeline", type="primary", use_container_width=True)
        st.divider()
        st.markdown(f"""<div style="font-family:'JetBrains Mono',monospace;font-size:.72rem;color:{P["muted"]}">
        <b style="color:{P["text"]}">δ por régimen</b><br><br>
        Estable &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0.06<br>
        Caos débil &nbsp;&nbsp; 0.05<br>
        Caótico &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0.08<br>
        Hipercaótico 0.15<br>
        Ruidoso &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0.20<br><br>
        <b style="color:{P["text"]}">Umbral coherencia</b><br>
        R3 ≥ 0.60 → continuo</div>""", unsafe_allow_html=True)

    # ── Ejecución ────────────────────────────────────────────────
    if run_btn:
        if len(xdata) == 0:
            st.warning("⚠ Cargar datos primero."); st.stop()

        if modo == "Ventanas temporales":
            ws   = extra.get("winsize", 1000)
            wst  = extra.get("winstep", 500)
            wmax = extra.get("winmax",  20)
            with st.spinner("Analizando ventanas..."):
                pipe  = AttractorPipeline(m=mdim, max_tau=maxtau, verbose=False)
                winrs = []
                prog  = st.progress(0); nw = 0
                for start in range(0, len(xdata)-ws, wst):
                    if nw >= wmax: break
                    w = xdata[start: start+ws]
                    if w.std() < 1e-6: continue
                    try:
                        r   = pipe.run(w, label=f"w{nw}")
                        r3  = r["R3"]
                        winrs.append({
                            "ts":        int(start),
                            "tau":       int(r["tau"]),
                            "epsilon":   float(r["epsilon"]),
                            "R3":        float(r3["R3_score"]),
                            "coherente": bool(r3["coherent"]),
                            "regimen":   str(r3["regime"]),
                            "delta":     float(r3["delta"]),
                        })
                    except Exception:
                        pass
                    nw += 1
                    prog.progress(min(nw/wmax, 1.0))
                prog.empty()
            st.session_state.update(mode=modo, x=xdata, label=label,
                                    win_results=winrs, result=None)
        else:
            with st.spinner("Calculando ε, τ, métricas, R3..."):
                try:
                    pipe = AttractorPipeline(m=mdim, max_tau=maxtau, verbose=False)
                    r    = pipe.run(xdata, label=label)
                    st.session_state.update(mode=modo, x=xdata, label=label,
                                            result=r, win_results=None)
                except Exception as e:
                    st.error(f"Error en pipeline: {e}")
                    st.code(traceback.format_exc()); st.stop()

    # ── Pantalla inicial ─────────────────────────────────────────
    if "mode" not in st.session_state:
        st.markdown(f"""<div style="text-align:center;padding:60px 20px">
        <div style="font-size:3.5rem;margin-bottom:16px">🌀</div>
        <h2 style="color:{P["accent"]};font-family:'JetBrains Mono',monospace">MODE Pipeline {VERSION}</h2>
        <p style="color:{P["muted"]};font-size:.95rem;max-width:680px;margin:0 auto;line-height:1.7">
        Framework de legibilidad observacional para sistemas dinámicos no lineales.<br>
        <b>Revisión 2026-05</b>: Cálculos HONESTOS · Gradientes RMS · R3 continuo · Arrow-safe<br>
        Seleccionar un modo en el sidebar y presionar <b>Ejecutar pipeline</b>.</p><br>
        <p style="color:{P["border"]};font-family:'JetBrains Mono',monospace;font-size:.75rem">
        {AUTHOR} &nbsp;·&nbsp; {VERSION}</p></div>""", unsafe_allow_html=True)
        st.stop()

    modo_act = st.session_state.get("mode", "")

    # ── Modo ventanas ────────────────────────────────────────────
    if modo_act == "Ventanas temporales":
        winrs = st.session_state.get("win_results", [])
        sigw  = st.session_state.get("x", np.array([]))
        if not winrs:
            st.info("Presionar Ejecutar pipeline para analizar."); st.stop()

        cohn   = sum(1 for r in winrs if r["coherente"])
        r3vals = [r["R3"] for r in winrs]
        r3med  = float(np.mean(r3vals))
        r3std  = float(np.std(r3vals))
        r3min  = float(np.min(r3vals))
        r3max  = float(np.max(r3vals))

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Ventanas",    len(winrs))
        c2.metric("R³ media",    f"{r3med:.6f}")
        c3.metric("R³ std",      f"{r3std:.6f}", help="Variabilidad de R³ — clave en sistemas fisiológicos")
        c4.metric("R³ min/max",  f"{r3min:.3f}/{r3max:.3f}")
        c5.metric("Coherentes",  f"{cohn}/{len(winrs)} ({cohn/len(winrs)*100:.1f}%)")
        c6.metric("Señal",       str(st.session_state.get("label",""))[:20])
        st.divider()
        fw = fig_windowed(sigw, winrs)
        if fw: st.image(topng(fw), use_container_width=True)
        st.divider(); st.subheader("Resultados por ventana")
        dfw = pd.DataFrame(winrs)
        for col in dfw.columns:
            if dfw[col].dtype == object:
                dfw[col] = dfw[col].astype(str)
        st.dataframe(dfw, use_container_width=True, hide_index=True)

        # Estadísticas R³ — el biomarcador temporal
        st.divider()
        st.subheader("Estadísticas R³ — biomarcador temporal")
        stats_rows = [
            ("R³ media",    f"{r3med:.8f}", "Nivel base de coherencia"),
            ("R³ std",      f"{r3std:.8f}", "Variabilidad → salud fisiológica"),
            ("R³ mínimo",   f"{r3min:.8f}", "Zona de menor legibilidad"),
            ("R³ máximo",   f"{r3max:.8f}", "Zona de mayor legibilidad"),
            ("R³ rango",    f"{r3max-r3min:.8f}", "Amplitud dinámica"),
            ("% coherentes",f"{cohn/len(winrs)*100:.2f}%", "Ventanas sobre umbral 0.60"),
        ]
        df_stats = pd.DataFrame(stats_rows, columns=["Estadística","Valor","Interpretación"])
        df_stats = df_stats.astype(str)
        st.dataframe(df_stats, use_container_width=True, hide_index=True)

        ce1, ce2 = st.columns(2)
        with ce1:
            st.download_button("Descargar CSV ventanas", dfw.to_csv(index=False).encode(),
                               "ventanas.csv", mime="text/csv")
        with ce2:
            st.download_button("Descargar CSV estadísticas R³",
                               df_stats.to_csv(index=False).encode(),
                               "estadisticas_r3.csv", mime="text/csv")
        st.stop()

    # ── Modos normales ───────────────────────────────────────────
    result = st.session_state.get("result")
    x      = st.session_state.get("x", np.array([]))
    label  = st.session_state.get("label", "")
    if result is None:
        st.info("Presionar Ejecutar pipeline"); st.stop()

    r3      = result["R3"]
    mvals   = result["metrics"]
    r3score = float(r3["R3_score"])

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("τ semidinamico",  str(result["tau"]))
    c2.metric("ε (mediana)",     f"{result['epsilon']:.8f}")
    c3.metric("R3 Score",        f"{r3score:.8f}",
              delta="coherente" if r3["coherent"] else "no coherente")
    c4.metric("Régimen",         str(r3["regime"]))
    c5.metric("δ activo",        str(r3["delta"]))
    c6.metric("N",               str(len(x)))
    st.divider()

    tab1,tab2,tab3,tab4,tab5 = st.tabs(
        ["🔵 Señal · ε", "🟣 Atractor", "🟠 Métricas R3", "🟢 Baselines", "📥 Export"])

    with tab1:
        st.subheader("Serie temporal y espectro")
        st.image(topng(fig_signal(x, label)), use_container_width=True)
        st.subheader("ε(t) dinámico")
        st.image(topng(fig_epsilon(result)), use_container_width=True)

    with tab2:
        fa = fig_attractor(result)
        if fa:
            cola, colb = st.columns([2,1])
            with cola:
                st.subheader("Atractor reconstruido")
                st.image(topng(fa), use_container_width=True)
            with colb:
                st.markdown(f"""<div style="background:{P["card"]};border:1px solid {P["border"]};
                border-radius:10px;padding:20px;margin-top:40px;font-family:'JetBrains Mono',monospace;font-size:.82rem">
                <div style="color:{P["muted"]};margin-bottom:8px">EMBEDDING</div>
                τ <b style="color:{P["accent"]}">{result["tau"]}</b><br>
                m <b style="color:{P["accent"]}">{result["embedding"].shape[1]}</b><br>
                ε <b style="color:{P["teal"]}">{result["epsilon"]:.8f}</b><br><br>
                <div style="color:{P["muted"]};margin-bottom:8px">RÉGIMEN</div>
                <div style="color:{P["orange"]}">{r3["regime_desc"]}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Se necesita m ≥ 3 para el atractor 3D.")

    with tab3:
        st.subheader("Sensibilidad a τ y R3 gauge")
        st.image(topng(fig_metrics(result)), use_container_width=True)
        st.divider(); st.subheader("Detalle por métrica (valores sin truncado)")
        smmap = r3["stability_map"]
        key_labels = {"lambda": "λ Lyapunov", "D2": "D₂ Corr.",
                      "LZ": "CLZ Compl.", "TE": "TE Transf.", "SampEn": "SampEn Muestra"}
        cols_m = st.columns(min(len(smmap), 5))
        for i, (k, vd) in enumerate(smmap.items()):
            with cols_m[i % len(cols_m)]:
                col = P["green"] if vd.get("stable", False) else P["red"]
                gradval   = float(vd.get("gradient", 0))
                weightval = float(vd.get("weight",   0))
                st.markdown(f"""<div style="background:{P["card"]};border:1px solid {col}40;
                border-radius:10px;padding:14px;text-align:center">
                <div style="color:{P["muted"]};font-size:.72rem;margin-bottom:6px">{key_labels.get(k,k)}</div>
                <div style="font-size:1.3rem">{"✔" if vd.get("stable",False) else "✘"}</div>
                <div style="color:{P["text"]};font-family:'JetBrains Mono',monospace;font-size:.70rem;margin-top:6px">
                grad: {gradval:.8f}<br>weight: {weightval:.8f}<br>
                <span style="color:{P["muted"]}">δ={vd.get("delta",0)}</span></div></div>""",
                unsafe_allow_html=True)

    with tab4:
        st.subheader("Comparación con métodos estándar")
        st.image(topng(fig_baselines(x, result)), use_container_width=True)
        st.divider()
        hspec, hshan, d2 = compute_baselines(x, result)
        bl_rows = [
            ["H Fourier (espectral)",  fmt(hspec,7),   "≤1 plano",         "No distingue caos-ruido"],
            ["H Shannon (señal)",      fmt(hshan,7),   "≤1 máx desor",     "Sin info temporal"],
            ["D₂ clásico",             fmt(d2,7),      "Lorenz≈2.05",      "Escalar puro"],
            ["R3 Score",               fmt(r3score,8), "≥0.60→coherente",  "Continuo+régimen"],
            ["Coherente",              "SÍ" if r3["coherent"] else "NO", "---", "---"],
            ["Régimen",                str(r3["regime_desc"]), "---",       "δ semidinamico"],
        ]
        dfbl = pd.DataFrame(bl_rows, columns=["Método","Valor","Ref","Nota"]).astype(str)
        st.dataframe(dfbl, use_container_width=True, hide_index=True)

    with tab5:
        st.subheader("Tabla completa (valores sin truncado ni redondeo)")
        rows = [
            ["τ semidinamico",    str(result["tau"]),                   "---"],
            ["τ inicial",         str(result.get("tau_initial","---")), "---"],
            ["ε mediana",         fmt(result["epsilon"],8),             "---"],
            ["Régimen",           str(r3["regime_desc"]),               "---"],
            ["δ activo",          str(r3["delta"]),                     "---"],
            ["R3 Score",          fmt(r3score,8),                       "≥0.60 → continuo"],
            ["Coherente",         "SÍ" if r3["coherent"] else "NO",     "---"],
            ["n válidas",         str(r3.get("n_valid","---")),          "métricas válidas"],
            ["λ Lyapunov",        fmt(mvals.get("lambda"),8),           "<0→estable, >0→caos"],
            ["D₂ dim. corr.",     fmt(mvals.get("D2"),8),               "Lorenz≈2.05"],
            ["CLZ compl.",        fmt(mvals.get("LZ"),8),               "0=orden, 1=compl"],
            ["TE trans. ent.",    fmt(mvals.get("TE"),8),               "Flujo inf"],
            ["SampEn muestra",    fmt(mvals.get("SampEn"),8),           "Richman & Moorman 2000"],
        ]
        dfres = pd.DataFrame(rows, columns=["Variable","Valor","Referencia"]).astype(str)
        st.dataframe(dfres, use_container_width=True, hide_index=True)
        st.divider()

        ce1, ce2, ce3 = st.columns(3)
        with ce1:
            st.download_button("📄 Resultados CSV", dfres.to_csv(index=False).encode(),
                               "resultados.csv", mime="text/csv")
        with ce2:
            emb  = result["embedding"]
            cols = [f"y(t-{i*result['tau']})" for i in range(emb.shape[1])]
            dfemb = pd.DataFrame(emb, columns=cols)
            dfemb["epsilon"] = result["epsilon_series"][:len(dfemb)]
            st.download_button("📊 Embedding CSV", dfemb.to_csv(index=False).encode(),
                               "embedding.csv", mime="text/csv")
        with ce3:
            hspec2, hshan2, d22 = compute_baselines(x, result)
            dfbl2 = pd.DataFrame({
                "metrica": ["H_Fourier","H_Shannon","D2","R3","coherente","regimen"],
                "valor":   [fmt(hspec2,8), fmt(hshan2,8), fmt(d22,8),
                            fmt(r3score,8), str(r3["coherent"]), str(r3["regime"])],
            })
            st.download_button("📈 Baselines CSV", dfbl2.to_csv(index=False).encode(),
                               "baselines.csv", mime="text/csv")

        st.markdown(f"""<div style="text-align:center;padding:10px;margin-top:6px;
        color:{P["border"]};font-family:'JetBrains Mono',monospace;font-size:.68rem">
        {AUTHOR} &nbsp;·&nbsp; {VERSION}<br>
        Gradientes RMS normalizados · R3 continuo ponderado · Precisión 8 decimales</div>""",
        unsafe_allow_html=True)

if __name__ == "__main__":
    main()
