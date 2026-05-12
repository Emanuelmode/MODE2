"""
result_saver.py
═══════════════════════════════════════════════════════════════
Módulo de persistencia de resultados para MODE Attractor Pipeline.
Guarda automáticamente cada análisis etiquetado según el archivo
de origen, con marca de agua de authorship.

Uso:
    from result_saver import save_result

    result = pipe.run(signal, label=filename)
    save_result(result, label=filename)

Autor: Investigador/dueño intelectual Emanuel Duarte
═══════════════════════════════════════════════════════════════
"""

import json
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# ── Configuración global ─────────────────────────────────────────
AUTHOR = "Investigador/dueño intelectual Emanuel Duarte"
RESULTS_DIR = "resultados_autosave"
HISTORY_JSONL = "history.jsonl"
INDEX_CSV = "index.csv"


# ── Función de sanitización recursiva ──────────────────────────
def _sanitize_value(obj):
    """Convierte cualquier objeto a JSON serializable recursivamente."""
    import numpy as np

    # None
    if obj is None:
        return None

    # Numpy bool
    type_name = type(obj).__name__
    if isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    if type_name == 'bool_':
        return bool(obj)

    # Numpy integers
    if isinstance(obj, np.integer):
        return int(obj)
    if 'int' in type_name and not isinstance(obj, bool):
        return int(obj)

    # Numpy floating
    if isinstance(obj, np.floating):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if 'float' in type_name:
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val

    # Numpy array
    if isinstance(obj, np.ndarray):
        return [_sanitize_value(v) for v in obj.tolist()]

    # Numpy complex/void
    if isinstance(obj, (np.complexfloating, np.void)):
        return str(obj)

    # Dict - recursively sanitize
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            key = str(k) if not isinstance(k, str) else k
            result[key] = _sanitize_value(v)
        return result

    # List/tuple - recursively sanitize
    if isinstance(obj, (list, tuple)):
        return [_sanitize_value(v) for v in obj]

    # Python bool
    if isinstance(obj, bool):
        return bool(obj)

    # Python float
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    # Python int
    if isinstance(obj, int):
        return int(obj)

    # Bytes
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')

    # String - return as is
    if isinstance(obj, str):
        return obj

    # Fallback - convert to string
    return str(obj)


# ── Utilidades internas ──────────────────────────────────────────

def _ensure_dir(path: str = RESULTS_DIR) -> Path:
    """Crea directorio de resultados si no existe."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sanitize_filename(name: str) -> str:
    """Limpia nombre para uso como nombre de archivo."""
    for c in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\n', '\r']:
        name = name.replace(c, '_')
    if '.' in name:
        name = name.rsplit('.', 1)[0]
    return name[:180]


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _timestamp_filename() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _fmt_val(v):
    """Formatea valor para CSV."""
    if v is None:
        return ""
    try:
        if hasattr(v, 'item'):
            v = v.item()
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return ""
        return v
    except:
        return str(v)


def _coherente_str(val):
    """Convierte coherente a string para CSV."""
    if val is None:
        return "UNKNOWN"
    return "TRUE" if val else "FALSE"


# ── Función principal ────────────────────────────────────────────

def save_result(
    result: dict,
    label: str = "unknown",
    results_dir: str = RESULTS_DIR,
    include_watermark: bool = True,
) -> dict:
    """
    Guarda resultado de análisis con etiquetado automático.

    Genera:
        {label}.csv           — métricas principales
        {label}_{timestamp}_detalle.json — resultado completo
        history.jsonl         — índice append
        index.csv             — tabla resumen
    """
    _ensure_dir(results_dir)
    ts = _timestamp()
    ts_file = _timestamp_filename()
    safe_label = _sanitize_filename(label)

    saved = {}

    # Extraer datos - usar get seguro
    r3 = result.get("R3") or {}
    metrics = result.get("metrics") or {}

    # Obtener valores con safe access
    coherente_val = r3.get("coherent") if r3 else None
    if coherente_val is not None:
        coherente_val = bool(coherente_val)

    # ── 1. CSV ─────────────────────────────────────────────────
    csv_path = Path(results_dir) / f"{safe_label}.csv"
    file_exists = csv_path.exists()

    row = {
        "timestamp": ts,
        "archivo_origen": label,
        "n_puntos": _fmt_val(len(result.get("x_normalized", []))),
        "tau": _fmt_val(result.get("tau")),
        "tau_inicial": _fmt_val(result.get("tau_initial")),
        "epsilon": _fmt_val(result.get("epsilon")),
        "regimen": _fmt_val(r3.get("regime")),
        "regimen_desc": _fmt_val(r3.get("regime_desc")),
        "R3_score": _fmt_val(r3.get("R3_score")),
        "coherente": _coherente_str(coherente_val),
        "delta": _fmt_val(r3.get("delta")),
        "n_validas": _fmt_val(r3.get("n_valid")),
        "lambda_lyapunov": _fmt_val(metrics.get("lambda")),
        "D2_corr_dim": _fmt_val(metrics.get("D2")),
        "LZ_complejidad": _fmt_val(metrics.get("LZ")),
        "TE_transferencia": _fmt_val(metrics.get("TE")),
        "SampEn": _fmt_val(metrics.get("SampEn")),
        "autor": AUTHOR if include_watermark else "",
    }

    with open(csv_path, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    saved["csv_principal"] = str(csv_path)

    # ── 2. JSON detalle (sanitizar todo ANTES de json.dump) ─────
    detail_raw = {
        "timestamp": ts,
        "archivo_origen": label,
        "autor": AUTHOR if include_watermark else "",
        "pipeline": {
            "tau": result.get("tau"),
            "tau_initial": result.get("tau_initial"),
            "epsilon": result.get("epsilon"),
        },
        "regimen": {
            "tipo": r3.get("regime"),
            "descripcion": r3.get("regime_desc"),
            "delta": r3.get("delta"),
        },
        "r3": {
            "score": r3.get("R3_score"),
            "coherente": coherente_val,
            "n_validas": r3.get("n_valid"),
        },
        "metrics": metrics,
        "gradients": r3.get("gradients", {}),
        "stability_map": r3.get("stability_map", {}),
    }

    # Sanitizar TODO recursivamente
    detail = _sanitize_value(detail_raw)

    detail_path = Path(results_dir) / f"{safe_label}_{ts_file}_detalle.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)

    saved["json_detalle"] = str(detail_path)

    # ── 3. History JSONL ───────────────────────────────────────
    history_raw = {
        "timestamp": ts,
        "archivo_origen": label,
        "regimen": r3.get("regime"),
        "R3_score": r3.get("R3_score"),
        "coherente": coherente_val,
        "autor": AUTHOR if include_watermark else "",
    }

    history_entry = _sanitize_value(history_raw)

    history_path = Path(results_dir) / HISTORY_JSONL
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(history_entry, ensure_ascii=False) + "\n")

    saved["history_jsonl"] = str(history_path)

    # ── 4. Index CSV ───────────────────────────────────────────
    _update_index(results_dir)

    return {
        "timestamp": ts,
        "archivo_origen": label,
        "autor": AUTHOR if include_watermark else "",
        "saved_files": saved,
        "regimen": r3.get("regime"),
        "R3_score": r3.get("R3_score"),
        "coherente": coherente_val,
    }


def _update_index(results_dir: str = RESULTS_DIR) -> None:
    index_path = Path(results_dir) / INDEX_CSV
    history_path = Path(results_dir) / HISTORY_JSONL
    if not history_path.exists():
        return

    entries = []
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    fieldnames = ["timestamp", "archivo_origen", "regimen", "R3_score", "coherente", "autor"]
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            row = {k: str(entry.get(k, "")) for k in fieldnames}
            writer.writerow(row)


# ── Funciones de consulta ───────────────────────────────────────

def get_history(results_dir: str = RESULTS_DIR, limit: Optional[int] = None) -> List[dict]:
    history_path = Path(results_dir) / HISTORY_JSONL
    if not history_path.exists():
        return []

    entries = []
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    return entries[-limit:] if limit else entries


def load_results_csv(label: str, results_dir: str = RESULTS_DIR) -> Optional[Path]:
    safe_label = _sanitize_filename(label)
    csv_path = Path(results_dir) / f"{safe_label}.csv"
    return csv_path if csv_path.exists() else None


def get_summary_stats(results_dir: str = RESULTS_DIR) -> dict:
    history = get_history(results_dir)
    if not history:
        return {
            "total_analisis": 0,
            "regimens": {},
            "coherente_count": 0,
            "coherente_pct": 0.0,
            "R3_mean": None,
            "R3_min": None,
            "R3_max": None,
            "archivos": [],
        }

    regimens = {}
    r3_scores = []
    coherent_count = 0

    for entry in history:
        r = entry.get("regimen", "unknown")
        regimens[r] = regimens.get(r, 0) + 1
        val = entry.get("R3_score")
        if val is not None:
            r3_scores.append(val)
        if entry.get("coherente"):
            coherent_count += 1

    return {
        "total_analisis": len(history),
        "regimens": regimens,
        "coherente_count": coherent_count,
        "coherente_pct": coherent_count / len(history) * 100 if history else 0.0,
        "R3_mean": sum(r3_scores) / len(r3_scores) if r3_scores else None,
        "R3_min": min(r3_scores) if r3_scores else None,
        "R3_max": max(r3_scores) if r3_scores else None,
        "archivos": [e.get("archivo_origen", "unknown") for e in history[-20:]],
    }


if __name__ == "__main__":
    print(f"result_saver.py — {AUTHOR}")
    print(f"Directorio: {RESULTS_DIR}")