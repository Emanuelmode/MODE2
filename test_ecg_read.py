#!/usr/bin/env python3
"""
test_ecg_read_corregido.py — Verifica lectura de archivos MIT-BIH con canales separados
Autor: Emanuel Duarte — 2026
"""

from io import BytesIO
import glob
import os

import numpy as np


def _parse_mitbih_header(heafile):
    lines = heafile.read().decode("latin-1").strip().splitlines()
    if not lines:
        raise ValueError("Header .hea vacío")

    hdr = lines[0].split()
    if len(hdr) < 4:
        raise ValueError("Header MIT-BIH inválido")

    n_sig = int(hdr[1])
    fs = int(float(hdr[2]))
    n_samples = int(hdr[3])

    signal_info = []
    for idx in range(n_sig):
        parts = lines[idx + 1].split()
        fmt = parts[1]
        baseline = int(parts[4])
        lead_name = " ".join(parts[8:]) if len(parts) >= 9 else f"ch{idx+1}"
        signal_info.append({"fmt": fmt, "baseline": baseline, "lead_name": lead_name})

    return {"n_sig": n_sig, "fs": fs, "n_samples": n_samples, "signals": signal_info}


def _decode_mitbih_format_212(raw: bytes, n_samples: int):
    ch1, ch2 = [], []
    i = 0
    while i + 2 < len(raw) and len(ch1) < n_samples:
        b0, b1, b2 = raw[i], raw[i + 1], raw[i + 2]
        s1 = b0 | ((b1 & 0x0F) << 8)
        s2 = b2 | ((b1 & 0xF0) << 4)
        if s1 >= 2048:
            s1 -= 4096
        if s2 >= 2048:
            s2 -= 4096
        ch1.append(s1)
        ch2.append(s2)
        i += 3
    return np.asarray(ch1[:n_samples], dtype=float), np.asarray(ch2[:n_samples], dtype=float)


def read_mitbih_bytes(datfile, heafile):
    meta = _parse_mitbih_header(heafile)
    raw = datfile.read()
    if meta["n_sig"] != 2:
        raise ValueError(f"Solo se soportan 2 canales por ahora (n_sig={meta['n_sig']})")
    if not all(sig["fmt"].startswith("212") for sig in meta["signals"][:2]):
        raise ValueError("Solo se soporta formato 212 por ahora")

    sig1, sig2 = _decode_mitbih_format_212(raw, meta["n_samples"])
    sig1 = sig1 - meta["signals"][0]["baseline"]
    sig2 = sig2 - meta["signals"][1]["baseline"]
    return {
        "signals": {"ch1": sig1, "ch2": sig2},
        "lead_names": {
            "ch1": meta["signals"][0]["lead_name"],
            "ch2": meta["signals"][1]["lead_name"],
        },
        "fs": meta["fs"],
        "n_samples": meta["n_samples"],
    }


def _pack_212_pair(s1: int, s2: int) -> bytes:
    s1u = s1 & 0x0FFF
    s2u = s2 & 0x0FFF
    b0 = s1u & 0xFF
    b1 = ((s1u >> 8) & 0x0F) | ((s2u << 4) & 0xF0)
    b2 = (s2u >> 4) & 0xFF
    return bytes([b0, b1, b2])


def test_simulated_channels():
    print("=" * 60)
    print("TEST: Lectura MIT-BIH simulada con canales separados")
    print("=" * 60)

    hea_content = """100 2 360 10
100.dat 212 200 11 0 0 0 0 MLII
100.dat 212 200 11 0 0 0 0 V5"""

    ch1_expected = [100 + i for i in range(10)]
    ch2_expected = [200 + 2 * i for i in range(10)]

    dat_content = b"".join(_pack_212_pair(a, b) for a, b in zip(ch1_expected, ch2_expected))
    record = read_mitbih_bytes(BytesIO(dat_content), BytesIO(hea_content.encode("latin-1")))

    ch1 = record["signals"]["ch1"].astype(int).tolist()
    ch2 = record["signals"]["ch2"].astype(int).tolist()

    ok = (ch1 == ch1_expected) and (ch2 == ch2_expected)
    print(f"Canal 1 esperado: {ch1_expected}")
    print(f"Canal 1 leído   : {ch1}")
    print(f"Canal 2 esperado: {ch2_expected}")
    print(f"Canal 2 leído   : {ch2}")
    print(f"Resultado: {'ÉXITO ✓' if ok else 'FALLO ✗'}")
    return ok


def test_with_real_files():
    print("\n" + "=" * 60)
    print("TEST: Verificando archivos reales en directorio actual")
    print("=" * 60)

    dat_files = glob.glob("*.dat") + glob.glob("*.DAT")
    hea_files = glob.glob("*.hea") + glob.glob("*.HEA")

    if not dat_files or not hea_files:
        print("No se encontraron archivos .dat/.hea para probar.")
        return None

    for dat_file in dat_files:
        base = os.path.splitext(dat_file)[0]
        hea_file = f"{base}.hea"
        if hea_file in hea_files:
            print(f"\nProbando par: {dat_file} + {hea_file}")
            with open(dat_file, "rb") as df, open(hea_file, "rb") as hf:
                record = read_mitbih_bytes(df, hf)
            print(
                f"✓ {record['n_samples']} muestras por canal @ {record['fs']} Hz | "
                f"Leads: {record['lead_names']['ch1']} / {record['lead_names']['ch2']}"
            )
            print(
                f"  ch1 rango: [{record['signals']['ch1'].min():.0f}, {record['signals']['ch1'].max():.0f}] | "
                f"ch2 rango: [{record['signals']['ch2'].min():.0f}, {record['signals']['ch2'].max():.0f}]"
            )

    return True


if __name__ == "__main__":
    ok = test_simulated_channels()
    test_with_real_files()
    print("\n" + "=" * 60)
    print(f"RESULTADO FINAL: {'PASSED ✓' if ok else 'FAILED ✗'}")
    print("=" * 60)
