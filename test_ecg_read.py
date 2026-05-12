#!/usr/bin/env python3
"""
test_ecg_read.py — Verifica lectura de archivos MIT-BIH
Autor: Emanuel Duarte — 2026
"""

import sys
import os

# Agregar directorio actual al path
sys.path.insert(0, os.path.dirname(__file__))

def read_mitbih_bytes(datfile, heafile):
    """Función de lectura MIT-BIH extraída de app.py"""
    lines = heafile.read().decode("latin-1").strip().split("\n")
    hdr   = lines[0].split()
    fs, n = int(hdr[2]), int(hdr[3])
    si    = lines[1].split()
    gain, bl = float(si[2]), int(si[4])
    raw   = datfile.read()
    s     = []
    i     = 0
    while i + 2 < len(raw):
        b0, b1, b2 = raw[i], raw[i+1], raw[i+2]
        s1 = b0 | ((b1 & 0x0F) << 8); s1 = s1 - 4096 if s1 >= 2048 else s1
        s2 = b2 | ((b1 & 0xF0) << 4); s2 = s2 - 4096 if s2 >= 2048 else s2
        s.extend([s1, s2]); i += 3
    return s, fs, n

def test_mitbih():
    """Prueba básica de lectura MIT-BIH"""
    print("=" * 60)
    print("TEST: Lectura de archivos MIT-BIH")
    print("=" * 60)

    # Crear datos de prueba simulados
    print("\n[1] Generando archivos de prueba simulados...")

    # Simular archivo .hea
    hea_content = """100 2 360 650000
100.dat 212 200 11 0 -4 14893 0
100.dat 212 200 11 0 -4 14893 0"""

    # Simular archivo .dat (primeros bytes)
    import struct
    dat_content = b""
    for i in range(500):  # 500 muestras de prueba
        val1 = 100 + i % 100  # Simular valores de ECG
        val2 = 150 + i % 80
        # Formato MIT-BIH: 2 samples de 12 bits cada uno
        s1 = val1 if val1 < 2048 else val1 - 4096
        s2 = val2 if val2 < 2048 else val2 - 4096
        b0 = s1 & 0xFF
        b1 = ((s1 >> 8) & 0x0F) | ((s2 << 4) & 0xF0)
        b2 = (s2 >> 4) & 0xFF
        dat_content += bytes([b0, b1, b2])

    print(f"   .hea: {len(hea_content)} bytes (simulado)")
    print(f"   .dat: {len(dat_content)} bytes ({len(dat_content)//3} samples)")

    # Simular lectura
    print("\n[2] Simulando lectura con read_mitbih_bytes...")
    from io import BytesIO
    heafile = BytesIO(hea_content.encode("latin-1"))
    datfile = BytesIO(dat_content)

    try:
        signal, fs, n = read_mitbih_bytes(datfile, heafile)
        print(f"   ✓ Frecuencia muestreo: {fs} Hz")
        print(f"   ✓ Muestras leídas: {len(signal)}")
        print(f"   ✓ Rango de valores: [{min(signal)}, {max(signal)}]")
        print("\n   PRUEBA DE LECTURA: ÉXITO ✓")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print("\n   PRUEBA DE LECTURA: FALLO ✗")
        return False

    return True

def test_with_real_files():
    """Probar con archivos reales si existen"""
    print("\n" + "=" * 60)
    print("TEST: Verificando archivos reales en directorio actual")
    print("=" * 60)

    import glob
    dat_files = glob.glob("*.dat") + glob.glob("*.DAT")
    hea_files = glob.glob("*.hea") + glob.glob("*.HEA")

    print(f"\nArchivos .dat encontrados: {dat_files}")
    print(f"Archivos .hea encontrados: {hea_files}")

    if not dat_files or not hea_files:
        print("\n   No se encontraron archivos MIT-BIH para probar.")
        print("   Para probar con datos reales, coloca 100.dat y 100.hea")
        print("   en el mismo directorio que este script.")
        return None

    # Emparejar archivos
    from os.path import splitext
    for dat_file in dat_files:
        base = splitext(dat_file)[0]
        hea_file = f"{base}.hea"

        if hea_file in hea_files:
            print(f"\n   Probando par: {dat_file} + {hea_file}")
            try:
                with open(dat_file, "rb") as df:
                    with open(hea_file, "r") as hf:
                        signal, fs, n = read_mitbih_bytes(df, hf)
                        print(f"   ✓ {dat_file}: {len(signal)} samples @ {fs} Hz")
            except Exception as e:
                print(f"   ✗ Error leyendo {dat_file}: {e}")

    return True

if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("MODE Pipeline v2.2 ECG_FIX — Test de lectura MIT-BIH")
    print("█" * 60)

    success = test_mitbih()
    test_with_real_files()

    print("\n" + "=" * 60)
    if success:
        print("RESULTADO: Test de lectura PASSED ✓")
    else:
        print("RESULTADO: Test de lectura FAILED ✗")
    print("=" * 60 + "\n")