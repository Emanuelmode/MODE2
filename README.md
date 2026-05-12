# MODE Attractor Pipeline — ECG Fix Edition v2.2

## Archivos incluidos
- `app.py` - Aplicación Streamlit principal
- `pipeline.py` - Motor de análisis de atractores
- `result_saver.py` - Módulo de persistencia de resultados
- `requirements.txt` - Dependencias de Python

## Instalación
```bash
pip install -r requirements.txt
```

## Ejecución
```bash
streamlit run app.py
```

## Cambios en v2.2 ECG_FIX
- Corrección del problema de carga de archivos ECG MIT-BIH
- Uso de `session_state` para persistir datos entre re-renders de Streamlit
- Validación mejorada de índices y mensajes de estado
- Soporte para extensiones `.dat`, `.bin`, `.hea`, `.txt`
- Sliders ajustados dinámicamente según la duración real del archivo

## Formato de datos ECG
La aplicación espera archivos en formato MIT-BIH:
- `100.dat` - Archivo binario con los datos de la señal
- `100.hea` - Archivo de header con metadatos (frecuencia de muestreo, número de muestras, etc.)

Ambos archivos deben corresponder al mismo registro y estar en el mismo directorio.

## Autor
Emanuel Duarte — Pergamino, Argentina — 2026