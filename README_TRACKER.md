Tracker (ByteTrack-style) — configuración y ejemplos
===============================================

Este archivo documenta los parámetros del tracker incluido en el proyecto y ejemplos de uso.

Parámetros del tracker
----------------------

- `--tracker`: habilita el tracker. Valores: `none` (sin tracker) o `iou` (tracker integrado). Por defecto `iou`.
- `--tracker-high-conf`: umbral alto de confianza para crear nuevas pistas (default: `0.6`). Las detecciones con score >= este valor se consideran "alta confianza" y se usan para crear nuevas pistas.
- `--tracker-low-conf`: umbral bajo de confianza para matching secundario (default: `0.1`). Las detecciones con score entre `low_conf` y `high_conf` se intentan emparejar en una segunda pasada para recuperar pistas con detecciones débiles.

Cómo funciona (resumen)
-----------------------

El tracker implementado en `src/tracker.py` usa una lógica inspirada en ByteTrack:

1. Predecir la posición de las pistas activas (predicción por velocidad simple).
2. Emparejar en dos fases:
   - Primero detectar y emparejar detecciones de alta confianza (>= `high_conf`).
   - Luego intentar emparejar detecciones de baja confianza (`low_conf` <= score < `high_conf`) con pistas no asignadas.
3. Crear nuevas pistas únicamente a partir de detecciones de alta confianza.

Esta estrategia reduce la creación de pistas duplicadas por ruidos o detecciones intermitentes.

Ejemplos de uso
----------------

Crear y guardar un video usando el tracker con ajustes por defecto:

```
python main.py --model "C:/ruta/a/yolov12l-face.pt" --source "C:/ruta/video.mp4" --tracker iou --save --out salida_detectada.mp4
```

Usar un umbral high más alto (menos tracks nuevos) y permitir matching secundario más agresivo:

```
python main.py --model model.pt --source 0 --tracker iou --tracker-high-conf 0.75 --tracker-low-conf 0.05
```

Notas de tuning
----------------

- Ajusta `--tracker-high-conf`/`--tracker-low-conf` según la precisión de tu detector. Detectores muy confiables permiten `high_conf` más alto; detectores ruidosos necesitan `low_conf` mayor para recuperar detecciones.
- Si tu escena tiene muchas oclusiones largas, incorporar embeddings de apariencia (DeepSORT/StrongSORT) o un filtro de estado (Kalman + Hungarian / SORT) mejorará la re-identificación.

Siguientes pasos posibles
------------------------

- Exponer más parámetros del tracker (p. ej. `max_lost`, `min_hits`) en CLI para facilitar tuning.
- Implementar Hungarian + Kalman (SORT/OC-SORT) para matching más óptimo.
- Integrar ReID embeddings (DeepSORT/StrongSORT) si hay oclusiones largas.

Si quieres, puedo generar ejemplos de configuración para tu video específico o integrar una de las alternativas mencionadas.
