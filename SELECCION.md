# Selección de código para el artículo

## Criterio

Se eligieron las rutas que producen los resultados centrales del manuscrito
actual: diseño conjunto P14, comparación pareada JT1, sustituto E1, figuras G8
y estadística SA1. Se cerraron sus importaciones relativas de `adt` y se añadió
el módulo de figuras que G8 importa dentro de la función de renderizado. El
resultado son 27 módulos originales, tres entradas de usuario y nueve YAML.
La unión de cinco entradas P14/JT1/E1 en `scripts/reproduce.py` solo delega en
los `main(argv)` originales; no cambia cálculos ni datos.

| Resultado | Entrada | Implementación principal |
| --- | --- | --- |
| Cohorte sintética P14 | `reproduce.py p14-run/p14-collect` | `final_cohort_cli.py`, `final_cohort_collection_cli.py`, núcleo ODE |
| Intervenciones JT1 | `reproduce.py jt1-run/jt1-collect` | `joint_interventions_cli.py`, `joint_interventions_collection_cli.py` |
| Sustituto E1 | `reproduce.py e1` | `therapeutic_surrogate.py`, `therapeutic_surrogate_cli.py` |
| Figuras y tablas G8 | `render_stage15_figures.py` | `stage15_publication.py`, `stage15_publication_cli.py` |
| Estadística SA1 | `analyze_stage16_results.py` | `stage16_statistics.py`, `stage16_statistics_cli.py` |

Se excluyeron los ejecutores de planificación o verificación de fases anteriores,
Morris/Sobol exploratorios, borradores del artículo, generador de manuscrito,
auditor de texto, notebooks y pruebas que no forman parte de la cadena central
P14→JT1→E1→G8→SA1 del manuscrito actual. `sweep_cli.py`, `sweep_plan.py` y
`paired_interventions_cli.py` sí permanecen como bibliotecas importadas por el
código P14/JT1; su presencia no añade los ejecutores de esas fases.

## Datos incluidos y límite de reproducción

Se copiaron sin modificación los 17 archivos fijados por G8 más el resumen G8
que SA1 exige. Sus rutas y hashes esperados están en
`config/stage15_publication.yaml` y `config/stage16_statistical_analysis.yaml`.
También se incluyen las seis figuras G8 revisadas en SVG, PDF y PNG.
Es suficiente para los preflights y la regeneración de G8/SA1. No es suficiente
para repetir desde cero las integraciones, la selección E1 ni la apertura de
prueba E1. El paquete no afirma equivalencia numérica verificada: Codex realizó
solo inspección y copiado estático.

## Decisión de no reescribir el núcleo

Una reescritura de las ecuaciones, el integrador, la selección por semilla, los
estimandos o el bootstrap puede cambiar resultados aunque reduzca líneas.
Por ello se quitaron lanzadores redundantes y se conservaron los módulos
científicos exactos. Cualquier consolidación adicional necesita comparación
humana de salidas, hashes de entradas y revisión de diferencias numéricas.
