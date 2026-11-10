# Código y datos mínimos del artículo ADT

Repositorio público del estudio del modelo
amiloidogénico v0.3. Conserva las implementaciones que produjeron el diseño
final P14, las intervenciones pareadas JT1, el sustituto E1, las figuras G8 y el
análisis estadístico SA1. Los 1024 sujetos son puntos sintéticos de diseño; no
representan pacientes ni una población humana calibrada.

## Qué contiene

- `src/adt/`: cierre de dependencias del código original de esas cinco tareas.
  Los 27 módulos se copiaron sin modificar sus ecuaciones, estimandos, semillas,
  criterios de aceptación ni cálculos. Algunos módulos de etapas anteriores
  permanecen porque P14/JT1 reutiliza sus funciones.
- `scripts/reproduce.py`: entrada única para ejecutar o recolectar P14/JT1 y
  preparar, seleccionar o evaluar E1. Sustituye cinco lanzadores de pocas líneas;
  llama a las mismas funciones `main` originales.
- `scripts/render_stage15_figures.py` y `scripts/analyze_stage16_results.py`:
  entradas originales G8 y SA1. Se conservan porque esos programas archivan una
  copia de su propia entrada.
- `config/`: nueve configuraciones científicas aprobadas.
- `results/`: 18 archivos de entrada congelados por SHA-256, suficientes para
  volver a generar G8 y SA1. No se incluyen trayectorias NPZ, lotes P14/JT1,
  modelos E1 ni las entradas históricas de etapas previas.
- `reference/sa1/`: cuatro tablas seleccionadas y el resumen del análisis SA1
  confirmado por ejecución humana, para comparar cifras.
- `figures/`: las seis figuras G8 revisadas, cada una en SVG, PDF y PNG.
- `SOURCE_SHA256SUMS.txt`: hashes de los módulos, YAML, entradas G8/SA1 y
  tablas y figuras de referencia copiados del proyecto principal.

La selección detallada y sus límites están en [SELECCION.md](SELECCION.md).

## Instalación y comprobación manual en WSL

Los comandos se ejecutan **por la persona usuaria**; Codex no ha ejecutado
Python, pruebas, simulaciones, entrenamiento ni el análisis en esta carpeta.

```bash
git clone https://github.com/joandogo/adt-amyloid-pathway-digital-twin.git
cd adt-amyloid-pathway-digital-twin
sha256sum -c SOURCE_SHA256SUMS.txt
python -m pip install -r requirements.txt
PYTHONPATH=src python -m compileall -q src/adt scripts
PYTHONPATH=src python -u scripts/render_stage15_figures.py --check-only
PYTHONPATH=src python -u scripts/analyze_stage16_results.py --check-only
```

Los dos `--check-only` deben aceptar todas las fuentes congeladas por SHA-256.
G8 debe anunciar seis figuras y nueve tablas. SA1 debe anunciar 1024 sujetos,
229376 contrastes, 43008 predicciones y 5000 réplicas estratificadas. Estos
comandos no generan resultados.

### Verificación manual comunicada

La persona usuaria ejecutó desde `/mnt/e/ADT/repositorio_articulo` los comandos
de comprobación e instalación del bloque anterior, sin clonar de nuevo, y
compartió la salida: los 79 archivos de
`SOURCE_SHA256SUMS.txt` dieron `OK`; la instalación editable terminó
correctamente; `compileall -q` terminó sin errores; y los preflights G8 y SA1
comunicaron los conteos previstos. Es una comprobación de integridad,
instalación y fuentes fijadas. No se ejecutaron en esta comprobación el
renderizado completo, el análisis estadístico ni simulaciones nuevas.

Si los preflights pasan, la persona usuaria puede generar nuevas copias de G8
y SA1:

```bash
PYTHONPATH=src python -u scripts/render_stage15_figures.py --progress-interval 5
PYTHONPATH=src python -u scripts/analyze_stage16_results.py --progress-interval 5
```

La ejecución debe crear seis figuras en SVG/PDF/PNG y nueve tablas G8, además
de quince tablas SA1. Comparar las cifras de respuesta primaria, compromiso
C99, heterogeneidad y bootstrap con `reference/sa1/`. El hash del resumen nuevo
puede diferir: el snapshot contiene un subconjunto distinto de archivos fuente
y registra el nuevo entorno. Una coincidencia de procedencia y de cifras debe
ser revisada por la persona usuaria antes de considerar verificada la
regeneración.

## Reproducción completa de las simulaciones

El código y las configuraciones P14/JT1/E1 están aquí por transparencia
metodológica. Sus ejecuciones completas requieren los archivos históricos que
las configuraciones fijan por ruta y SHA-256: el piloto C1, Sobol C, la colección
P14, la colección P12, los NPZ de estados finales, los lotes JT1 y el conjunto
de datos E1. Esos archivos no forman parte de este paquete reducido. No ejecutar
estas etapas con fuentes sustituidas o hashes modificados para forzar una corrida.

Con los archivos originales restaurados en sus rutas relativas, el punto de
entrada único conserva las opciones originales:

```bash
PYTHONPATH=src python -u scripts/reproduce.py p14-run --batch 1 --check-only
PYTHONPATH=src python -u scripts/reproduce.py p14-collect --from-root --check-only
PYTHONPATH=src python -u scripts/reproduce.py jt1-run --batch 1 --check-only
PYTHONPATH=src python -u scripts/reproduce.py jt1-collect --from-root --check-only
PYTHONPATH=src python -u scripts/reproduce.py e1 prepare --check-only
```

La secuencia de ejecución científica completa se debe seguir en la documentación
original de cada etapa; estos cinco ejemplos son solo preflights y dependen de
sus archivos originales.

## Dependencias y disponibilidad

Python >=3.11. `pyproject.toml` declara NumPy, SciPy, PyYAML, scikit-learn y el
extra de Matplotlib para G8; `requirements.txt` es la entrada de instalación.
Los archivos CSV incluidos son productos de un diseño sintético. Para un depósito
definitivo falta decidir la ubicación persistente de los archivos grandes
omitidos y obtener revisión humana de código, datos, figuras y declaraciones.

## Licencias

El código, las configuraciones y la documentación están bajo [MIT](LICENSE).
Los CSV sintéticos de `results/` y `reference/` están bajo
[CC0 1.0](LICENSE-DATA.md). Las figuras de `figures/` están bajo
[CC BY 4.0](LICENSE-FIGURES.md). `.gitattributes` conserva los bytes originales
de los archivos fijados por SHA-256 al clonar en distintos sistemas.

---

# ADT article code and data — English version

Public repository for the amyloidogenic pathway model v0.3 study. It preserves
the implementations used for the final P14 design, paired JT1 interventions,
the E1 surrogate, G8 figures, and SA1 statistical analysis. The 1,024 subjects
are synthetic design points; they are neither patients nor a calibrated human
population.

## Contents

- `src/adt/`: the dependency closure of the original code for those five
  workflows. The 27 modules were copied without changing equations, estimands,
  seeds, acceptance criteria, or calculations. Some earlier-stage modules remain
  because P14/JT1 reuses their functions.
- `scripts/reproduce.py`: one entry point to run or collect P14/JT1 and to
  prepare, select, or evaluate E1. It replaces five short launchers and calls
  the same original `main` functions.
- `scripts/render_stage15_figures.py` and `scripts/analyze_stage16_results.py`:
  the original G8 and SA1 entry points. They remain separate because those
  programs archive a copy of their own entry-point file.
- `config/`: nine approved scientific configurations.
- `results/`: 18 SHA-256-pinned input files sufficient to regenerate G8 and
  SA1. The package excludes NPZ trajectories, P14/JT1 batches, E1 models, and
  historical inputs from earlier stages.
- `reference/sa1/`: four selected tables and the summary of the SA1 analysis
  confirmed by human execution, for comparing results.
- `figures/`: the six reviewed G8 figures, each in SVG, PDF, and PNG.
- `SOURCE_SHA256SUMS.txt`: hashes of the modules, YAML files, G8/SA1 inputs,
  reference tables, and figures copied from the main project.

[SELECCION.md](SELECCION.md) documents the selection and its limits in Spanish.

## Manual installation and preflight in WSL

The human user runs these commands. Codex has not run Python, tests,
simulations, training, or the statistical analysis in this repository.

```bash
git clone https://github.com/joandogo/adt-amyloid-pathway-digital-twin.git
cd adt-amyloid-pathway-digital-twin
sha256sum -c SOURCE_SHA256SUMS.txt
python -m pip install -r requirements.txt
PYTHONPATH=src python -m compileall -q src/adt scripts
PYTHONPATH=src python -u scripts/render_stage15_figures.py --check-only
PYTHONPATH=src python -u scripts/analyze_stage16_results.py --check-only
```

Both `--check-only` commands should accept all SHA-256-pinned sources. G8
should report six figures and nine tables. SA1 should report 1,024 subjects,
229,376 contrasts, 43,008 predictions, and 5,000 stratified replicates. These
commands do not generate results.

### Human-reported verification

The user ran the verification and installation commands above from
`/mnt/e/ADT/repositorio_articulo`, without cloning again, and shared the output:
all 79 entries in `SOURCE_SHA256SUMS.txt` returned `OK`; the editable
installation completed; `compileall -q` completed without errors; and the G8
and SA1 preflights reported the expected counts. This verifies file integrity,
installation, and pinned inputs in that local checkout. It does not establish
that a fresh clone or a full G8/SA1 regeneration has been run.

After the preflights pass, the human user may regenerate G8 and SA1:

```bash
PYTHONPATH=src python -u scripts/render_stage15_figures.py --progress-interval 5
PYTHONPATH=src python -u scripts/analyze_stage16_results.py --progress-interval 5
```

The runs should create six figures in SVG/PDF/PNG and nine G8 tables, plus
fifteen SA1 tables. Compare the primary-response, C99 trade-off,
heterogeneity, and bootstrap values with `reference/sa1/`. A new summary hash
may differ because the source snapshot contains a different subset of files
and records the new environment. The human user should review provenance and
numerical agreement before treating a regeneration as verified.

## Full simulation reproduction

The P14/JT1/E1 code and configurations are included for methodological
transparency. Full execution also requires the historical files pinned by
path and SHA-256 in those configurations: the C1 pilot, Sobol C, the P14 and
P12 collections, NPZ final states, JT1 batches, and the E1 dataset. Those files
are outside this reduced package. Do not replace inputs or change hashes to
force a run.

With the original files restored at their relative paths, the single entry
point retains the original options:

```bash
PYTHONPATH=src python -u scripts/reproduce.py p14-run --batch 1 --check-only
PYTHONPATH=src python -u scripts/reproduce.py p14-collect --from-root --check-only
PYTHONPATH=src python -u scripts/reproduce.py jt1-run --batch 1 --check-only
PYTHONPATH=src python -u scripts/reproduce.py jt1-collect --from-root --check-only
PYTHONPATH=src python -u scripts/reproduce.py e1 prepare --check-only
```

Follow the original documentation for the complete scientific execution
sequence. These five commands are preflight examples and depend on the
original archived inputs.

## Dependencies and availability

Python >=3.11. `pyproject.toml` declares NumPy, SciPy, PyYAML, scikit-learn,
and the Matplotlib extra for G8; `requirements.txt` is the installation entry
point. The included CSV files are products of a synthetic design. A definitive
archive still needs a persistent location for the omitted large files and
human review of code, data, figures, and statements.

## Licenses

Code, configurations, and documentation are under [MIT](LICENSE). Synthetic
CSV files in `results/` and `reference/` are under [CC0 1.0](LICENSE-DATA.md).
Figures in `figures/` are under [CC BY 4.0](LICENSE-FIGURES.md).
`.gitattributes` preserves the original bytes of SHA-256-pinned files across
platforms.
