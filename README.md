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
ser revisada por la persona usuaria antes de publicar el repositorio.

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
