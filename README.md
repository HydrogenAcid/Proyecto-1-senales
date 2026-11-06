# Proyecto 1 — Relaciones de escala en series temporales autoafines

Estudio numérico de las relaciones entre el exponente de Hurst, DFA, espectros, wavelets y dimensión de gráfica en ruido gaussiano fraccionario (fGn), movimiento browniano fraccionario (fBm) y señales de contraste. El [artículo](article/main.pdf) presenta la fundamentación y los resultados; el [notebook](notebooks/proyecto_1.ipynb) contiene los experimentos ejecutados.

## Contenido

| Ruta | Contenido |
| --- | --- |
| `analysis.py` | Generadores, estimadores y análisis estadístico reutilizables. |
| `make_notebook.py` | Código fuente que construye el notebook. |
| `notebooks/proyecto_1.ipynb` | Experimentos, tablas y gráficas con sus salidas guardadas. |
| `results/` | Resultados CSV, incluidos los valores por realización y las discrepancias de la Figura 7. |
| `article/main.tex` y `article/references.bib` | Fuente LaTeX y bibliografía del artículo. |
| `article/main.pdf` y `article/figures/` | Artículo compilado y figuras reproducibles. |

En este repositorio se excluyen `tests/` y `docs/`. También se omiten cachés y archivos auxiliares de compilación; los resultados CSV, el notebook ejecutado, las figuras y el PDF final sí se incluyen.

## Reproducir los cálculos

Desde la raíz del proyecto, crea un entorno e instala las dependencias:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Abre `notebooks/proyecto_1.ipynb`, selecciona ese entorno como kernel y ejecuta todas las celdas. Las semillas están fijadas. La malla de referencia comprende 2 generadores × 9 valores de H × 4 longitudes × 200 realizaciones (14 400 registros). La malla ampliada de contrastes contiene otras 28 800 realizaciones; el corte lorentziano tiene además un experimento separado. La ejecución completa puede tardar varios minutos.

Si modificas `make_notebook.py`, reconstruye primero el notebook con `python make_notebook.py` desde esta carpeta. Ese comando borra las salidas guardadas: ejecuta después todas las celdas para restaurarlas y regenerar los CSV y las figuras.

`results/all_relations.csv` contiene 576 pruebas de residuos pareados para ocho identidades. Los intervalos percentiles del 95 % usan 5 000 remuestreos y no están ajustados; los valores p de las pruebas t se corrigen con Bonferroni sobre las 576 pruebas. `results/residuals.csv` guarda las 216 filas de las tres relaciones principales. `results/h_equivalent_discrepancies.csv` guarda las diez diferencias entre pares de estimadores, en unidades equivalentes de Hurst, para los nueve valores de H. El notebook conserva las diez dispersiones entre pares en `article/figures/all_pairs.pdf`; el artículo utiliza su resumen como mapa de calor en la Figura 7.

## Compilar el artículo

Se requiere una distribución de LaTeX con `latexmk` y BibTeX. Desde la raíz del proyecto:

```bash
cd article
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

El PDF actualizado queda en `article/main.pdf`. La configuración local de VS Code también coloca allí la salida de LaTeX Workshop.

## Fuentes del enunciado

El enunciado `Tarea Proyecto 1.pdf` y tres PDF bibliográficos se encuentran fuera de esta carpeta, en el directorio padre y en su subcarpeta `fuentes/`; por eso **no forman parte de este repositorio**. El notebook indica esas rutas relativas y `article/references.bib` documenta la bibliografía. Los archivos CSV y el código incluidos permiten ejecutar los cálculos sin esos PDF; para abrir los enlaces documentales del notebook, coloca las fuentes en esas rutas relativas.
