# ML Estimation of Obesity

## Propósito
Este repositorio implementa una solución reproducible para **clasificar niveles de obesidad ordinal (7 clases)** usando datos de hábitos alimentarios y estilo de vida. Se enfoca en:

- **Entender el dataset** con EDA y controles de desbalance.
- **Preprocesar** con encoding, scaler y guardado de transformadores.
- **Entrenar y comparar** múltiples modelos con validación cruzada orientada a métricas ordinales.
- **Analizar reducción de dimensión** (PCA, t-SNE, UMAP) y documentar decisiones.

## Requisitos

1. Python 3.10+ y las dependencias descritas en `requirements.txt` (crea si aún no existe).
2. Ejecuta `pip install -r requirements.txt`.

## Estrategia de uso

1. **Análisis Exploratorio (notebook 01)**  
   - Abre `01_analisis_exploratorio.ipynb` y corre las celdas para revisar distribución de clases, IMC y visualizaciones.
2. **Preprocesamiento (notebook 02)**  
   - Ejecuta todo el notebook `02_preprocesamiento.ipynb`.  
   - Genera `data/processed/*.csv` y `models/preprocessing/*.pkl` que se usarán después.
3. **Entrenamiento y evaluación (notebook 03)**  
   - Corre `03_entrenamiento_modelos.ipynb` en su totalidad.  
   - Usa los CSV y transformadores guardados para entrenar los modelos, registrar métricas (MAE ordinal, accuracy, Spearman) y guardar figuras en `results/figuras/`.
4. **Reducción de dimensión (notebook 04)**  
   - Ejecuta `04_reduccion_dimension.ipynb` para probar PCA y UMAP.  
   - La sección final genera tabulado resumido en `results/tablas/reduction_comparison.csv`.

## Organización del repositorio

```
ml-estimation-of-obesity/
├── data/processed/                 # Entradas preprocesadas (X_train/X_test y targets)
├── models/preprocessing/           # Transformers guardados (scaler, encoders, info)
├── results/figuras/                # Gráficas generadas (curvas, matriz de confusión)
├── results/tablas/                 # Tablas summary (comparativas u otras)
├── ml-estimation-of-obesity.pdf    # Paper
├── 01_analisis_exploratorio.ipynb
├── 02_preprocesamiento.ipynb
├── 03_entrenamiento_modelos.ipynb
├── 04_reduccion_dimension.ipynb
└── README.md
```

## Consejos para colaborar

- **Mantén consistencia**: no modifiques manualmente los CSV de `data/processed/` si el notebook de preprocesamiento los genera automáticamente.
- **Guarda resultados**: si actualizas modelos, vuelve a correr la celda final del notebook 03 para refrescar `results/figuras/*`.
- **Documenta conclusiones** en los notebooks o en un `docs/` adicional cuando tomes decisiones (por ejemplo, por qué se mantuvo PCA o qué clases quedaron peor).

## Cómo verificar

1. Ejecuta los notebooks en orden (`01` → `02` → `03` → `04`).
2. Verifica que:
   - `data/processed/` contiene los CSV esperados.
   - `models/preprocessing/` tiene los pickle del scaler y encoders.
   - `results/figuras/` contiene las gráficas (curvas, matriz, comparativa).
   - `results/tablas/reduction_comparison.csv` se actualizó después del notebook 04.
