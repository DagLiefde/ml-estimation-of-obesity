# 📋 PLANIFICACIÓN COMPLETA DEL PROYECTO
## Clasificación de Niveles de Obesidad - Regresión Ordinal

---

## 🎯 OBJETIVO DEL PROYECTO

Desarrollar un sistema de clasificación ordinal para predecir niveles de obesidad a partir de hábitos alimenticios y condiciones físicas, utilizando técnicas de Machine Learning.

**Problema**: Regresión Ordinal (7 clases ordenadas)
- Insufficient_Weight
- Normal_Weight
- Overweight_Level_I
- Overweight_Level_II
- Obesity_Type_I
- Obesity_Type_II
- Obesity_Type_III

---

## 📊 ESTRUCTURA DEL PROYECTO

### ✅ COMPLETADO
- [x] **Sección 1-3**: Análisis Exploratorio de Datos (EDA)
  - Caracterización del dataset
  - Distribución de clases
  - Identificación de desbalance
  - Visualizaciones

### 🔄 EN PROGRESO / PENDIENTE

---

## 📑 SECCIÓN 4: ENTRENAMIENTO Y EVALUACIÓN DE MODELOS (30%)

### 4.1 Configuración Experimental

#### **4.1.1 Metodología de Validación**

**Tipo de Partición:**
- **Estrategia**: Validación cruzada estratificada (Stratified K-Fold) + conjunto de test independiente
- **Justificación**: 
  - Para regresión ordinal, necesitamos mantener la distribución de clases en cada fold
  - El conjunto de test se reserva para evaluación final (no se usa en entrenamiento/validación)
  - K-Fold (K=5 o K=10) permite mejor uso de los datos disponibles

**División de Datos:**
```
Total: 2111 registros
├── Train (60%): ~1267 registros
├── Validation (20%): ~422 registros  
└── Test (20%): ~422 registros
```

**Técnicas de Balanceo (si se requieren):**
- **SMOTE (Synthetic Minority Oversampling Technique)**: Para generar muestras sintéticas de clases minoritarias
- **ADASYN**: Variante adaptativa de SMOTE
- **Undersampling**: Reducir clases mayoritarias (solo si es necesario)
- **Aplicación**: Solo en el conjunto de entrenamiento, NO en validación ni test

**Justificación del Esquema:**
- Validación cruzada estratificada asegura representatividad de todas las clases
- Conjunto de test independiente evita sobreajuste
- Balanceo solo en entrenamiento mantiene realismo en evaluación

#### **4.1.2 Tabla de Hiperparámetros por Modelo**

| Modelo | Hiperparámetro | Rango/Valores | Tipo de Búsqueda | Justificación |
|--------|---------------|---------------|------------------|---------------|
| **1. Regresión Logística Ordinal** | `C` (regularización) | [0.001, 0.01, 0.1, 1, 10, 100, 1000] | Grid Search | Controla el trade-off entre ajuste y generalización |
| | `solver` | ['lbfgs', 'liblinear', 'saga'] | Grid Search | Diferentes algoritmos de optimización |
| | `max_iter` | [100, 500, 1000, 2000] | Grid Search | Número máximo de iteraciones |
| | `class_weight` | ['balanced', None] | Grid Search | Manejo de clases desbalanceadas |
| **2. k-NN (k-Nearest Neighbors)** | `n_neighbors` | [3, 5, 7, 9, 11, 15, 20] | Grid Search | Número de vecinos (afecta suavizado) |
| | `weights` | ['uniform', 'distance'] | Grid Search | Peso de los vecinos en la predicción |
| | `metric` | ['euclidean', 'manhattan', 'minkowski'] | Grid Search | Distancia entre puntos |
| | `p` (para Minkowski) | [1, 2] | Grid Search | Parámetro de distancia Minkowski |
| **3. Random Forest** | `n_estimators` | [50, 100, 200, 300, 500] | Random Search | Número de árboles en el ensamble |
| | `max_depth` | [5, 10, 15, 20, None] | Random Search | Profundidad máxima de árboles |
| | `min_samples_split` | [2, 5, 10, 20] | Random Search | Mínimo de muestras para dividir |
| | `min_samples_leaf` | [1, 2, 4, 8] | Random Search | Mínimo de muestras en hoja |
| | `max_features` | ['sqrt', 'log2', None] | Random Search | Características consideradas por split |
| | `class_weight` | ['balanced', 'balanced_subsample', None] | Random Search | Manejo de desbalance |
| **4. Gradient Boosting** | `n_estimators` | [50, 100, 200, 300] | Random Search | Número de estimadores |
| | `learning_rate` | [0.01, 0.05, 0.1, 0.2] | Random Search | Tasa de aprendizaje |
| | `max_depth` | [3, 5, 7, 10] | Random Search | Profundidad de árboles base |
| | `min_samples_split` | [2, 5, 10] | Random Search | Mínimo para dividir |
| | `subsample` | [0.8, 0.9, 1.0] | Random Search | Fracción de muestras por árbol |
| **5. Red Neuronal (MLP)** | `hidden_layer_sizes` | [(50,), (100,), (50,50), (100,50), (100,100)] | Random Search | Arquitectura de la red |
| | `activation` | ['relu', 'tanh', 'logistic'] | Random Search | Función de activación |
| | `alpha` (regularización) | [0.0001, 0.001, 0.01, 0.1] | Random Search | Penalización L2 |
| | `learning_rate` | ['constant', 'adaptive'] | Random Search | Tipo de tasa de aprendizaje |
| | `max_iter` | [200, 500, 1000] | Random Search | Iteraciones máximas |
| **6. SVM (Support Vector Machine)** | `C` | [0.1, 1, 10, 100, 1000] | Grid Search | Parámetro de regularización |
| | `kernel` | ['linear', 'rbf', 'poly', 'sigmoid'] | Grid Search | Tipo de kernel |
| | `gamma` (para RBF) | ['scale', 'auto', 0.001, 0.01, 0.1, 1] | Grid Search | Coeficiente del kernel |
| | `degree` (para poly) | [2, 3, 4, 5] | Grid Search | Grado del polinomio |
| | `class_weight` | ['balanced', None] | Grid Search | Manejo de desbalance |

**Nota**: Para regresión ordinal, algunos modelos necesitarán adaptación:
- Usar `OrdinalLogisticRegression` o convertir a problema de clasificación ordinal
- Considerar métricas específicas para datos ordinales

#### **4.1.3 Métricas de Desempeño**

**Métricas para Regresión Ordinal:**

| Métrica | Descripción | Relevancia | Limitaciones |
|---------|-------------|------------|--------------|
| **Accuracy** | Porcentaje de predicciones correctas | Útil para comparación general | No considera el orden (clasificar Obesity_Type_I como Obesity_Type_III es igual de malo que clasificarlo como Normal_Weight) |
| **Mean Absolute Error (MAE) Ordinal** | Distancia promedio en la escala ordinal | **MUY RELEVANTE**: Considera el orden de las clases | No penaliza igual errores grandes vs pequeños |
| **Mean Squared Error (MSE) Ordinal** | Distancia cuadrática promedio | Penaliza más los errores grandes | Puede ser sensible a outliers |
| **Matriz de Confusión Ordinal** | Tabla de predicciones vs reales | Visualiza patrones de error | Difícil de interpretar con muchas clases |
| **Cohen's Kappa** | Acuerdo corregido por azar | Considera el desbalance | No considera el orden ordinal |
| **Spearman Correlation** | Correlación de rangos | Considera el orden | No es una métrica de error directa |
| **Macro/Micro F1-Score** | F1 promedio por clase | Útil para clases desbalanceadas | No considera el orden |

**Métricas Recomendadas:**
1. **MAE Ordinal** (principal)
2. **Accuracy** (secundaria)
3. **Matriz de Confusión** (visualización)
4. **Spearman Correlation** (validación del orden)

### 4.2 Resultados del Entrenamiento

#### **4.2.1 Tablas de Resultados Requeridas**

**Tabla 1: Resultados por Modelo (Validación Cruzada)**
| Modelo | MAE Ordinal | Accuracy | Spearman ρ | Tiempo (s) |
|--------|-------------|----------|------------|------------|
| Regresión Logística Ordinal | X.XX ± Y.YY | X.XX% ± Y.YY% | X.XX | XX |
| k-NN | ... | ... | ... | ... |
| Random Forest | ... | ... | ... | ... |
| Gradient Boosting | ... | ... | ... | ... |
| MLP | ... | ... | ... | ... |
| SVM | ... | ... | ... | ... |

**Tabla 2: Mejores Hiperparámetros por Modelo**
| Modelo | Hiperparámetros Óptimos |
|--------|-------------------------|
| Regresión Logística | C=X, solver='...', ... |
| ... | ... |

**Tabla 3: Resultados Train/Validation/Test (Mejores 2 Modelos)**
| Modelo | Conjunto | MAE Ordinal | Accuracy | F1-Macro |
|--------|----------|-------------|----------|----------|
| Modelo 1 | Train | ... | ... | ... |
| | Validation | ... | ... | ... |
| | Test | ... | ... | ... |
| Modelo 2 | Train | ... | ... | ... |
| | Validation | ... | ... | ... |
| | Test | ... | ... | ... |

#### **4.2.2 Visualizaciones Requeridas**

1. **Curvas de Aprendizaje** (Learning Curves)
   - Eje X: Tamaño del conjunto de entrenamiento
   - Eje Y: MAE Ordinal / Accuracy
   - Líneas: Train score vs Validation score
   - Caption: "Curvas de aprendizaje para [Modelo]. Se observa..."

2. **Matrices de Confusión Ordinales**
   - Una por cada modelo (o al menos los mejores 2)
   - Caption: "Matriz de confusión para [Modelo] en el conjunto de test..."

3. **Comparación de Métricas por Modelo**
   - Gráfico de barras comparando MAE, Accuracy, etc.
   - Caption: "Comparación de desempeño entre modelos..."

4. **Heatmap de Hiperparámetros**
   - Para modelos con 2 hiperparámetros principales
   - Caption: "Efecto de hiperparámetros [X] y [Y] en el desempeño..."

---

## 📉 SECCIÓN 5: REDUCCIÓN DE DIMENSIÓN (20%)

### 5.1 Análisis Individual de Variables

#### **5.1.1 Métodos de Análisis**

**1. Correlación**
- Matriz de correlación de Pearson (variables numéricas)
- Correlación punto-biserial (numérica vs categórica)
- Identificar variables altamente correlacionadas (>0.8 o <-0.8)

**2. Índice de Discriminación**
- Para cada variable, calcular capacidad de discriminar entre clases
- Usar ANOVA F-statistic o Chi-square según tipo de variable
- Ranking de variables por poder discriminativo

**3. Importancia de Características**
- Usar Random Forest para obtener feature importance
- Usar Permutation Importance
- Comparar ambos métodos

**Conclusión**: Identificar variables que pueden eliminarse sin pérdida significativa de información.

### 5.2 Extracción de Características Lineal (PCA)

#### **5.2.1 Criterio para Número de Componentes**

**Métodos:**
1. **Varianza Explicada Acumulada**: Mantener componentes que expliquen ≥95% de varianza
2. **Criterio de Kaiser**: Componentes con eigenvalue > 1
3. **Scree Plot**: Punto de inflexión en la curva
4. **Validación Cruzada**: Probar diferentes números y elegir el que maximice métricas

**Justificación Matemática:**
- PCA busca maximizar varianza explicada
- Trade-off: Más componentes = más información pero más complejidad
- Objetivo: Reducir dimensionalidad manteniendo ≥95% de varianza

#### **5.2.2 Tabla de Resultados PCA**

| Número de Componentes | Varianza Explicada | Reducción Dimensionalidad | MAE Ordinal (Modelo 1) | MAE Ordinal (Modelo 2) | Accuracy (Modelo 1) | Accuracy (Modelo 2) |
|----------------------|-------------------|---------------------------|------------------------|------------------------|---------------------|---------------------|
| Original (16) | 100% | 0% | X.XX | X.XX | X.XX% | X.XX% |
| 10 | XX% | XX% | X.XX | X.XX | X.XX% | X.XX% |
| 8 | XX% | XX% | X.XX | X.XX | X.XX% | X.XX% |
| 5 | XX% | XX% | X.XX | X.XX | X.XX% | X.XX% |

**Evaluación**: Comparar los 2 mejores modelos de la Sección 4 con y sin PCA.

### 5.3 Extracción de Características No Lineal (UMAP)

#### **5.3.1 Criterio para Número de Componentes**

**Parámetros UMAP:**
- `n_components`: [2, 3, 5, 8, 10] (probar diferentes)
- `n_neighbors`: [5, 10, 15, 30, 50] (afecta estructura local vs global)
- `min_dist`: [0.0, 0.1, 0.5] (separación entre puntos)

**Criterio de Selección:**
- Validación cruzada con métricas de desempeño
- Visualización 2D/3D para verificar estructura
- Comparación con PCA

**Justificación Técnica:**
- UMAP preserva estructura no lineal mejor que PCA
- Útil cuando hay relaciones complejas entre variables
- Puede revelar clusters no visibles con métodos lineales

#### **5.3.2 Tabla Comparativa Final**

| Método | Número de Componentes | Reducción | MAE Ordinal (M1) | MAE Ordinal (M2) | Accuracy (M1) | Accuracy (M2) |
|--------|----------------------|-----------|------------------|------------------|---------------|---------------|
| Sin Reducción | 16 | 0% | X.XX | X.XX | X.XX% | X.XX% |
| PCA | X | XX% | X.XX | X.XX | X.XX% | X.XX% |
| UMAP | X | XX% | X.XX | X.XX | X.XX% | X.XX% |

**Evaluación**: Usar los 2 mejores modelos de la Sección 4.

#### **5.3.3 Discusión y Conclusiones**

**Temas a cubrir:**
1. **Resultados con y sin reducción**
   - ¿Mejora o empeora el desempeño?
   - ¿Vale la pena la reducción?

2. **PCA vs UMAP**
   - ¿Cuál preserva mejor la información?
   - ¿Cuál es más interpretable?
   - ¿Cuál es más rápido?

3. **Hallazgos respecto al estado del arte**
   - Comparar con literatura
   - ¿Qué técnicas funcionan mejor para regresión ordinal?

4. **Riesgos, Fortalezas y Limitaciones**
   - Riesgos: Pérdida de información, sobreajuste
   - Fortalezas: Reducción de complejidad, visualización
   - Limitaciones: Interpretabilidad, tiempo de cómputo

---

## 🎯 SECCIÓN 6: EVALUACIÓN FINAL (20%)

### 6.1 Qué Evaluar

**Componentes del Proyecto:**
1. **Calidad del Código**
   - Organización y estructura
   - Comentarios y documentación
   - Reproducibilidad

2. **Metodología Científica**
   - Rigor en experimentación
   - Validación adecuada
   - Interpretación de resultados

3. **Resultados y Análisis**
   - Comparación de modelos
   - Análisis de reducción de dimensión
   - Conclusiones fundamentadas

4. **Presentación**
   - Claridad del informe
   - Visualizaciones apropiadas
   - Video de sustentación

### 6.2 Estructura del Informe (Máximo 10 páginas)

**Distribución Sugerida:**
1. **Resumen/Abstract** (0.5 páginas)
2. **Introducción** (0.5 páginas)
3. **Metodología** (1 página)
4. **Análisis Exploratorio** (1 página)
5. **Entrenamiento y Evaluación** (3 páginas)
   - Configuración experimental
   - Resultados por modelo
   - Comparaciones
6. **Reducción de Dimensión** (2 páginas)
   - Análisis individual
   - PCA
   - UMAP
   - Discusión
7. **Conclusiones** (1 página)
8. **Referencias** (0.5 páginas)
9. **Apéndices** (si hay espacio)

**Consejos para no exceder:**
- Usar tablas compactas
- Figuras pequeñas pero claras
- Texto conciso y directo
- Eliminar redundancias

### 6.3 Estructura del Repositorio GitHub

```
ml-estimation-of-obesity/
├── README.md                 # Descripción del proyecto, instalación, uso
├── requirements.txt          # Dependencias de Python
├── .gitignore               # Archivos a ignorar
├── LICENSE                   # Licencia
│
├── data/
│   └── ObesityDataSet_raw_and_data_sinthetic.csv
│
├── notebooks/               # Jupyter notebooks reproducibles
│   ├── 01_analisis_exploratorio.ipynb
│   ├── 02_preprocesamiento.ipynb
│   ├── 03_entrenamiento_modelos.ipynb
│   ├── 04_reduccion_dimension.ipynb
│   └── 05_evaluacion_final.ipynb
│
├── scripts/                 # Scripts Python ejecutables
│   ├── 01_analisis_exploratorio.py
│   ├── 02_preprocesamiento.py
│   ├── 03_entrenamiento.py
│   └── 04_reduccion_dimension.py
│
├── models/                  # Modelos entrenados (si se guardan)
│   └── .gitkeep
│
├── results/                 # Resultados y visualizaciones
│   ├── figuras/
│   ├── tablas/
│   └── reportes/
│
├── docs/                    # Documentación adicional
│   └── informe_final.pdf
│
└── tests/                   # Tests unitarios (opcional)
    └── test_preprocesamiento.py
```

**README.md debe incluir:**
- Descripción del proyecto
- Instalación de dependencias
- Cómo ejecutar los scripts
- Estructura del repositorio
- Autores y licencia

### 6.4 Video de Sustentación (10 minutos)

**Estructura Sugerida:**

1. **Introducción** (1 min)
   - Problema a resolver
   - Objetivos del proyecto

2. **Dataset y Análisis Exploratorio** (1.5 min)
   - Características del dataset
   - Distribución de clases
   - Hallazgos principales

3. **Metodología** (1.5 min)
   - Estrategia de validación
   - Modelos evaluados
   - Métricas utilizadas

4. **Resultados de Modelos** (3 min)
   - Mejores modelos
   - Comparaciones
   - Visualizaciones clave

5. **Reducción de Dimensión** (2 min)
   - PCA y UMAP
   - Comparación de resultados
   - Conclusiones

6. **Conclusiones y Trabajo Futuro** (1 min)
   - Hallazgos principales
   - Limitaciones
   - Mejoras posibles

**Consejos:**
- Practicar el timing
- Usar visualizaciones claras
- Hablar con claridad
- Preparar respuestas a preguntas comunes

### 6.5 Preguntas Típicas en la Presentación

**Sobre Metodología:**
- ¿Por qué elegiste validación cruzada en lugar de train/test simple?
- ¿Por qué no aplicaste balanceo si hay desbalance?
- ¿Cómo manejaste el problema de regresión ordinal?

**Sobre Modelos:**
- ¿Por qué el modelo X tuvo mejor desempeño que Y?
- ¿Qué hiperparámetros fueron más importantes?
- ¿Hay evidencia de sobreajuste?

**Sobre Reducción de Dimensión:**
- ¿Por qué PCA mejoró/empeoró los resultados?
- ¿Qué ventajas tiene UMAP sobre PCA?
- ¿Qué información se perdió con la reducción?

**Sobre Resultados:**
- ¿Los resultados son estadísticamente significativos?
- ¿Cómo se compara con el estado del arte?
- ¿Qué limitaciones tiene tu enfoque?

**Técnicas:**
- ¿Cómo funciona [técnica X]?
- ¿Por qué no probaste [técnica Y]?
- ¿Qué harías diferente si tuvieras más tiempo?

---

## 📝 CHECKLIST DE ENTREGABLES

### Código y Repositorio
- [ ] Repositorio GitHub organizado
- [ ] README completo y claro
- [ ] Código comentado y reproducible
- [ ] Requirements.txt actualizado
- [ ] .gitignore configurado

### Análisis y Modelos
- [ ] Scripts de análisis exploratorio ejecutados
- [ ] Preprocesamiento de datos completo
- [ ] 5+ modelos entrenados y evaluados
- [ ] Hiperparámetros optimizados
- [ ] Métricas calculadas para todos los modelos
- [ ] Análisis de reducción de dimensión (PCA y UMAP)

### Documentación
- [ ] Informe final (máximo 10 páginas)
- [ ] Todas las tablas requeridas
- [ ] Todas las figuras con captions
- [ ] Referencias bibliográficas

### Presentación
- [ ] Video de 10 minutos grabado
- [ ] Slides de apoyo (opcional)
- [ ] Preparación para preguntas

---

## 🚀 PRÓXIMOS PASOS INMEDIATOS

1. **Preprocesamiento de Datos**
   - Encoding de variables categóricas
   - Normalización/estandarización
   - División train/validation/test

2. **Implementación de Modelos Base**
   - Configurar pipeline de entrenamiento
   - Implementar validación cruzada
   - Crear funciones de evaluación

3. **Optimización de Hiperparámetros**
   - Grid Search / Random Search
   - Guardar mejores modelos
   - Generar tablas de resultados

4. **Análisis de Reducción de Dimensión**
   - Análisis individual de variables
   - Implementar PCA
   - Implementar UMAP

5. **Generación de Reporte**
   - Compilar resultados
   - Crear visualizaciones finales
   - Escribir informe

---

**Última actualización**: [Fecha]
**Estado**: Planificación completa - Listo para implementación




