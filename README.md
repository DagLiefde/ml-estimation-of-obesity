# 🏥 Clasificación de Niveles de Obesidad - Regresión Ordinal

Proyecto de Modelos II: Predicción de niveles de obesidad a partir de hábitos alimenticios y condiciones físicas usando Machine Learning.

**Problema**: Regresión Ordinal con 7 clases ordenadas de niveles de obesidad

---

## 📋 Descripción del Proyecto

Este proyecto implementa modelos de Machine Learning para clasificar niveles de obesidad usando regresión ordinal. El dataset contiene información sobre hábitos alimenticios, actividad física y características físicas de individuos.

**Clases de Obesidad** (ordenadas):
1. Insufficient_Weight
2. Normal_Weight
3. Overweight_Level_I
4. Overweight_Level_II
5. Obesity_Type_I
6. Obesity_Type_II
7. Obesity_Type_III

---

## 🚀 Inicio Rápido

### 1. Instalación de Dependencias

```bash
# Crear entorno virtual (si no existe)
python3 -m venv venv

# Activar entorno virtual
source venv/bin/activate  # En macOS/Linux
# o
venv\Scripts\activate  # En Windows

# Instalar dependencias
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 2. Orden de Ejecución de Notebooks

**⚠️ IMPORTANTE**: Ejecuta los notebooks en el siguiente orden:

#### 📊 **Notebook 1: Análisis Exploratorio**
```bash
01_analisis_exploratorio.ipynb
```
**¿Qué hace?**
- Carga y explora el dataset original
- Analiza la distribución de clases de obesidad
- Identifica problemas de desbalance
- Genera visualizaciones y reportes

**Tiempo estimado**: 2-3 minutos

---

#### 🔧 **Notebook 2: Preprocesamiento**
```bash
02_preprocesamiento.ipynb
```
**¿Qué hace?**
- Calcula IMC (Índice de Masa Corporal) a partir de Weight y Height
- Codifica variables categóricas (Label Encoding + One-Hot Encoding)
- Estandariza variables numéricas
- Divide datos en Train (70%) y Test (30%) con estratificación
- Guarda datos preprocesados y transformadores

**⚠️ Requisito**: Debe ejecutarse después del Notebook 1

**Tiempo estimado**: 1-2 minutos

**Archivos generados**:
- `data/processed/X_train.csv`, `X_test.csv`
- `data/processed/y_train.csv`, `y_test.csv`
- `models/preprocessing/*.pkl` (transformadores)

---

#### 🤖 **Notebook 3: Entrenamiento de Modelos** (Próximamente)
```bash
03_entrenamiento_modelos.ipynb
```
**¿Qué hace?**
- Entrena y evalúa 6 modelos diferentes:
  1. Regresión Logística Ordinal
  2. k-NN
  3. Random Forest
  4. Gradient Boosting
  5. Red Neuronal (MLP)
  6. SVM
- Optimiza hiperparámetros con validación cruzada
- Compara resultados entre modelos

**⚠️ Requisito**: Debe ejecutarse después del Notebook 2

---

#### 📉 **Notebook 4: Reducción de Dimensión** (Próximamente)
```bash
04_reduccion_dimension.ipynb
```
**¿Qué hace?**
- Análisis individual de variables (correlación, importancia)
- Aplica PCA (Análisis de Componentes Principales)
- Aplica UMAP (reducción no lineal)
- Compara resultados con y sin reducción

**⚠️ Requisito**: Debe ejecutarse después del Notebook 3

---

#### 📊 **Notebook 5: Evaluación Final** (Próximamente)
```bash
05_evaluacion_final.ipynb
```
**¿Qué hace?**
- Compila todos los resultados
- Genera reportes finales
- Crea visualizaciones comparativas
- Selecciona el mejor modelo

**⚠️ Requisito**: Debe ejecutarse después del Notebook 4

---

## 📁 Estructura del Repositorio

```
ml-estimation-of-obesity/
├── README.md                          # Este archivo
├── LICENSE                            # Licencia MIT
│
├── notebooks/                         # Jupyter notebooks (próximamente)
│   ├── 01_analisis_exploratorio.ipynb
│   ├── 02_preprocesamiento.ipynb
│   ├── 03_entrenamiento_modelos.ipynb
│   ├── 04_reduccion_dimension.ipynb
│   └── 05_evaluacion_final.ipynb
│
├── data/
│   ├── ObesityDataSet_raw_and_data_sinthetic.csv  # Dataset original
│   └── processed/                     # Datos preprocesados
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── models/
│   └── preprocessing/                 # Transformadores guardados
│       ├── scaler.pkl
│       ├── target_encoder.pkl
│       └── label_encoders.pkl
│
├── results/
│   ├── figuras/                       # Visualizaciones
│   └── reportes/                      # Reportes generados
│
└── venv/                              # Entorno virtual (no versionar)
```

---

## 🔄 Flujo de Trabajo Completo

```
1. 01_analisis_exploratorio.ipynb
   ↓
   [Explora datos, identifica problemas]
   ↓
2. 02_preprocesamiento.ipynb
   ↓
   [Prepara datos para modelado]
   ↓
3. 03_entrenamiento_modelos.ipynb
   ↓
   [Entrena y compara modelos]
   ↓
4. 04_reduccion_dimension.ipynb
   ↓
   [Reduce dimensionalidad, evalúa impacto]
   ↓
5. 05_evaluacion_final.ipynb
   ↓
   [Genera reportes finales]
```

---

## 📝 Notas Importantes

### ⚠️ Orden de Ejecución
- **NUNCA** ejecutes los notebooks fuera de orden
- Cada notebook depende de los archivos generados por los anteriores
- Si ejecutas un notebook sin los anteriores, obtendrás errores

### 💾 Datos Preprocesados
- Los datos preprocesados se guardan en `data/processed/`
- Si ya ejecutaste el Notebook 2, puedes saltarlo en ejecuciones futuras
- Los transformadores se guardan en `models/preprocessing/`

### 🔄 Reproducibilidad
- Todos los notebooks usan `random_state=42` para reproducibilidad
- Los resultados deberían ser consistentes entre ejecuciones

---

## 🛠️ Requisitos del Sistema

- Python 3.8+
- Jupyter Notebook o Jupyter Lab
- 4GB+ RAM recomendado
- Espacio en disco: ~100MB

---

## 📚 Dependencias Principales

- `pandas`: Manipulación de datos
- `numpy`: Operaciones numéricas
- `scikit-learn`: Machine Learning
- `matplotlib` / `seaborn`: Visualizaciones
- `jupyter`: Notebooks interactivos

---

## 👥 Autores

- Tomás Cadavid Martínez

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 📖 Documentación Adicional

- `00_PLANIFICACION_PROYECTO.md`: Planificación completa del proyecto
- `00_RESUMEN_EJECUTIVO.md`: Resumen ejecutivo y checklist
- `00_ANALISIS_IMC.md`: Análisis sobre el uso de IMC
- `00_VALIDACION_CRUZADA_VS_VALIDATION_SET.md`: Explicación de metodología

---

## 🆘 Solución de Problemas

### Error: "ModuleNotFoundError"
- Asegúrate de haber activado el entorno virtual
- Instala las dependencias: `pip install -r requirements.txt` (si existe)

### Error: "FileNotFoundError"
- Verifica que ejecutaste los notebooks en orden
- Asegúrate de estar en el directorio raíz del proyecto

### Error: "Data already exists"
- Los datos preprocesados ya existen, puedes continuar con el siguiente notebook
- O elimina `data/processed/` para regenerar

---

**Última actualización**: 2025
