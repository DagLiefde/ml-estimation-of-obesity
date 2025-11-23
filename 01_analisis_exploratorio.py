"""
Script de Análisis Exploratorio de Datos (EDA)
Problema: Regresión Ordinal para Clasificación de Niveles de Obesidad

Este script tiene como objetivo:
1. Cargar y entender la estructura del dataset
2. Analizar la distribución de las clases de obesidad
3. Identificar el problema de desbalance
4. Generar visualizaciones y reportes
"""

# ============================================================================
# PASO 1: IMPORTAR LIBRERÍAS NECESARIAS
# ============================================================================
"""
¿Por qué estas librerías?
- pandas: Para manipulación y análisis de datos estructurados
- numpy: Para operaciones matemáticas y estadísticas
- matplotlib y seaborn: Para crear visualizaciones
- warnings: Para suprimir mensajes innecesarios
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Configuración para mejorar la visualización
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# ============================================================================
# PASO 2: CARGAR EL DATASET
# ============================================================================
"""
¿Qué hace este paso?
- Lee el archivo CSV con los datos
- Crea un DataFrame de pandas para trabajar con los datos
- Verifica que se cargó correctamente
"""

print("=" * 80)
print("PASO 2: CARGANDO EL DATASET")
print("=" * 80)

# Cargar el dataset
df = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# Mostrar información básica del dataset
print(f"\n✓ Dataset cargado exitosamente")
print(f"  - Número de filas: {len(df)}")
print(f"  - Número de columnas: {len(df.columns)}")
print(f"\nPrimeras 5 filas del dataset:")
print(df.head())

print(f"\nInformación general del dataset:")
print(df.info())

# ============================================================================
# PASO 3: IDENTIFICAR LA VARIABLE OBJETIVO
# ============================================================================
"""
¿Qué hace este paso?
- Identifica la columna que contiene las clases de obesidad (NObeyesdad)
- Verifica que existe y tiene datos
"""

print("\n" + "=" * 80)
print("PASO 3: IDENTIFICANDO LA VARIABLE OBJETIVO")
print("=" * 80)

# La variable objetivo es 'NObeyesdad'
target_column = 'NObeyesdad'

# Verificar que la columna existe
if target_column in df.columns:
    print(f"\n✓ Variable objetivo encontrada: '{target_column}'")
    print(f"  - Tipo de datos: {df[target_column].dtype}")
    print(f"  - Valores únicos: {df[target_column].nunique()}")
    print(f"\nClases de obesidad encontradas:")
    print(df[target_column].unique())
else:
    print(f"\n✗ Error: La columna '{target_column}' no existe en el dataset")
    print(f"Columnas disponibles: {list(df.columns)}")

# ============================================================================
# PASO 4: ANALIZAR LA DISTRIBUCIÓN DE CLASES
# ============================================================================
"""
¿Qué hace este paso?
- Cuenta cuántos registros hay en cada clase de obesidad
- Calcula porcentajes para entender la proporción de cada clase
- Identifica si hay clases con muy pocos o muchos registros (desbalance)
"""

print("\n" + "=" * 80)
print("PASO 4: ANALIZANDO LA DISTRIBUCIÓN DE CLASES")
print("=" * 80)

# Contar la frecuencia de cada clase
distribucion = df[target_column].value_counts().sort_index()

# Calcular porcentajes
porcentajes = (df[target_column].value_counts(normalize=True) * 100).sort_index()

# Crear un DataFrame con la información
distribucion_df = pd.DataFrame({
    'Clase': distribucion.index,
    'Cantidad': distribucion.values,
    'Porcentaje': porcentajes.values
})

print("\n" + "-" * 80)
print("DISTRIBUCIÓN DE CLASES DE OBESIDAD")
print("-" * 80)
print(distribucion_df.to_string(index=False))

# Estadísticas adicionales
print("\n" + "-" * 80)
print("ESTADÍSTICAS DEL DESBALANCE")
print("-" * 80)
print(f"Total de registros: {len(df)}")
print(f"Número de clases: {len(distribucion)}")
print(f"Clase con MÁS registros: {distribucion.idxmax()} ({distribucion.max()} registros)")
print(f"Clase con MENOS registros: {distribucion.idxmin()} ({distribucion.min()} registros)")

# Calcular el ratio de desbalance (Imbalance Ratio - IR)
# IR = tamaño de la clase mayoritaria / tamaño de la clase minoritaria
clase_mayoritaria = distribucion.max()
clase_minoritaria = distribucion.min()
imbalance_ratio = clase_mayoritaria / clase_minoritaria

print(f"\nRatio de Desbalance (IR): {imbalance_ratio:.2f}")
print(f"  - Esto significa que la clase mayoritaria tiene {imbalance_ratio:.2f}x más registros")
print(f"  - que la clase minoritaria")

# Clasificar el nivel de desbalance
if imbalance_ratio < 2:
    nivel = "BALANCEADO"
elif imbalance_ratio < 5:
    nivel = "DESBALANCEO LEVE"
elif imbalance_ratio < 10:
    nivel = "DESBALANCEO MODERADO"
else:
    nivel = "DESBALANCEO SEVERO"

print(f"\nNivel de desbalance: {nivel}")

# ============================================================================
# PASO 5: CREAR VISUALIZACIONES
# ============================================================================
"""
¿Qué hace este paso?
- Crea gráficos para visualizar la distribución de clases
- Facilita entender el problema de desbalance de forma visual
"""

print("\n" + "=" * 80)
print("PASO 5: CREANDO VISUALIZACIONES")
print("=" * 80)

# Crear figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Análisis de Distribución de Clases de Obesidad', fontsize=16, fontweight='bold')

# 1. Gráfico de barras - Cantidad de registros por clase
ax1 = axes[0, 0]
distribucion.plot(kind='bar', ax=ax1, color='steelblue', edgecolor='black')
ax1.set_title('Distribución de Clases (Cantidad de Registros)', fontweight='bold')
ax1.set_xlabel('Clase de Obesidad', fontweight='bold')
ax1.set_ylabel('Cantidad de Registros', fontweight='bold')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)

# Agregar valores en las barras
for i, v in enumerate(distribucion.values):
    ax1.text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')

# 2. Gráfico de barras - Porcentajes
ax2 = axes[0, 1]
porcentajes.plot(kind='bar', ax=ax2, color='coral', edgecolor='black')
ax2.set_title('Distribución de Clases (Porcentajes)', fontweight='bold')
ax2.set_xlabel('Clase de Obesidad', fontweight='bold')
ax2.set_ylabel('Porcentaje (%)', fontweight='bold')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)

# Agregar valores en las barras
for i, v in enumerate(porcentajes.values):
    ax2.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

# 3. Gráfico de pastel (pie chart)
ax3 = axes[1, 0]
colors = sns.color_palette("Set3", len(distribucion))
wedges, texts, autotexts = ax3.pie(distribucion.values, 
                                    labels=distribucion.index,
                                    autopct='%1.1f%%',
                                    colors=colors,
                                    startangle=90)
ax3.set_title('Distribución de Clases (Gráfico Circular)', fontweight='bold')

# Mejorar la legibilidad del gráfico de pastel
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')

# 4. Gráfico de barras horizontal ordenado
ax4 = axes[1, 1]
distribucion_sorted = distribucion.sort_values(ascending=True)
distribucion_sorted.plot(kind='barh', ax=ax4, color='mediumseagreen', edgecolor='black')
ax4.set_title('Distribución Ordenada (de Menor a Mayor)', fontweight='bold')
ax4.set_xlabel('Cantidad de Registros', fontweight='bold')
ax4.set_ylabel('Clase de Obesidad', fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

# Agregar valores en las barras
for i, v in enumerate(distribucion_sorted.values):
    ax4.text(v + 10, i, str(v), ha='left', va='center', fontweight='bold')

plt.tight_layout()

# Guardar la figura
nombre_archivo = '01_distribucion_clases_obesidad.png'
plt.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
print(f"\n✓ Visualizaciones guardadas en: {nombre_archivo}")

# Mostrar el gráfico
plt.show()

# ============================================================================
# PASO 6: GENERAR REPORTE DE CARACTERIZACIÓN
# ============================================================================
"""
¿Qué hace este paso?
- Crea un archivo de texto con un reporte completo de la caracterización
- Documenta todos los hallazgos importantes
"""

print("\n" + "=" * 80)
print("PASO 6: GENERANDO REPORTE DE CARACTERIZACIÓN")
print("=" * 80)

# Crear el reporte
reporte = f"""
{'='*80}
REPORTE DE CARACTERIZACIÓN DEL DATASET
Problema: Regresión Ordinal - Clasificación de Niveles de Obesidad
{'='*80}

1. INFORMACIÓN GENERAL DEL DATASET
   - Total de registros: {len(df)}
   - Total de columnas: {len(df.columns)}
   - Variable objetivo: {target_column}
   - Número de clases: {len(distribucion)}

2. DISTRIBUCIÓN DE CLASES DE OBESIDAD
   (Ordenadas según su orden ordinal)

"""

# Agregar información de cada clase
orden_ordinal = [
    'Insufficient_Weight',
    'Normal_Weight', 
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
]

reporte += "\n   Clase                    | Cantidad | Porcentaje | Estado\n"
reporte += "   " + "-"*70 + "\n"

for clase in orden_ordinal:
    if clase in distribucion.index:
        cantidad = distribucion[clase]
        porcentaje = porcentajes[clase]
        
        # Clasificar el estado de la clase
        if porcentaje < 10:
            estado = "CRÍTICA (muy pocos datos)"
        elif porcentaje < 15:
            estado = "BAJA"
        elif porcentaje < 25:
            estado = "MODERADA"
        else:
            estado = "ALTA"
        
        reporte += f"   {clase:24} | {cantidad:8} | {porcentaje:9.2f}% | {estado}\n"

reporte += f"""
3. ANÁLISIS DE DESBALANCE

   Clase Mayoritaria: {distribucion.idxmax()}
   - Cantidad: {clase_mayoritaria} registros
   - Porcentaje: {porcentajes[distribucion.idxmax()]:.2f}%

   Clase Minoritaria: {distribucion.idxmin()}
   - Cantidad: {clase_minoritaria} registros
   - Porcentaje: {porcentajes[distribucion.idxmin()]:.2f}%

   Ratio de Desbalance (IR): {imbalance_ratio:.2f}
   Nivel de Desbalance: {nivel}

4. PROBLEMAS IDENTIFICADOS

   ✓ La caracterización de la base de datos ahora está completa
   ✓ Se ha identificado el problema de desbalance:
     - El ratio de desbalance es {imbalance_ratio:.2f}
     - Esto indica que hay una diferencia significativa entre las clases
     - Las clases con menos datos pueden tener peor rendimiento en el modelo

5. RECOMENDACIONES

   - Implementar técnicas de balanceo de datos (SMOTE, ADASYN, etc.)
   - Usar métricas de evaluación apropiadas para datos desbalanceados
   - Considerar pesos de clase en los modelos
   - Usar validación cruzada estratificada
   - Considerar modelos específicos para regresión ordinal

{'='*80}
Reporte generado automáticamente
{'='*80}
"""

# Guardar el reporte
nombre_reporte = '01_reporte_caracterizacion.txt'
with open(nombre_reporte, 'w', encoding='utf-8') as f:
    f.write(reporte)

print(f"\n✓ Reporte guardado en: {nombre_reporte}")

# Mostrar el reporte en consola
print(reporte)

print("\n" + "=" * 80)
print("ANÁLISIS EXPLORATORIO COMPLETADO")
print("=" * 80)
print("\nPróximos pasos sugeridos:")
print("  1. Revisar las visualizaciones generadas")
print("  2. Leer el reporte de caracterización")
print("  3. Proceder con técnicas de balanceo de datos")
print("  4. Implementar modelos de regresión ordinal")

