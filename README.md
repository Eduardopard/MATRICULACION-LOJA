# EDA del proceso de matriculación vehicular (ANT) – Hito 2

**UIDE · Escuela de Ciencias de la Computación · Prácticas y Herramientas de Big Data**
Integrantes: Eduardo Pardo, James Romero y Ángel Paladines · Docente: Daniel Patiño

Proyecto Integrador: simulación del proceso de matriculación vehicular en Ecuador (con foco en Loja).
Este repositorio contiene el **Análisis Exploratorio de Datos (EDA)**, la **auditoría de nulos**, el
**análisis multivariado** y la **demostración de The Memory Wall** que justifica pasar a Apache Spark
(Databricks) en la Fase I.

## Datos

| Dataset | Filas | Tamaño | Origen |
|---|---:|---:|---|
| INEC – Vehículos matriculados 2019 (real) | 2.311.960 | 185,7 MB | https://www.ecuadorencifras.gob.ec/estadisticas-de-transporte-2019/ |
| SRI – Vehículos nuevos 2026 (real) | 509.950 | 89,1 MB | https://www.sri.gob.ec/datasets |
| Base ANT sintética (6 tablas) | 21.773.742 | 2.153,8 MB | `src/generar_base_ant.py` (semilla 2026) |

La ANT no publica datos personales (cédulas, nombres, placas). Por eso la **base ANT es sintética**:
cada vehículo hereda las características **reales** de una fila del INEC o del SRI (provincia, marca, clase,
combustible, año, avalúo, cantón), y sobre ellas se simula el proceso 2026 con reglas oficiales
(calendario por dígito de placa, cierre de la ANT del 30-ene-2026, pago 48 h antes, tabla del impuesto
del SRI, tasa RTV de Loja, multas según el SBU de USD 482). **Cédulas, nombres, placas, chasis, teléfonos
y correos son ficticios** (los correos usan el dominio `correo.ejemplo.ec`).

Tablas: `vehiculo`, `propietario`, `matricula`, `revision_vehicular`, `pagos`, `multas`.

> Los CSV pesan más de 100 MB y GitHub no los acepta. No se suben: se descargan las dos fuentes y se
> regeneran con la misma semilla (mismo resultado bit a bit). En `data/muestra/` hay una muestra de 3.000 vehículos de Loja con sus registros en las 6 tablas.

## Cómo reproducir

```bash
pip install -r requirements.txt
# 1) Descargar las dos fuentes y guardarlas como:
#    data/raw/inec_vehiculos_matriculados_2019.csv
#    data/raw/sri_vehiculos_nuevos_2026.csv
# 2) Generar la base ANT sintética (~1 minuto, ~2,2 GB)
python src/generar_base_ant.py --inec data/raw/inec_vehiculos_matriculados_2019.csv \
       --sri data/raw/sri_vehiculos_nuevos_2026.csv --salida data/sintetico --semilla 2026
# 3) Ejecutar los cuadernos (en este orden)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_EDA_Proceso_ANT.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_Memory_Wall_Spark.ipynb
```

Requiere Java 17+ para PySpark local. En Databricks, `spark/fase1_spark_proceso_ant.py` corre igual
cambiando la ruta base a un Volume.

## Estructura

```
data/ref/cantones_provincias.csv   catálogo de 221 cantones y su provincia
data/muestra/                      muestra pequeña (Loja) de las 6 tablas
src/generar_base_ant.py            generador de la base ANT sintética
src/memory_wall.py                 mediciones RAM vs disco y prueba de colapso (con límite de memoria)
spark/fase1_spark_proceso_ant.py   mismo análisis con Spark DataFrames (Fase I)
notebooks/01_EDA_Proceso_ANT.ipynb perfilado, nulos, correlaciones, dependencias y outliers
notebooks/02_Memory_Wall_Spark.ipynb Memory Wall y prueba con Spark
resultados/tablas, resultados/figuras  salidas que usa el informe
docs/                              informe del Hito 2 (PDF y DOCX)
```

## Supuestos de la simulación (no publicados oficialmente)

- Tasa de matrícula ANT: USD 27 livianos, 15 motos, 40 pesados (valor referencial).
- Tasa vial provincial: USD 5 livianos y 10 pesados (se tomó el valor de un cantón como referencia).
- Proporción de rezagados, abandonos y errores de digitación: elegidos para que el EDA tenga casos reales que detectar.
