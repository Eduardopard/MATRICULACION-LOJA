"""
Fase I (Semana 05) - Mismo análisis que colapsa en pandas, ahora con Apache Spark DataFrames.

En Databricks: subir los CSV a un Volume/DBFS y cambiar BASE (la variable `spark` ya existe).
En local:      python spark/fase1_spark_proceso_ant.py --base data/sintetico --memoria 2g
"""
import argparse, json, time

def run(spark, base):
    from pyspark.sql import functions as F
    t0 = time.time()
    leer = lambda n: spark.read.option("header", True).option("inferSchema", True).csv(f"{base}/{n}.csv")
    v, m, p, mu = leer("vehiculo"), leer("matricula"), leer("pagos"), leer("multas")
    m = m.select("vehiculo_id", "estado", "fecha_matriculacion")
    p = p.select("vehiculo_id", F.col("tipo").alias("tipo_pago"), F.col("valor").alias("valor_pago"))
    mu = mu.select("vehiculo_id", F.col("estado").alias("estado_multa"), F.col("valor").alias("valor_multa"))
    x = (v.join(m, v.id == m.vehiculo_id).drop(m.vehiculo_id)
           .join(p, v.id == p.vehiculo_id).drop(p.vehiculo_id)
           .join(mu, v.id == mu.vehiculo_id, "left").drop(mu.vehiculo_id))
    g = (x.groupBy("provincia", "canton", "marca", "clase", "tipo", "estado", "estado_multa")
           .agg(F.sum("valor_pago").alias("total_pagos"), F.sum("valor_multa").alias("total_multas"),
                F.count("*").alias("n")))
    n_grupos = g.count()
    filas_union = x.count()
    loja = (g.filter(F.col("provincia") == "LOJA").groupBy("canton")
              .agg(F.sum("total_pagos").alias("recaudado")).orderBy(F.desc("recaudado")).limit(5).collect())
    return {"resultado": "COMPLETO", "grupos": n_grupos, "filas_union": filas_union,
            "particiones_pagos": p.rdd.getNumPartitions(), "seg": round(time.time() - t0, 1),
            "top_loja": [(r["canton"], round(r["recaudado"], 2)) for r in loja]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/sintetico"); ap.add_argument("--memoria", default="2g")
    a = ap.parse_args()
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.master("local[2]").appName("memory-wall-ant")
             .config("spark.driver.memory", a.memoria).config("spark.sql.shuffle.partitions", "64")
             .config("spark.ui.enabled", "false").config("spark.ui.showConsoleProgress", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    r = run(spark, a.base)
    r["memoria_driver"] = a.memoria
    print(json.dumps(r, ensure_ascii=False))
    spark.stop()
