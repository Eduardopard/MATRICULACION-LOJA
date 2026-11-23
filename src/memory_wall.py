"""
Demostración de The Memory Wall con pandas.

Cada prueba corre en un proceso aparte con un LÍMITE DE MEMORIA (RLIMIT_AS), para simular
la RAM libre de una laptop de estudiante sin tumbar la máquina:

  python src/memory_wall.py medir   --csv data/sintetico/pagos.csv  [--sep ,] [--limite-gb 6]
  python src/memory_wall.py colapso --dir data/sintetico            [--limite-gb 6]

La salida es una línea JSON con los resultados.
"""
import argparse, json, os, resource, sys, time

def limitar(gb):
    b = int(gb * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (b, b))

def rss_pico_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB -> MB

def medir(csv, sep, enc, objeto):
    import pandas as pd
    if objeto:                       # comportamiento clásico (pandas 1.x/2.x, Colab): textos como objetos de Python
        pd.set_option("future.infer_string", False)
    import psutil, gc
    proc = psutil.Process()
    disco = os.path.getsize(csv) / 1e6
    gc.collect(); base = proc.memory_info().rss
    t = time.time()
    df = pd.read_csv(csv, sep=sep, encoding=enc, low_memory=False)          # carga "ingenua" por defecto
    seg = time.time() - t
    gc.collect(); reposo = (proc.memory_info().rss - base) / 1e6    # RAM real ocupada por el DataFrame ya cargado
    ram = df.memory_usage(deep=True).sum() / 1e6                       # estimación de pandas
    # versión optimizada: textos repetidos como 'category', números con el tipo más pequeño
    opt = df.copy()
    for c in opt.columns:
        if (opt[c].dtype == object or str(opt[c].dtype) == "str") and opt[c].nunique(dropna=True) < 0.5 * len(opt):
            opt[c] = opt[c].astype("category")
        elif opt[c].dtype.kind in "if":
            opt[c] = pd.to_numeric(opt[c], downcast="integer" if opt[c].dtype.kind == "i" else "float")
    ram_opt = opt.memory_usage(deep=True).sum() / 1e6
    return {"archivo": os.path.basename(csv), "motor": "object" if objeto else "pyarrow-str", "filas": len(df), "columnas": df.shape[1],
            "disco_mb": round(disco, 1), "ram_reposo_mb": round(reposo, 1), "ratio_reposo": round(reposo / disco, 2),
            "ram_deep_mb": round(ram, 1), "ratio_deep": round(ram / disco, 2),
            "ram_opt_mb": round(ram_opt, 1), "ratio_opt": round(ram_opt / disco, 2),
            "rss_pico_mb": round(rss_pico_mb(), 1), "ratio_pico": round(rss_pico_mb() / disco, 2), "seg_lectura": round(seg, 1)}

def colapso(d, objeto):
    """Une las tablas del proceso y agrupa, como lo haría un analista con pandas."""
    import pandas as pd
    if objeto:
        pd.set_option("future.infer_string", False)
    pasos, t0 = [], time.time()
    def paso(nombre):
        pasos.append({"paso": nombre, "rss_pico_mb": round(rss_pico_mb(), 1), "seg": round(time.time() - t0, 1)})
    try:
        v = pd.read_csv(os.path.join(d, "vehiculo.csv"), low_memory=False);            paso("leer vehiculo")
        m = pd.read_csv(os.path.join(d, "matricula.csv"), low_memory=False);           paso("leer matricula")
        x = v.merge(m, left_on="id", right_on="vehiculo_id", suffixes=("", "_mat"));   paso("vehiculo + matricula")
        del v, m
        p = pd.read_csv(os.path.join(d, "pagos.csv"), low_memory=False);               paso("leer pagos")
        x = x.merge(p, left_on="id", right_on="vehiculo_id", suffixes=("", "_pago"));  paso("+ pagos")
        del p
        mu = pd.read_csv(os.path.join(d, "multas.csv"), low_memory=False);             paso("leer multas")
        x = x.merge(mu, left_on="id", right_on="vehiculo_id", how="left", suffixes=("", "_multa")); paso("+ multas")
        g = x.groupby(["provincia", "canton", "marca", "clase", "tipo", "estado", "estado_multa"],
                      dropna=False).agg(total=("valor", "sum"), multas=("valor_multa", "sum"), n=("id", "size"))
        paso("groupby")
        return {"resultado": "COMPLETO", "grupos": len(g), "filas_union": len(x), "pasos": pasos}
    except MemoryError:
        paso("MemoryError")
        return {"resultado": "MemoryError", "pasos": pasos}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["medir", "colapso"])
    ap.add_argument("--csv"); ap.add_argument("--sep", default=","); ap.add_argument("--enc", default="utf-8")
    ap.add_argument("--dir"); ap.add_argument("--limite-gb", type=float, default=6.0)
    ap.add_argument("--objeto", action="store_true", help="textos como object (pandas 2.x)")
    a = ap.parse_args()
    limitar(a.limite_gb)
    try:
        r = medir(a.csv, a.sep, a.enc, a.objeto) if a.modo == "medir" else colapso(a.dir, a.objeto)
    except MemoryError:
        r = {"archivo": os.path.basename(a.csv or a.dir), "resultado": "MemoryError", "rss_pico_mb": round(rss_pico_mb(), 1)}
    r["limite_gb"] = a.limite_gb
    print(json.dumps(r, ensure_ascii=False))
