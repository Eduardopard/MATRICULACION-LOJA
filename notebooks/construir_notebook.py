"""Construye notebooks/EDA_Proceso_ANT.ipynb (celdas de texto + código). Luego se ejecuta con nbconvert."""
import nbformat as nbf

C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# EDA – Simulación del proceso de matriculación vehicular (ANT) · Hito 2
**UIDE – Prácticas y Herramientas de Big Data.** Integrantes: Eduardo Pardo, James Romero, Ángel Paladines.

Este cuaderno es **reproducible**: con los dos CSV públicos en `data/raw/` ejecuta
`python src/generar_base_ant.py ...` y luego todas las celdas. Cada tabla y figura del informe sale de aquí
(`resultados/`).""")

code(r'''
import os, json, subprocess, sys, time, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
warnings.filterwarnings("ignore")
RAIZ = os.path.abspath(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
RAW, SIN = f"{RAIZ}/data/raw", f"{RAIZ}/data/sintetico"
TAB, FIG = f"{RAIZ}/resultados/tablas", f"{RAIZ}/resultados/figuras"
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "black",
                     "axes.prop_cycle": plt.cycler(color=["black", "0.45", "0.7"])})
HOY = pd.Timestamp("2026-11-22")
RES = {}                                   # resumen de cifras que usa el informe
def guardar(df, nombre): df.to_csv(f"{TAB}/{nombre}.csv", index=False); return df
print(pd.__version__, np.__version__)
''')

md("## 1. Ficha técnica: volumen en disco y recuento exacto de registros")
code(r'''
def contar_filas(ruta):
    n = 0
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 24), b""):
            n += bloque.count(b"\n")
    return n - 1                                       # sin encabezado
FUENTES = {
  "INEC vehículos matriculados 2019 (real)": (f"{RAW}/inec_vehiculos_matriculados_2019.csv", "https://www.ecuadorencifras.gob.ec/estadisticas-de-transporte-2019/"),
  "SRI vehículos nuevos 2026 (real)":        (f"{RAW}/sri_vehiculos_nuevos_2026.csv", "https://www.sri.gob.ec/datasets"),
}
for t in ["vehiculo", "propietario", "matricula", "revision_vehicular", "pagos", "multas"]:
    FUENTES[f"ANT sintética · {t.upper()}"] = (f"{SIN}/{t}.csv", "generado con src/generar_base_ant.py (semilla 2026)")
filas = []
for nombre, (ruta, url) in FUENTES.items():
    filas.append({"dataset": nombre, "filas": contar_filas(ruta), "columnas": len(pd.read_csv(ruta, nrows=0, sep=None, engine="python").columns),
                  "disco_mb": round(os.path.getsize(ruta) / 1e6, 1), "origen": url})
ficha = guardar(pd.DataFrame(filas), "01_ficha_tecnica")
sint = ficha[ficha.dataset.str.contains("sintética")]
RES["ficha"] = ficha.to_dict("records")
RES["sint_filas"] = int(sint.filas.sum()); RES["sint_mb"] = float(sint.disco_mb.sum())
ficha
''')

md("## 2. Perfilado univariado (taxonomía de Abedjan et al., 2015)")
code(r'''
V = pd.read_parquet(f"{SIN}/vehiculo.parquet")
ATOMICO = {  # tipo atómico (semántico) asignado tras revisar valores y patrones
 "id":"Entero · identificador (clave)", "placa":"Texto · código con patrón", "marca":"Categórico nominal",
 "modelo":"Texto libre (alta cardinalidad)", "anio":"Entero · ordinal temporal", "color":"Categórico nominal",
 "tipo":"Categórico binario", "clase":"Categórico nominal", "subcategoria":"Categórico nominal",
 "cilindraje":"Decimal · continuo (cm³)", "combustible":"Categórico nominal", "servicio":"Categórico nominal",
 "pasajeros":"Entero · discreto", "tonelaje":"Decimal · continuo (t)", "avaluo":"Decimal · moneda (USD)",
 "numero_chasis":"Texto · código 17 car.", "numero_motor":"Texto · código 11 car.", "provincia":"Categórico nominal",
 "canton":"Categórico nominal", "propietario_id":"Entero · clave foránea", "origen":"Categórico binario"}
def perfil(df, atom):
    out = []
    n = len(df)
    for c in df.columns:
        s = df[c]; nn = s.dropna(); vc = nn.value_counts()
        fila = {"columna": c, "tipo_atomico": atom.get(c, ""), "tipo_fisico": str(s.dtype),
                "nulos_%": round(100 * s.isna().mean(), 2), "distintos": int(nn.nunique()),
                "unicidad": round(nn.nunique() / max(len(nn), 1), 4),
                "constancia": round(vc.iloc[0] / max(len(nn), 1), 4) if len(vc) else np.nan,
                "moda": str(vc.index[0])[:22] if len(vc) else ""}
        if s.dtype.kind in "if":
            fila.update({"min": nn.min(), "mediana": nn.median(), "max": nn.max(), "asimetria": round(stats.skew(nn), 2)})
        else:
            L = nn.astype(str).str.len()
            fila.update({"min": L.min(), "mediana": L.median(), "max": L.max(), "asimetria": np.nan})
        out.append(fila)
    return pd.DataFrame(out)
perf_v = guardar(perfil(V, ATOMICO), "02_perfil_vehiculo")
perf_v
''')
code(r'''
# Perfil de la fuente real INEC leída como TEXTO para ver los formatos "crudos"
I = pd.read_csv(f"{RAW}/inec_vehiculos_matriculados_2019.csv", sep=";", encoding="utf-8-sig", dtype=str)
hall = {
 "filas_duplicadas_%": round(100 * I.duplicated().mean(), 1),
 "TONELAJE_con_coma_decimal_sin_cero": int(I.TONELAJE.str.match(r"^,\d").sum()),
 "MODELO_censurado_2003_y_anteriores": int((I.MODELO == "2003 y Años Anteriores").sum()),
 "MARCA_OTRAS_MARCAS": int((I.MARCA == "OTRAS MARCAS").sum()),
 "TONELAJE_max": float(I.TONELAJE.str.replace(",", ".").astype(float).max()),
 "PASAJEROS_max": int(I.PASAJEROS.astype(int).max()),
}
perf_i = guardar(perfil(I, {}), "02_perfil_inec_texto")
RES["inec"] = hall; print(hall); del I
S = pd.read_csv(f"{RAW}/sri_vehiculos_nuevos_2026.csv", dtype=str, keep_default_na=False)
S["cil"] = S.CILINDRAJE.astype(float); S["comb"] = S["TIPO COMBUSTIBLE"]
RES["sri"] = {"COLOR2_vacio_%": round(100 * (S["COLOR 2"] == "").mean(), 2),
              "cil0_no_electricos": int(((S.cil == 0) & (S.comb != "ELECTRICO")).sum()),
              "cil_max": float(S.cil.max()), "filas_duplicadas": int(S.duplicated().sum())}
print(RES["sri"])
''')
code(r'''
# Frecuencias y cardinalidad de las variables categóricas clave
cat = ["clase", "combustible", "servicio", "tipo", "origen", "provincia", "marca", "color"]
card = guardar(pd.DataFrame([{"variable": c, "cardinalidad": V[c].nunique(),
         "top_3": "; ".join(f"{k} ({v:.1%})" for k, v in V[c].value_counts(normalize=True).head(3).items())} for c in cat]),
         "02_cardinalidad")
freq_clase = guardar(V.clase.value_counts().rename_axis("clase").reset_index(name="n").assign(pct=lambda d: (100 * d.n / d.n.sum()).round(1)), "02_frecuencia_clase")
RES["clases"] = freq_clase.head(5).to_dict("records")
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4))
freq_clase.set_index("clase").n.div(1e6).plot.barh(ax=ax[0], color="0.3"); ax[0].invert_yaxis(); ax[0].set_xlabel("millones de vehículos"); ax[0].set_ylabel("")
ax[1].hist(V.anio, bins=np.arange(1984, 2029), color="0.3"); ax[1].set_xlabel("año del modelo"); ax[1].set_ylabel("vehículos")
plt.tight_layout(); plt.savefig(f"{FIG}/f1_univariado.png", dpi=200); card
''')

md("## 3. Auditoría de valores nulos y mecanismo de ausencia (MCAR / MAR / MNAR)")
code(r'''
P = pd.read_parquet(f"{SIN}/propietario.parquet")
M = pd.read_parquet(f"{SIN}/matricula.parquet")
R = pd.read_parquet(f"{SIN}/revision_vehicular.parquet")
tablas = {"vehiculo": V, "propietario": P, "matricula": M, "revision_vehicular": R}
mat = pd.DataFrame({t: df.isna().mean() * 100 for t, df in tablas.items()}).T
mat = mat.loc[:, (mat > 0).any()]
guardar(mat.round(2).reset_index(names="tabla"), "03_matriz_nulos")
fig, ax = plt.subplots(figsize=(7.2, 1.9))
im = ax.imshow(mat.fillna(0).values, cmap="Greys", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns, rotation=40, ha="right", fontsize=7)
ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index, fontsize=7)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat.values[i, j]
        if not np.isnan(v) and v > 0: ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6, color="white" if v > 50 else "black")
plt.colorbar(im, ax=ax, label="% nulos"); plt.tight_layout(); plt.savefig(f"{FIG}/f2_matriz_nulos.png", dpi=200)
mat.round(2)
''')
code(r'''
# Pruebas: ¿la ausencia depende de alguna variable OBSERVADA?
#  - categórica: chi-cuadrado sobre la tabla (nulo/no nulo) x categoría; tamaño de efecto = V de Cramér
#  - numérica: Mann-Whitney entre filas con y sin nulo; tamaño de efecto = AUC
def chi(ind, x):
    t = pd.crosstab(ind, x); c2, p, _, _ = stats.chi2_contingency(t)
    return p, np.sqrt(c2 / (t.values.sum() * (min(t.shape) - 1)))
def mw(ind, x):
    a, b = x[ind].dropna(), x[~ind].dropna(); u, p = stats.mannwhitneyu(a, b)
    return p, u / (len(a) * len(b))
rng = np.random.default_rng(1)
P["edad"] = (HOY - pd.to_datetime(P.fecha_nacimiento)).dt.days / 365.25
nat = P[P.tipo_persona == "NATURAL"].sample(400_000, random_state=1)
own = V.groupby("propietario_id").agg(canton=("canton", "first"), provincia=("provincia", "first"))
capitales = set(V[V.origen == "PARQUE_INEC_2019"].groupby("provincia").canton.agg(lambda s: s.value_counts().index[0]))
nat = nat.join(own, on="id"); nat["rural"] = ~nat.canton.isin(capitales)
r1all = R[R.intento == 1].merge(V[["id", "anio", "servicio", "clase"]], left_on="vehiculo_id", right_on="id")
r1all["edad_vehiculo"] = 2026 - r1all.anio
r1 = r1all.sample(400_000, random_state=1)
S2 = S.assign(c2=(S["COLOR 2"] == ""))
pruebas = []
def add(tabla, col, var, ind, x, kind):
    p, ef = chi(ind, x) if kind == "cat" else mw(ind, x)
    pruebas.append({"tabla": tabla, "columna": col, "contra": var, "p_valor": p, "efecto": round(ef, 3),
                    "medida": "V de Cramér" if kind == "cat" else "AUC"})
add("propietario", "telefono", "edad", nat.telefono.isna(), nat.edad, "num")
add("propietario", "telefono", "rural", nat.telefono.isna(), nat.rural, "cat")
add("propietario", "telefono", "provincia", nat.telefono.isna(), nat.provincia, "cat")
add("propietario", "email", "edad", nat.email.isna(), nat.edad, "num")
add("propietario", "email", "rural", nat.email.isna(), nat.rural, "cat")
add("propietario", "direccion", "edad", nat.direccion.isna(), nat.edad, "num")
add("propietario", "direccion", "rural (vía join)", nat.direccion.isna(), nat.rural, "cat")
add("revision_vehicular", "kilometraje", "edad del vehículo", r1.kilometraje.isna(), r1.edad_vehiculo, "num")
add("revision_vehicular", "kilometraje", "servicio", r1.kilometraje.isna(), r1.servicio, "cat")
add("vehiculo", "modelo", "origen", V.modelo.isna(), V.origen, "cat")
add("SRI real", "COLOR 2", "CLASE", S2.c2, S2.CLASE, "cat")
add("SRI real", "COLOR 2", "PAÍS", S2.c2, S2["PAÍS"], "cat")
pr = guardar(pd.DataFrame(pruebas), "03_pruebas_mecanismo")
pr.assign(p_valor=pr.p_valor.map(lambda p: f"{p:.2e}"))
''')
code(r'''
# Indicio de MNAR en kilometraje (prueba con medición repetida):
# si un vehículo volvió a la RTV, el kilometraje del 2.º intento es un "testigo" de su recorrido real.
# Comparamos ese testigo entre vehículos cuyo 1.er kilometraje FALTÓ y los que sí lo tienen, dentro del mismo grupo de edad.
w = R.pivot_table(index="vehiculo_id", columns="intento", values="kilometraje", aggfunc="first", dropna=False)
w = w[w.index.isin(R[R.intento == 2].vehiculo_id)].join(V.set_index("id")[["anio", "clase"]])
w["edad"] = 2026 - w.anio
w = w[w[2].notna() & (w.edad >= 3)]
w["falto_1"] = w[1].isna()
w["km_anual_testigo"] = w[2] / w.edad
w["grupo_edad"] = pd.cut(w.edad, [2, 10, 20, 50])
q = w.groupby(["grupo_edad", "falto_1"], observed=True).km_anual_testigo.median().unstack().round(0)
q.columns = ["mediana_km_anual_si_hubo_dato", "mediana_km_anual_si_falto"]
pm = stats.mannwhitneyu(w.km_anual_testigo[w.falto_1], w.km_anual_testigo[~w.falto_1], alternative="greater")
RES["mnar"] = {"n_vehiculos": int(len(w)), "n_falto": int(w.falto_1.sum()), "p_valor": float(pm.pvalue),
               "auc": round(pm.statistic / (w.falto_1.sum() * (~w.falto_1).sum()), 3),
               "med_falto": float(w.km_anual_testigo[w.falto_1].median()), "med_dato": float(w.km_anual_testigo[~w.falto_1].median())}
guardar(q.reset_index().astype(str), "03_indicio_mnar_km")
print(RES["mnar"])
km = r1.kilometraje
imp = km.fillna(km.mean())
RES["km"] = {"nulos_%": round(100 * km.isna().mean(), 2), "media_obs": round(km.mean()), "sd_obs": round(km.std()),
             "sd_imputada_media": round(imp.std()), "reduccion_var_%": round(100 * (1 - imp.var() / km.var()), 1)}
print(RES["km"]); q
''')
code(r'''
# Clasificación final y estrategia (sin imputaciones ciegas)
clasif = pd.DataFrame([
 ["propietario.telefono", "3,0", "MCAR", "Efecto nulo: AUC con edad = 0,50 y V de Cramér ≤ 0,007 con zona y provincia.", "No imputar: es un dato de contacto; se deja vacío."],
 ["propietario.email", "17,3", "MAR", "Depende de la edad (AUC = 0,90): falta más en mayores.", "No imputar; si un modelo lo usa, crear indicador 'sin_email' y condicionar por edad."],
 ["propietario.direccion", "3,0", "MAR", "Depende de si el cantón es rural (V = 0,15; dato que existe vía join).", "Imputación condicional: completar con el cantón del vehículo (nivel cantón, no calle)."],
 ["revision.kilometraje", "≈8", "MNAR", "Prueba con medición repetida: quienes no tienen el 1.er dato recorren más km por año (hasta 3 veces más; AUC 0,79).", "No usar la media (sesgo hacia abajo). Indicador 'km_faltante' + mediana por clase y edad solo para describir; análisis de sensibilidad."],
 ["vehiculo.modelo / subcategoria", "81,9", "Estructural (MAR por origen)", "Falta en todo el parque INEC: la fuente no trae esa columna.", "No imputar; analizar el modelo solo en vehículos SRI."],
 ["vehiculo.pasajeros / tonelaje", "18,1", "Estructural (MAR por origen)", "Falta en todo vehículo SRI.", "Imputar con mediana por clase (dependencia clase → tonelaje)."],
 ["revision.observaciones", "≈90", "No aplica", "Vacío cuando el vehículo aprueba sin defectos.", "Reemplazar por 'SIN OBSERVACIONES'."],
 ["matricula.fecha_matriculacion", "11,5", "No aplica", "Vacío cuando el estado no es VIGENTE.", "Mantener nulo; se valida con la regla estado ↔ fecha."],
 ["SRI COLOR 2 (real)", "12,0", "MAR", "Depende de la clase y el país de origen.", "Imputar con COLOR 1: cuando hay dato, coinciden en el 98,8 %."],
 ["INEC 'OTRAS MARCAS' / SRI cilindraje = 0 (no eléctricos)", "0,4 / <0,01", "Nulos disfrazados", "Valores comodín que esconden un dato desconocido.", "Convertir a nulo y, en cilindraje, imputar con la mediana del mismo modelo."],
], columns=["columna", "nulos_%", "mecanismo", "evidencia", "estrategia"])
guardar(clasif, "03_clasificacion_nulos"); clasif
''')

md("## 4. Relaciones multivariadas, dependencias funcionales y outliers")
code(r'''
PG = pd.read_parquet(f"{SIN}/pagos.parquet", columns=["vehiculo_id", "tipo", "valor", "fecha"],
                     filters=[("tipo", "==", "IMPUESTO_PROPIEDAD_SRI")])
MU = pd.read_parquet(f"{SIN}/multas.parquet", columns=["vehiculo_id", "tipo", "valor", "fecha", "estado"])
impp = PG[PG.tipo == "IMPUESTO_PROPIEDAD_SRI"].groupby("vehiculo_id").valor.sum().rename("impuesto")
mul = MU.groupby("vehiculo_id").valor.agg(n_multas="size", valor_multas="sum")
A = (V[["id", "anio", "cilindraje", "avaluo", "pasajeros", "tonelaje"]].set_index("id")
       .join(r1all.set_index("vehiculo_id").kilometraje).join(impp).join(mul))
A["edad"] = 2026 - A.pop("anio")
A[["n_multas", "valor_multas"]] = A[["n_multas", "valor_multas"]].fillna(0)
num = ["edad", "cilindraje", "avaluo", "impuesto", "kilometraje", "pasajeros", "tonelaje", "n_multas"]
B = A[num].dropna(); B = B.sample(min(600_000, len(B)), random_state=1)
pear, spear = B.corr("pearson"), B.corr("spearman")
guardar(pear.round(3).reset_index(names="variable"), "04_pearson"); guardar(spear.round(3).reset_index(names="variable"), "04_spearman")
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
for a, m, t in [(ax[0], pear, "Pearson"), (ax[1], spear, "Spearman")]:
    a.imshow(m.values, cmap="Greys", vmin=-1, vmax=1)
    a.set_xticks(range(len(num))); a.set_xticklabels(num, rotation=60, ha="right", fontsize=6.5)
    a.set_yticks(range(len(num))); a.set_yticklabels(num if a is ax[0] else [], fontsize=6.5); a.set_title(t, fontsize=9)
    for i in range(len(num)):
        for j in range(len(num)):
            a.text(j, i, f"{m.values[i,j]:.2f}", ha="center", va="center", fontsize=5.2, color="white" if abs(m.values[i,j]) > .55 else "black")
plt.tight_layout(); plt.savefig(f"{FIG}/f3_correlaciones.png", dpi=200)
RES["corr"] = {"pear_av_imp": round(pear.loc["avaluo", "impuesto"], 3), "spear_av_imp": round(spear.loc["avaluo", "impuesto"], 3),
               "pear_edad_km": round(pear.loc["edad", "kilometraje"], 3), "spear_edad_km": round(spear.loc["edad", "kilometraje"], 3),
               "pear_edad_av": round(pear.loc["edad", "avaluo"], 3), "spear_edad_av": round(spear.loc["edad", "avaluo"], 3),
               "pear_pas_ton": round(pear.loc["pasajeros", "tonelaje"], 3), "spear_pas_ton": round(spear.loc["pasajeros", "tonelaje"], 3)}
RES["corr"]
''')
code(r'''
# Dependencias funcionales X -> Y: se cumple si cada valor de X tiene UN solo valor de Y
def fd(df, X, Y):
    g = df.groupby(X, observed=True)[Y].nunique()
    return int((g > 1).sum()), int(len(g))
S["AV"] = S["AVALÚO"]
dep = []
for nombre, df, X, Y in [
    ("vehiculo", V, "placa", "id"), ("vehiculo", V, "numero_chasis", "id"),
    ("vehiculo", V, "canton", "provincia"), ("vehiculo", V, "modelo", "marca"),
    ("SRI real", S, "CATEGORÍA", "MODELO"), ("SRI real", S, "CATEGORÍA", "AV"), ("SRI real", S, "MODELO", "MARCA"),
    ("vehiculo", V, "clase", "tipo")]:
    v_, n_ = fd(df, X, Y); dep.append({"tabla": nombre, "dependencia": f"{X} → {Y}", "grupos": n_, "violaciones": v_,
                                       "cumple": "sí" if v_ == 0 else ("casi" if v_ / n_ < .01 else "no")})
# avalúo -> impuesto (regla del SRI): se recalcula la tabla y se compara
def imp_sri(av):
    out = np.zeros_like(av)
    for lim, base, tasa in [(0,0,.005),(4000,20,.01),(8000,60,.02),(12000,140,.03),(16000,260,.04),(20000,420,.05),(24000,620,.06)]:
        m = av > lim; out[m] = base + (av[m] - lim) * tasa
    return np.round(out, 2)
chk = A[["avaluo", "impuesto"]].dropna().join(V.set_index("id").servicio)
chk = chk[chk.servicio != "ESTADO"]
dif = np.abs(imp_sri(chk.avaluo.values) - chk.impuesto.values) > .01
dep.append({"tabla": "vehiculo+pagos", "dependencia": "avaluo → impuesto (tabla SRI)", "grupos": len(chk), "violaciones": int(dif.sum()), "cumple": "sí" if dif.sum() == 0 else "no"})
dep = guardar(pd.DataFrame(dep), "04_dependencias_funcionales"); dep
''')
code(r'''
# Outliers univariados: IQR (Tukey, k = 1,5) y Z-score (|z| > 3)
out = []
for c in ["avaluo", "cilindraje", "kilometraje", "tonelaje", "pasajeros", "impuesto"]:
    x = A[c].dropna(); q1, q3 = x.quantile([.25, .75]); iqr = q3 - q1
    z = (x - x.mean()) / x.std()
    out.append({"variable": c, "n": len(x), "lim_sup_IQR": round(q3 + 1.5 * iqr, 1), "outliers_IQR": int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()),
                "outliers_Z": int((z.abs() > 3).sum()), "max": round(x.max(), 1)})
out = guardar(pd.DataFrame(out).assign(pct_IQR=lambda d: (100 * d.outliers_IQR / d.n).round(2), pct_Z=lambda d: (100 * d.outliers_Z / d.n).round(2)), "04_outliers_univariados")
out
''')
code(r'''
# Anomalías MULTIVARIADAS: el valor solo es raro en combinación con otras variables.
# 1) IQR condicional: kilometraje por año de uso, calculado DENTRO de cada clase.
A2 = A.join(V.set_index("id")[["clase", "servicio", "combustible", "origen"]])
A2["km_anual"] = A2.kilometraje / A2.edad.clip(lower=1)
def iqr_grupo(s):
    q1, q3 = s.quantile([.25, .75]); return (s > q3 + 3 * (q3 - q1))
A2["anom_km"] = A2.groupby("clase").km_anual.transform(iqr_grupo).fillna(False).astype(bool)
# 2) Z robusto del avalúo dentro de (clase, edad): |z| > 4 con mediana y MAD
g = A2.groupby(["clase", "edad"]).avaluo
med, mad = g.transform("median"), g.transform(lambda s: (s - s.median()).abs().median())
A2["z_rob_av"] = 0.6745 * (A2.avaluo - med) / mad.replace(0, np.nan)
A2["anom_av"] = A2.z_rob_av.abs() > 4
# 3) Reglas de negocio cruzadas
A2["moto_cil_alto"] = (A2.clase == "MOTOCICLETA") & (A2.cilindraje > 2000)
A2["nuevo_km_alto"] = (A2.origen == "NUEVO_SRI_2026") & (A2.kilometraje > 5000)
fm = M.set_index("vehiculo_id").fecha_matriculacion
p48 = PG[PG.tipo == "IMPUESTO_PROPIEDAD_SRI"].join(fm, on="vehiculo_id").dropna(subset=["fecha_matriculacion"])
viol48 = int(((pd.to_datetime(p48.fecha_matriculacion) - pd.to_datetime(p48.fecha)).dt.days < 2).sum())
mm = MU[(MU.tipo == "CONTRAVENCION") & (MU.estado == "PENDIENTE")].join(fm, on="vehiculo_id").dropna(subset=["fecha_matriculacion"])
fmm = pd.to_datetime(mm.fecha_matriculacion)
viol_multa = int(((fmm >= "2026-10-01") & (pd.to_datetime(mm.fecha) < fmm)).sum())
anom = guardar(pd.DataFrame([
  ["km por año extremo dentro de su clase (IQR k=3)", int(A2.anom_km.sum())],
  ["avalúo atípico para su clase y edad (Z robusto > 4)", int(A2.anom_av.sum())],
  ["motocicleta con cilindraje > 2.000 cm³", int(A2.moto_cil_alto.sum())],
  ["impuesto pagado < 48 h antes de matricular (regla Municipio)", viol48],
  ["matriculado desde oct-2026 con multa pendiente previa", viol_multa]],
  columns=["anomalia_multivariada", "casos"]), "04_anomalias_multivariadas")
RES["anom"] = anom.to_dict("records"); anom
''')

code(r'''
json.dump(RES, open(f"{TAB}/resumen_eda.json", "w"), ensure_ascii=False, indent=1, default=str)
print("resumen guardado")
''')
nb = nbf.v4.new_notebook(); nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/01_EDA_Proceso_ANT.ipynb")
C = []
md("""# The Memory Wall y dictamen hacia Spark · Hito 2
Este cuaderno corre aparte del EDA para empezar con la memoria vacía: cada prueba se ejecuta en un
proceso hijo con un límite de memoria de 6 GB (simula una laptop de 8 GB con el sistema operativo abierto).""")
code(r'''
import os, json, subprocess, sys, time, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
warnings.filterwarnings("ignore")
RAIZ = os.path.abspath(os.path.join(os.getcwd(), "..")) if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
RAW, SIN = f"{RAIZ}/data/raw", f"{RAIZ}/data/sintetico"
TAB, FIG = f"{RAIZ}/resultados/tablas", f"{RAIZ}/resultados/figuras"
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "black",
                     "axes.prop_cycle": plt.cycler(color=["black", "0.45", "0.7"])})
HOY = pd.Timestamp("2026-11-22")
RES = {}                                   # resumen de cifras que usa el informe
def guardar(df, nombre): df.to_csv(f"{TAB}/{nombre}.csv", index=False); return df
print(pd.__version__, np.__version__)
''')
md("## 5. The Memory Wall: RAM en reposo vs. disco y colapso de pandas")
code(r'''
import gc
for _n in ["I", "S", "S2", "P", "M", "R", "PG", "MU", "A", "A2", "B", "V", "r1", "r1all", "nat", "own", "w", "chk", "p48", "mm", "impp", "mul"]:
    globals().pop(_n, None)
gc.collect(); print("memoria liberada para las pruebas")
''')
code(r'''
def correr(args):
    r = subprocess.run([sys.executable, f"{RAIZ}/src/memory_wall.py"] + args, capture_output=True, text=True, cwd=RAIZ)
    return json.loads(r.stdout.strip().splitlines()[-1])
med = []
for t in ["vehiculo", "propietario", "matricula", "revision_vehicular", "pagos", "multas"]:
    for obj in (True, False):
        med.append(correr(["medir", "--csv", f"{SIN}/{t}.csv", "--limite-gb", "6"] + (["--objeto"] if obj else [])))
med.append(correr(["medir", "--csv", f"{RAW}/inec_vehiculos_matriculados_2019.csv", "--sep", ";", "--enc", "utf-8-sig", "--objeto"]))
mw = guardar(pd.DataFrame(med), "05_memory_wall_tablas")
mw[["archivo", "motor", "filas", "disco_mb", "ram_deep_mb", "ratio_deep", "rss_pico_mb", "ratio_pico", "ram_opt_mb"]]
''')
code(r'''
ob = mw[(mw.motor == "object") & (mw.archivo != "inec_vehiculos_matriculados_2019.csv")]
ar = mw[(mw.motor == "pyarrow-str")]
RES["mw"] = {"disco_total_mb": round(ob.disco_mb.sum(), 1),
             "ram_obj_total_mb": round(ob.ram_deep_mb.sum(), 1), "ratio_obj": round(ob.ram_deep_mb.sum() / ob.disco_mb.sum(), 2),
             "ram_arrow_total_mb": round(ar.ram_deep_mb.sum(), 1), "ratio_arrow": round(ar.ram_deep_mb.sum() / ar.disco_mb.sum(), 2),
             "pico_max_ratio": round(ob.ratio_pico.max(), 2), "ram_opt_total_mb": round(ob.ram_opt_mb.sum(), 1)}
col = correr(["colapso", "--dir", SIN, "--limite-gb", "6"])
RES["colapso"] = col
pd.DataFrame(col["pasos"])
''')
code(r'''
# Spark local con SOLO 2 GB para el driver: el mismo join + groupby
import psutil, threading
pico = [0]
def vigilar(p):
    while p.poll() is None:
        try:
            pr = psutil.Process(p.pid); rss = pr.memory_info().rss + sum(c.memory_info().rss for c in pr.children(recursive=True))
            pico[0] = max(pico[0], rss)
        except Exception: pass
        time.sleep(.5)
t = time.time()
proc = subprocess.Popen([sys.executable, f"{RAIZ}/spark/fase1_spark_proceso_ant.py", "--base", SIN, "--memoria", "2g"],
                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, cwd=RAIZ)
th = threading.Thread(target=vigilar, args=(proc,)); th.start(); salida = proc.communicate()[0]; th.join()
sp = json.loads([l for l in salida.splitlines() if l.startswith("{")][-1]); sp["rss_pico_mb"] = round(pico[0] / 1e6)
RES["spark"] = sp; sp
''')
code(r'''
# Extrapolación (regla de tres con el ratio medido) para el escenario nacional real
b_por_veh = ob.disco_mb.sum() * 1e6 / 2_821_910                    # bytes en disco por vehículo (6 tablas)
esc = []
for anios in (1, 5, 10):
    disco = 3_310_000 * anios * b_por_veh / 1e9                     # parque 2025 = 3,31 millones (INEC)
    esc.append({"escenario": f"parque nacional, {anios} año(s) de historia", "disco_gb": round(disco, 1),
                "ram_pandas_gb": round(disco * RES["mw"]["ratio_obj"], 1), "ram_pico_join_gb": round(disco * RES["mw"]["pico_max_ratio"], 1),
                "particiones_spark_128MB": int(np.ceil(disco * 1e3 / 128))})
esc = guardar(pd.DataFrame(esc), "05_extrapolacion"); RES["esc"] = esc.to_dict("records")
json.dump(RES, open(f"{TAB}/resumen_memory_wall.json", "w"), ensure_ascii=False, indent=1, default=str)
esc
''')


nb = nbf.v4.new_notebook(); nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/02_Memory_Wall_Spark.ipynb")
print("ok", len(C), "celdas")
