"""
Generador de la base SINTÉTICA del proceso de matriculación vehicular (ANT) - Ecuador 2026.

Construye 6 tablas relacionales (VEHICULO, PROPIETARIO, MATRICULA, REVISION_VEHICULAR,
PAGOS, MULTAS) a partir de dos fuentes REALES y públicas:

  * INEC - Anuario de Estadísticas de Transporte, vehículos matriculados 2019 (2.311.960 filas)
  * SRI  - Vehículos nuevos 2026 (509.950 filas)

Cada vehículo sintético hereda las características reales de una fila de esas fuentes
(provincia, marca, clase, combustible, año, avalúo, cantón...). Los datos personales
(cédulas, nombres, placas, chasis, teléfonos, correos) son FICTICIOS: se generan al azar
y no corresponden a ninguna persona ni vehículo real.

Reglas del proceso tomadas de fuentes oficiales (ver README):
  * Calendario ANT por último dígito de placa: 1=feb ... 9=oct, 0=nov; diciembre = rezagados.
  * Cierre de la ANT (caso Jaque) del 30-ene-2026; vehículos nuevos desde el 12/13-feb y
    resto de trámites desde el 9-mar-2026. Dígito 2 extendido a abril, dígito 1 a junio.
  * Pago de matrícula al menos 48 h antes del trámite (Municipio de Loja).
  * Tabla del impuesto a la propiedad (SRI) y depreciación del avalúo 20 %/año, piso 10 %.
  * Revisión técnica: 2,5 % de condicionados (referencia AMT Quito 2024).
  * Tasa RTV Loja: USD 5,10 (< 3,5 t) y 10,10 (> 3,5 t). Multas: SBU 2026 = USD 482.

Uso:  python src/generar_base_ant.py --inec RUTA --sri RUTA --salida data/sintetico [--semilla 2026]
"""
import argparse, os, time, unicodedata
import numpy as np
import pandas as pd
import pyarrow as pa, pyarrow.csv as pacsv, pyarrow.parquet as pq

HOY = np.datetime64("2026-11-22")
SBU = 482.0

def norm(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s).upper().strip())
                   if unicodedata.category(c) != "Mn")

def guardar(df, nombre, salida):
    t = pa.Table.from_pandas(df, preserve_index=False)
    pacsv.write_csv(t, os.path.join(salida, f"{nombre}.csv"))
    pq.write_table(t, os.path.join(salida, f"{nombre}.parquet"), compression="zstd")
    print(f"  {nombre}: {len(df):,} filas")

def cadenas(rng, n, largo, alfabeto):
    a = np.array(list(alfabeto), dtype="<U1")
    m = a[rng.integers(0, len(a), size=(n, largo))]
    return np.ascontiguousarray(m).view(f"<U{largo}").ravel()

# ---------------------------------------------------------------- catálogos
LETRA_PROV = {"AZUAY":"A","BOLIVAR":"B","CANAR":"U","CARCHI":"C","COTOPAXI":"X","CHIMBORAZO":"H",
 "EL ORO":"O","ESMERALDAS":"E","GALAPAGOS":"W","GUAYAS":"G","IMBABURA":"I","LOJA":"L","LOS RIOS":"R",
 "MANABI":"M","MORONA SANTIAGO":"V","NAPO":"N","PASTAZA":"S","PICHINCHA":"P","ORELLANA":"Q",
 "SUCUMBIOS":"K","TUNGURAHUA":"T","ZAMORA CHINCHIPE":"Z","SANTA ELENA":"Y","SANTO DOMINGO DE LOS TSACHILAS":"J"}
COD_PROV = {k: f"{i+1:02d}" for i, k in enumerate(["AZUAY","BOLIVAR","CANAR","CARCHI","COTOPAXI","CHIMBORAZO",
 "EL ORO","ESMERALDAS","GUAYAS","IMBABURA","LOJA","LOS RIOS","MANABI","MORONA SANTIAGO","NAPO","PASTAZA",
 "PICHINCHA","TUNGURAHUA","ZAMORA CHINCHIPE","GALAPAGOS","SUCUMBIOS","ORELLANA","SANTO DOMINGO DE LOS TSACHILAS","SANTA ELENA"])}
CAPITAL = {"AZUAY":"CUENCA","BOLIVAR":"GUARANDA","CANAR":"AZOGUES","CARCHI":"TULCAN","COTOPAXI":"LATACUNGA",
 "CHIMBORAZO":"RIOBAMBA","EL ORO":"MACHALA","ESMERALDAS":"ESMERALDAS","GALAPAGOS":"SAN CRISTOBAL",
 "GUAYAS":"GUAYAQUIL","IMBABURA":"IBARRA","LOJA":"LOJA","LOS RIOS":"BABAHOYO","MANABI":"PORTOVIEJO",
 "MORONA SANTIAGO":"MORONA","NAPO":"TENA","PASTAZA":"PASTAZA","PICHINCHA":"QUITO","ORELLANA":"FRANCISCO DE ORELLANA",
 "SUCUMBIOS":"LAGO AGRIO","TUNGURAHUA":"AMBATO","ZAMORA CHINCHIPE":"ZAMORA","SANTA ELENA":"SANTA ELENA",
 "SANTO DOMINGO DE LOS TSACHILAS":"SANTO DOMINGO"}
ALIAS_CANTON = {"PUEBLO VIEJO":"PUEBLOVIEJO","ORELLANA":"FRANCISCO DE ORELLANA","SAN PEDRO DE PELILEO":"PELILEO",
 "LIMON - INDANZA":"LIMON INDANZA","24 DE MAYO":"VEINTICUATRO DE MAYO","RIO VERDE":"RIOVERDE","SANTIAGO DE PILLARO":"PILLARO"}
CLASE_INEC = {"AUTOMOVIL":"AUTOMOVIL","MOTOCICLETA":"MOTOCICLETA","CAMIONETA":"CAMIONETA","SUV":"SUV","CAMION":"CAMION",
 "FURGONETA":"FURGONETA","AUTOBUS":"BUS","TRAILER":"TRAILER","VOLQUETA":"VOLQUETA","OTRA CLASE":"OTRA CLASE","TANQUERO":"TANQUERO"}
CLASE_SRI = {"MOTOCICLETA":"MOTOCICLETA","JEEP":"SUV","CAMIONETA":"CAMIONETA","AUTOMOVIL":"AUTOMOVIL","CAMION":"CAMION",
 "TRAILER":"TRAILER","OMNIBUS":"BUS","VOLQUETA":"VOLQUETA","TANQUERO":"TANQUERO","ESPECIAL":"OTRA CLASE"}
COMB_INEC = {"GASOLINA":"GASOLINA","DIESEL":"DIESEL","HIBRIDO":"HIBRIDO","ELECTRICO":"ELECTRICO","GAS LICUADO DE PETROLEO":"GLP"}
COMB_SRI = {"GASOLINA":"GASOLINA","DIESEL":"DIESEL","HIBRIDO_GASOLINA_BATERIAS":"HIBRIDO","HIBRIDO_DIESEL_BATERIAS":"HIBRIDO",
 "ELECTRICO":"ELECTRICO","DUAL_GAS_GASOLINA":"GLP"}
SERV_INEC = {"PARTICULAR":"PARTICULAR","USO PUBLICO, ALQUILER O COMERCIAL":"COMERCIAL","ESTADO":"ESTADO","OTROS":"OTROS"}
COLORES = {"NEG":"NEGRO","PLO":"PLOMO","BLA":"BLANCO","ROJ":"ROJO","PLA":"PLATEADO","AZU":"AZUL","VRD":"VERDE",
 "AMA":"AMARILLO","CRE":"CREMA","ANA":"ANARANJADO","DOR":"DORADO","CAF":"CAFE","VIN":"VINO","CEL":"CELESTE",
 "MOR":"MORADO","ROS":"ROSADO","COB":"COBRE","GRY":"GRIS","CAM":"CAMUFLAJE","CUA":"CUARZO"}
NOMBRES_H = ("JOSE LUIS CARLOS JUAN MIGUEL DIEGO ANDRES JORGE PABLO DAVID LUIS FERNANDO MARCO EDISON SANTIAGO "
             "CRISTIAN DANIEL FRANKLIN SEGUNDO MANUEL VICTOR HUGO EDUARDO GABRIEL JAMES ANGEL RAMIRO GALO "
             "PATRICIO WILSON BYRON KLEBER STALIN RICARDO JAVIER MAURICIO").split()
NOMBRES_M = ("MARIA ANA ROSA CARMEN GLORIA DIANA PAOLA ANDREA GABRIELA KATHERINE JESSICA VERONICA LORENA "
             "SILVIA PATRICIA MONICA CRISTINA DANIELA FERNANDA ESTEFANIA JOHANNA MARTHA BLANCA ELENA "
             "SOFIA VALERIA XIMENA KARLA NANCY LUCIA ISABEL TERESA").split()
APELLIDOS = ("JARAMILLO OCHOA CUEVA VALDIVIESO CASTILLO ARMIJOS TORRES CARRION LUDENA GONZALEZ RODRIGUEZ "
             "SANCHEZ ROMERO PALADINES PARDO DAVILA MOROCHO GUAMAN QUIZHPE CHAMBA SARANGO LOJAN MALDONADO "
             "ESPINOZA GUERRERO VELEZ ZAMBRANO MOREIRA MACIAS VERA MENDOZA CEDENO PINARGOTE BRAVO ORTIZ "
             "CABRERA VACA HERRERA MORALES SALAZAR PAREDES SUAREZ RAMIREZ FLORES CASTRO RUIZ VEGA REYES "
             "LEON AGUIRRE MERINO ABAD ASTUDILLO CALLE PESANTEZ RIOFRIO GRANDA SOTOMAYOR VALAREZO ENCALADA "
             "TAPIA ROJAS CORDOVA SALINAS NARVAEZ BENITEZ CHALCO JIMENEZ AVILA OJEDA YAGUACHE ROMAN").split()
CALLES = ["BOLIVAR","SUCRE","OLMEDO","10 DE AGOSTO","18 DE NOVIEMBRE","UNIVERSITARIA","MERCADILLO","ROCAFUERTE",
          "AZUAY","IMBABURA","COLON","JOSE ANTONIO EGUIGUREN","LOURDES","CATACOCHA","MIGUEL RIOFRIO",
          "EMILIANO ORTEGA","MANUEL AGUSTIN AGUIRRE","ORILLAS DEL ZAMORA","CUXIBAMBA","PIO JARAMILLO ALVARADO",
          "AMAZONAS","9 DE OCTUBRE","GARCIA MORENO","24 DE MAYO","JUAN DE SALINAS","CHILE","VENEZUELA"]
DEFECTOS = np.array(["EMISION DE GASES FUERA DE NORMA","FRENOS CON EFICIENCIA INSUFICIENTE","LUCES DEFECTUOSAS",
                     "LLANTAS LISAS","ALINEACION FUERA DE TOLERANCIA","RUIDO SOBRE EL LIMITE","CINTURONES EN MAL ESTADO"])
PESO_DEF = np.array([.28,.22,.18,.14,.10,.05,.03])
CONTRAV = [  # tipo, descripcion, % SBU, peso
    ("CONTRAVENCION", "NO USAR CINTURON DE SEGURIDAD", .05, .22),
    ("CONTRAVENCION", "USO DE CELULAR AL CONDUCIR", .10, .14),
    ("CONTRAVENCION", "ESTACIONAR EN SITIO PROHIBIDO", .10, .24),
    ("CONTRAVENCION", "EXCESO DE VELOCIDAD DENTRO DEL RANGO MODERADO", .15, .26),
    ("CONTRAVENCION", "EXCESO DE VELOCIDAD FUERA DEL RANGO MODERADO", .30, .10),
    ("CONTRAVENCION", "CIRCULAR SIN PLACAS", 1.00, .04),
]

def impuesto_propiedad(av):
    tramos = [(0,0,.005),(4000,20,.01),(8000,60,.02),(12000,140,.03),(16000,260,.04),(20000,420,.05),(24000,620,.06)]
    out = np.zeros_like(av)
    for lim, base, tasa in tramos:
        m = av > lim
        out[m] = base + (av[m] - lim) * tasa
    return np.round(out, 2)

# ---------------------------------------------------------------- main
def main(inec_path, sri_path, salida, semilla):
    t0 = time.time()
    rng = np.random.default_rng(semilla)
    os.makedirs(salida, exist_ok=True)
    ref = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "ref", "cantones_provincias.csv"), sep=";")
    ref["k"] = ref.CANTON.map(norm); ref["p"] = ref.PROVINCIA.map(norm)

    # ------------------ fuentes reales
    inec = pd.read_csv(inec_path, sep=";", encoding="utf-8-sig", dtype=str)
    sri = pd.read_csv(sri_path, dtype=str, keep_default_na=False)
    print(f"fuentes leídas {time.time()-t0:.0f}s")

    # ------------------ VEHICULO  (A = parque INEC, B = nuevos SRI 2026)
    na, nb = len(inec), len(sri)
    A = pd.DataFrame({
        "provincia": inec.PROVINCIA.map(norm).values,
        "marca": inec.MARCA.str.strip().values,
        "clase": inec.CLASE.map(norm).map(CLASE_INEC).values,
        "combustible": inec.COMBUSTIBLE.map(norm).map(COMB_INEC).values,
        "servicio": inec.SERVICIO.map(norm).map(SERV_INEC).values,
        "pasajeros": inec.PASAJEROS.astype("int16").values,
        "tonelaje": inec.TONELAJE.str.replace(",", ".").astype("float32").values,
    })
    anio = pd.to_numeric(inec.MODELO, errors="coerce").to_numpy(copy=True)
    cens = np.isnan(anio)                      # "2003 y Años Anteriores" (dato censurado)
    w = np.linspace(1, 6, 19); w /= w.sum()
    anio[cens] = rng.choice(np.arange(1985, 2004), size=cens.sum(), p=w)
    A["anio"] = anio.astype("int16")
    A["origen"] = "PARQUE_INEC_2019"
    del inec

    cant = sri["CANTÓN"].map(norm).replace(ALIAS_CANTON)
    mapa = ref.drop_duplicates("k").set_index("k")["p"]
    prov_b = cant.map(mapa).to_numpy(copy=True)
    amb = cant.isin(["BOLIVAR", "OLMEDO"]).values                       # nombres repetidos en 2 provincias
    prov_b[amb & (cant.values == "BOLIVAR")] = np.where(rng.random((amb & (cant.values=="BOLIVAR")).sum()) < .5, "CARCHI", "MANABI")
    prov_b[amb & (cant.values == "OLMEDO")] = np.where(rng.random((amb & (cant.values=="OLMEDO")).sum()) < .5, "LOJA", "MANABI")
    Bv = pd.DataFrame({
        "provincia": prov_b, "canton": cant.values,
        "marca": sri.MARCA.str.strip().values, "modelo": sri.MODELO.str.strip().values,
        "anio": sri["AÑO MODELO"].astype("int16").values,
        "clase": sri.CLASE.map(norm).map(CLASE_SRI).values,
        "subcategoria": sri["SUB CLASE"].values, "tipo": sri.TIPO.values,
        "cilindraje": sri.CILINDRAJE.astype("float32").values,
        "combustible": sri["TIPO COMBUSTIBLE"].map(COMB_SRI).values,
        "color": sri["COLOR 1"].map(COLORES).values,
        "avaluo": sri["AVALÚO"].str.replace(".", "", regex=False).str.replace(",", ".").astype("float64").values,
        "servicio": np.where(sri["TIPO SERVICIO"].values == "ALQ", "COMERCIAL", "PARTICULAR"),
        "persona_sri": sri["PERSONA NATURAL - JURÍDICA"].map(norm).values,
        "fecha_compra": pd.to_datetime(sri["FECHA COMPRA (DD/MM/AAAA)"], format="%d/%m/%Y").values.astype("datetime64[D]"),
    })
    Bv["origen"] = "NUEVO_SRI_2026"
    del sri

    # características que la fuente INEC no trae: se toman de la distribución real del SRI por clase
    A["canton"] = A.provincia.map(CAPITAL).values
    loja = A.provincia.values == "LOJA"                     # en Loja se reparte por los 16 cantones
    cl = ref[ref.p == "LOJA"].k.values
    pesos = np.array([10 if c == "LOJA" else 1 for c in cl], float); pesos /= pesos.sum()
    A.loc[loja, "canton"] = rng.choice(cl, size=loja.sum(), p=pesos)
    for c in A.clase.unique():
        m = A.clase.values == c
        src = Bv[Bv.clase == c] if (Bv.clase == c).sum() > 50 else Bv
        idx = rng.integers(0, len(src), m.sum())
        A.loc[m, "color"] = src.color.values[idx]
        cil = src.cilindraje.values[idx].copy()
        A.loc[m, "cilindraje"] = cil
        # avalúo de referencia del modelo nuevo de la misma marca y clase (mediana SRI) depreciado 20 %/año, piso 10 %
    med_mc = Bv.groupby(["marca", "clase"]).avaluo.median()
    med_c = Bv.groupby("clase").avaluo.median()
    base = pd.Series(list(zip(A.marca, A.clase))).map(med_mc).values
    base = np.where(np.isnan(base), A.clase.map(med_c).values, base)
    base = np.where(np.isnan(base), Bv.loc[Bv.clase.isin(["CAMIONETA", "CAMION"]), "avaluo"].median(), base)  # FURGONETA no existe en SRI
    base = base * rng.lognormal(0, .25, na)
    edad = (2026 - A.anio.values).clip(0)
    A["avaluo"] = np.round(np.maximum(base * (0.8 ** edad), base * 0.10), 2)
    A["tipo"] = np.where(A.tonelaje.values > 3.5, "PESADO", "LIVIANO")
    A.loc[A.combustible.values == "ELECTRICO", "cilindraje"] = 0

    V = pd.concat([A, Bv], ignore_index=True)
    del A, Bv
    n = len(V)
    V.insert(0, "id", np.arange(1, n + 1, dtype="int32"))
    print(f"vehiculos {n:,}  {time.time()-t0:.0f}s")

    # placas únicas por provincia (autos: L+2 letras+4 dígitos; motos: L+letra+3 dígitos+letra)
    letras = np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    placa = np.empty(n, dtype="<U7")
    moto = V.clase.values == "MOTOCICLETA"
    for p, L in LETRA_PROV.items():
        for es_moto in (False, True):
            m = (V.provincia.values == p) & (moto == es_moto)
            k = m.sum()
            if k == 0: continue
            if es_moto:
                cod = rng.choice(26 * 1000 * 26, size=k, replace=False)
                a, r = divmod(cod, 1000 * 26); d, b = divmod(r, 26)
                placa[m] = np.char.add(np.char.add(np.char.add(L, letras[a]), np.char.zfill(d.astype(str), 3)), letras[b])
            else:
                cod = rng.choice(26 * 26 * 10000, size=k, replace=False)
                a, r = divmod(cod, 26 * 10000); b, d = divmod(r, 10000)
                placa[m] = np.char.add(np.char.add(np.char.add(L, letras[a]), letras[b]), np.char.zfill(d.astype(str), 4))
    V.insert(1, "placa", placa)
    V["numero_chasis"] = np.char.add("9SY", cadenas(rng, n, 14, "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"))
    V["numero_motor"] = cadenas(rng, n, 11, "ABCDEFGHJKLMNPRSTUVWXYZ0123456789")
    digito = np.array([int(x[-1]) if x[-1].isdigit() else int(x[-2]) for x in placa], dtype="int8")
    V["_digito"] = digito

    # ------------------ PROPIETARIO
    serv = V.servicio.values
    p_jur = np.select([serv == "ESTADO", serv == "COMERCIAL", serv == "OTROS"], [1.0, .45, .5], .06)
    jur = np.where(V.persona_sri.notna().values, V.persona_sri.values == "JURIDICA", rng.random(n) < p_jur)
    idx_nat = np.flatnonzero(~jur); idx_jur = np.flatnonzero(jur)
    n_nat = int(len(idx_nat) * .86); n_jur = max(2000, int(len(idx_jur) * .12))
    owner = np.empty(n, dtype="int32")
    perm = rng.permutation(idx_nat)
    owner[perm[:n_nat]] = np.arange(1, n_nat + 1)
    owner[perm[n_nat:]] = rng.integers(1, n_nat + 1, len(perm) - n_nat)
    z = rng.zipf(1.6, len(idx_jur)); z = (z - 1) % n_jur
    owner[idx_jur] = n_nat + 1 + z
    V["propietario_id"] = owner
    n_prop = n_nat + n_jur

    prov_owner = np.empty(n_prop, dtype=object)
    prov_owner[owner - 1] = V.provincia.values
    vac = pd.isna(prov_owner); prov_owner[vac] = rng.choice(list(LETRA_PROV), vac.sum())
    cant_owner = np.empty(n_prop, dtype=object); cant_owner[owner - 1] = V.canton.values
    vac = pd.isna(cant_owner); cant_owner[vac] = [CAPITAL[p] for p in prov_owner[vac]]

    es_jur = np.zeros(n_prop, bool); es_jur[n_nat:] = True
    sexo_f = rng.random(n_prop) < .42
    nom1 = np.where(sexo_f, rng.choice(NOMBRES_M, n_prop), rng.choice(NOMBRES_H, n_prop))
    nom2 = np.where(sexo_f, rng.choice(NOMBRES_M, n_prop), rng.choice(NOMBRES_H, n_prop))
    ap1 = rng.choice(APELLIDOS, n_prop); ap2 = rng.choice(APELLIDOS, n_prop)
    nombres = np.char.add(np.char.add(nom1, " "), nom2).astype(object)
    apellidos = np.char.add(np.char.add(ap1, " "), ap2).astype(object)
    tipos_emp = np.array(["CIA. LTDA.", "S.A.", "S.A.S."])
    razon = np.char.add(np.char.add(np.char.add(np.char.add(ap1, " "), ap2), " "), rng.choice(tipos_emp, n_prop))
    coop = rng.random(n_prop) < .35
    razon = np.where(coop, np.char.add("COOPERATIVA DE TRANSPORTE ", ap1), razon)
    nombres[es_jur] = razon[es_jur]; apellidos[es_jur] = None

    # cédula: provincia (2) + 3er dígito <6 + 6 dígitos + verificador módulo 10 ; RUC jurídica: 3er dígito 9 + 001
    cp = np.array([COD_PROV[p] for p in prov_owner])
    cuerpo = np.char.add(np.char.add(cp, rng.integers(0, 6, n_prop).astype(str)),
                         np.char.zfill(rng.integers(0, 10**6, n_prop).astype(str), 6))
    dig = (np.frombuffer(cuerpo.astype("S9").tobytes(), dtype=np.uint8).reshape(-1, 9) - 48).astype(int)
    coef = np.array([2,1,2,1,2,1,2,1,2])
    prod = dig * coef; prod = np.where(prod > 9, prod - 9, prod)
    ver = (10 - prod.sum(1) % 10) % 10
    cedula = np.char.add(cuerpo, ver.astype(str)).astype(object)
    ruc = np.char.add(np.char.add(np.char.add(cp, "9"), np.char.zfill(rng.integers(0, 10**6, n_prop).astype(str), 6)), "1001")
    cedula[es_jur] = ruc[es_jur]

    edad_p = np.clip(rng.normal(46, 14, n_prop), 18, 90).astype(int)
    nac = (np.datetime64("2026-01-01") - (edad_p * 365.25).astype(int).astype("timedelta64[D]") - rng.integers(0, 365, n_prop).astype("timedelta64[D]"))
    calle = np.char.add(np.char.add(np.char.add(rng.choice(CALLES, n_prop), " Y "), rng.choice(CALLES, n_prop)), ", ")
    direccion = np.char.add(calle, cant_owner.astype(str)).astype(object)
    telefono = np.char.add("09", np.char.zfill(rng.integers(0, 10**8, n_prop).astype(str), 8)).astype(object)
    email = np.char.add(np.char.add(np.char.add(np.char.lower(nom1), "."), np.char.lower(ap1)),
                        np.char.add(rng.integers(1, 999, n_prop).astype(str), "@correo.ejemplo.ec")).astype(object)
    # faltantes con mecanismo CONOCIDO (sirven para validar la auditoría):
    #   email  -> MAR: depende de la edad (observada)
    #   telefono -> MCAR: 3 % al azar
    #   direccion -> MNAR simulado: falta más en cantones rurales (no registrados en la fila)
    p_email = 1 / (1 + np.exp(-(edad_p - 62) / 6))
    email[rng.random(n_prop) < p_email] = None
    telefono[rng.random(n_prop) < .03] = None
    rural = ~np.isin(cant_owner, list(CAPITAL.values()))
    direccion[rng.random(n_prop) < np.where(rural, .12, .02)] = None
    nac = nac.astype(object); nac[es_jur] = None

    P = pd.DataFrame({"id": np.arange(1, n_prop + 1, dtype="int32"),
                      "tipo_persona": np.where(es_jur, "JURIDICA", "NATURAL"),
                      "cedula": cedula, "nombres": nombres, "apellidos": apellidos,
                      "fecha_nacimiento": nac, "direccion": direccion, "telefono": telefono, "email": email})
    guardar(P, "propietario", salida)
    del P, nombres, apellidos, razon, direccion, telefono, email, cedula, ruc, dig, prod
    print(f"propietarios {time.time()-t0:.0f}s")

    # ------------------ PROCESO 2026: fecha objetivo, RTV, matrícula
    nuevo = V.origen.values == "NUEVO_SRI_2026"
    mes_due = np.where(digito == 0, 11, digito + 1)                     # 1=feb ... 9=oct, 0=nov
    inicio_mes = np.array([np.datetime64(f"2026-{m:02d}-01") for m in range(1, 13)] + [np.datetime64("2027-01-01")])
    ini = inicio_mes[mes_due - 1]; fin = inicio_mes[mes_due]
    largo = (fin - ini).astype(int)
    u = rng.beta(2.2, 1.2, n)                                            # la gente deja el trámite para el final
    objetivo = ini + (u * largo).astype(int).astype("timedelta64[D]")
    # cierre ANT (30-ene a 8-mar): dígitos 1 y 2 se reprograman (dígito 1 hasta 20-jun; dígito 2 hasta abril)
    d1 = (digito == 1) & ~nuevo; d2 = (digito == 2) & ~nuevo
    objetivo[d1] = np.datetime64("2026-03-09") + rng.integers(0, 103, d1.sum()).astype("timedelta64[D]")
    objetivo[d2] = np.datetime64("2026-03-09") + rng.integers(0, 53, d2.sum()).astype("timedelta64[D]")
    # rezagados: pasan a diciembre o más tarde; más probable en vehículos viejos y motos (MAR)
    edad_v = (2026 - V.anio.values).clip(0)
    p_rez = np.clip(.05 + .006 * edad_v + .06 * moto, 0, .45)
    rez = (rng.random(n) < p_rez) & ~nuevo & ~d1 & ~d2
    objetivo[rez] = fin[rez] + rng.integers(5, 60, rez.sum()).astype("timedelta64[D]")
    # vehículos nuevos: primera matrícula días después de la compra (sistema cerrado 30-ene a 12-feb)
    fc = V.fecha_compra.values.astype("datetime64[D]")
    obj_n = fc[nuevo] + rng.integers(1, 20, nuevo.sum()).astype("timedelta64[D]")
    bloq = (obj_n >= np.datetime64("2026-01-30")) & (obj_n < np.datetime64("2026-02-12"))
    obj_n[bloq] = np.datetime64("2026-02-12") + rng.integers(0, 15, bloq.sum()).astype("timedelta64[D]")
    obj_n = np.maximum(obj_n, np.datetime64("2026-01-02"))
    objetivo[nuevo] = obj_n
    # en domingo no se atiende
    dow = (objetivo.astype("datetime64[D]").view("int64") - 4) % 7          # 0 = lunes
    objetivo = objetivo + np.where(dow == 6, 1, 0).astype("timedelta64[D]")
    # falla del sistema ANT en Loja (6 al 9 de octubre): se corre al 12-oct
    caida = (V.provincia.values == "LOJA") & (objetivo >= np.datetime64("2026-10-06")) & (objetivo <= np.datetime64("2026-10-09"))
    objetivo[caida] = np.datetime64("2026-10-12")
    # una parte nunca se presenta (abandono), más probable en vehículos viejos (MAR)
    nunca = (rng.random(n) < np.clip(.01 + .004 * edad_v, 0, .25)) & ~nuevo

    # revisión técnica (hasta 3 intentos); 2,5 % de condicionados en promedio, más en vehículos viejos
    logit = -5.0 + .085 * edad_v + .35 * (serv == "COMERCIAL") + .25 * moto - 2.5 * nuevo
    p1 = 1 / (1 + np.exp(-logit))
    presenta = (objetivo <= HOY) & ~nunca
    rows = []
    fecha = objetivo.copy()
    aprobado = np.zeros(n, bool)
    intento = np.zeros(n, "int8")
    activos = presenta.copy()
    for k in range(3):
        idx = np.flatnonzero(activos)
        f = fecha[idx]
        p = p1[idx] * (0.35 ** k)
        falla = rng.random(len(idx)) < p
        rows.append(pd.DataFrame({"vehiculo_id": idx + 1, "fecha_revision": f, "intento": k + 1,
                                  "resultado": np.where(falla, "CONDICIONAL", "APROBADO")}))
        aprobado[idx[~falla]] = True
        fecha[idx[~falla]] = f[~falla]
        nxt = idx[falla]
        fecha[nxt] = fecha[nxt] + rng.integers(3, 45, len(nxt)).astype("timedelta64[D]")
        activos = np.zeros(n, bool); activos[nxt[fecha[nxt] <= HOY]] = True
        intento[idx] = k + 1
    R = pd.concat(rows, ignore_index=True)
    nR = len(R)
    vid = R.vehiculo_id.values - 1
    # kilometraje: según edad y uso; nuevos casi 0; errores de digitación (x10) como anomalías
    # recorrido anual propio de cada vehículo (el mismo en todos sus intentos de RTV)
    anual_v = np.select([serv == "COMERCIAL", moto], [38000, 7000], 14000) * rng.lognormal(0, .35, n)
    dias = (R.fecha_revision.values.astype("datetime64[D]") - objetivo[vid].astype("datetime64[D]")).astype(int).clip(0)
    km = np.where(nuevo[vid], rng.integers(2, 400, nR) + dias * 30, edad_v[vid].clip(.5) * anual_v[vid] + dias / 365 * anual_v[vid])
    anual = anual_v
    err = rng.random(nR) < .0008
    km[err] = km[err] * 10
    km = np.round(km).astype("float64")
    # odómetro dañado/adulterado: falta más cuando el kilometraje real es alto -> MNAR
    p_na_km = 1 / (1 + np.exp(-(km - 450000) / 60000)) * .6 + .004
    km[rng.random(nR) < p_na_km] = np.nan
    obs = np.full(nR, None, dtype=object)
    cond = R.resultado.values == "CONDICIONAL"
    nd = rng.integers(1, 3, cond.sum())
    d_a = rng.choice(DEFECTOS, cond.sum(), p=PESO_DEF); d_b = rng.choice(DEFECTOS, cond.sum(), p=PESO_DEF)
    obs[cond] = np.where(nd == 1, d_a, np.char.add(np.char.add(d_a, "; "), d_b))
    leve = (~cond) & (rng.random(nR) < .08)
    obs[leve] = np.char.add("DEFECTO TIPO I (LEVE): ", rng.choice(DEFECTOS, leve.sum(), p=PESO_DEF))
    R.insert(0, "id", np.arange(1, nR + 1, dtype="int32"))
    R["kilometraje"] = km
    R["observaciones"] = obs
    guardar(R, "revision_vehicular", salida)
    del R, rows, obs, km, anual
    print(f"revisiones {nR:,} {time.time()-t0:.0f}s")

    # matrícula
    matriculado = aprobado & (fecha <= HOY)
    fmat = np.where(matriculado, fecha, np.datetime64("NaT"))
    fcad = np.where(matriculado, inicio_mes[np.minimum(mes_due, 12)] + np.timedelta64(365, "D") - np.timedelta64(1, "D"), np.datetime64("NaT"))
    vencido = ~matriculado & (fin <= HOY) & ~nuevo
    estado = np.select([matriculado, vencido], ["VIGENTE", "VENCIDA"], "PENDIENTE").astype(object)
    bloqueo = rng.random(n) < .004
    estado[bloqueo] = "BLOQUEADA"
    tramite = np.where(nuevo, "PRIMERA_VEZ", np.where(rng.random(n) < .045, "TRANSFERENCIA", "RENOVACION"))
    M = pd.DataFrame({"id": np.arange(1, n + 1, dtype="int32"), "vehiculo_id": V.id.values,
                      "propietario_id": V.propietario_id.values, "tipo_tramite": tramite,
                      "fecha_matriculacion": fmat.astype("datetime64[D]"), "fecha_caducidad": fcad.astype("datetime64[D]"),
                      "provincia": V.provincia.values, "canton": V.canton.values, "estado": estado})
    guardar(M, "matricula", salida)
    print(f"matriculas {time.time()-t0:.0f}s")

    # ------------------ PAGOS (antes de la matrícula; impuesto >= 48 h antes)
    av = V.avaluo.values.astype(float)
    imp = impuesto_propiedad(av)
    imp[(serv == "ESTADO")] = 0.0
    pesado = (V.tipo.values == "PESADO")
    pagan = presenta | ((rng.random(n) < .5) & (objetivo - np.timedelta64(3, "D") <= HOY))   # algunos pagan y aún no llegan a la RTV
    base_f = np.where(matriculado, fecha, objetivo)
    lista = []
    def pagos(tipo, valor, dias_min, dias_max, mask):
        idx = np.flatnonzero(mask)
        f = np.minimum(base_f[idx] - rng.integers(dias_min, dias_max, len(idx)).astype("timedelta64[D]"), HOY)
        lista.append(pd.DataFrame({"vehiculo_id": idx + 1, "tipo": tipo, "valor": np.round(valor[idx], 2),
                                   "fecha": f, "estado": "PAGADO"}))
    pagos("IMPUESTO_PROPIEDAD_SRI", imp, 2, 40, pagan)
    pagos("TASA_MATRICULA_ANT", np.where(moto, 15.0, np.where(pesado, 40.0, 27.0)), 2, 40, pagan)
    pagos("TASA_VIAL_PROVINCIAL", np.where(pesado, 10.0, 5.0), 1, 40, pagan)
    pagos("TASA_RTV", np.where(V.tonelaje.fillna(1).values > 3.5, 10.10, 5.10), 0, 2, presenta)
    PG = pd.concat(lista, ignore_index=True); del lista
    nP = len(PG)
    # incumplimiento de la regla de 48 h (anomalía para el EDA) y pagos aún pendientes
    viol = rng.random(nP) < .002
    PG.loc[viol, "fecha"] = PG.loc[viol, "fecha"].values + rng.integers(3, 10, viol.sum()).astype("timedelta64[D]")
    pend = (~matriculado[PG.vehiculo_id.values - 1]) & (rng.random(nP) < .35)
    PG.loc[pend, "estado"] = "PENDIENTE"
    PG.insert(0, "id", np.arange(1, nP + 1, dtype="int32"))
    PG["fecha"] = PG.fecha.values.astype("datetime64[D]")
    guardar(PG, "pagos", salida)
    del PG
    print(f"pagos {nP:,} {time.time()-t0:.0f}s")

    # ------------------ MULTAS
    ml = []
    # calendarización (USD 25): matriculados después de su mes; exonerados dígitos 1 y 2 por el cierre de la ANT
    tarde = matriculado & (fecha >= fin) & ~nuevo & ~d1 & ~d2
    idx = np.flatnonzero(tarde)
    ml.append(pd.DataFrame({"vehiculo_id": idx + 1, "tipo": "CALENDARIZACION",
                            "descripcion": "MATRICULACION FUERA DEL MES SEGUN ULTIMO DIGITO DE PLACA",
                            "valor": 25.0, "fecha": fecha[idx], "estado": "PAGADA"}))
    idx = np.flatnonzero(vencido)
    ml.append(pd.DataFrame({"vehiculo_id": idx + 1, "tipo": "CALENDARIZACION",
                            "descripcion": "MATRICULACION FUERA DEL MES SEGUN ULTIMO DIGITO DE PLACA",
                            "valor": 25.0, "fecha": fin[idx], "estado": "PENDIENTE"}))
    # no aprobar la RTV a tiempo (USD 50)
    noap = presenta & ~aprobado
    idx = np.flatnonzero(noap)
    ml.append(pd.DataFrame({"vehiculo_id": idx + 1, "tipo": "REVISION_TECNICA",
                            "descripcion": "NO APROBAR LA REVISION TECNICA VEHICULAR EN EL PLAZO", "valor": 50.0,
                            "fecha": np.minimum(fecha[idx], HOY), "estado": "PENDIENTE"}))
    # contravenciones de tránsito (% del SBU 2026 = USD 482)
    lam = np.select([serv == "COMERCIAL", moto], [.75, .30], .16) * np.where(nuevo, .3, 1)
    k = rng.poisson(lam)
    idx = np.repeat(np.arange(n), k)
    pw = np.array([c[3] for c in CONTRAV]); pw /= pw.sum()
    t = rng.choice(len(CONTRAV), len(idx), p=pw)
    f = np.datetime64("2025-01-01") + rng.integers(0, 690, len(idx)).astype("timedelta64[D]")
    est = np.where(rng.random(len(idx)) < .62, "PAGADA", np.where(rng.random(len(idx)) < .9, "PENDIENTE", "IMPUGNADA"))
    ml.append(pd.DataFrame({"vehiculo_id": idx + 1, "tipo": "CONTRAVENCION",
                            "descripcion": np.array([c[1] for c in CONTRAV])[t],
                            "valor": np.round(np.array([c[2] for c in CONTRAV])[t] * SBU, 2),
                            "fecha": f, "estado": est}))
    MU = pd.concat(ml, ignore_index=True); del ml
    # Regla: desde el 1-oct-2026 (fin de la excepción de la ANT) no se matricula con multas pendientes.
    # Se cumple en el 99,5 % de los casos; el 0,5 % restante queda como ANOMALÍA de control para el EDA
    # (similar a lo detectado por la Contraloría en Loja: matrículas sin validar multas).
    fm = np.where(matriculado, fecha, np.datetime64("NaT"))[MU.vehiculo_id.values - 1]
    regla = (MU.tipo.values == "CONTRAVENCION") & (MU.estado.values == "PENDIENTE") & \
            (fm >= np.datetime64("2026-10-01")) & (MU.fecha.values.astype("datetime64[D]") < fm)
    corrige = regla & (rng.random(len(MU)) < .995)
    MU.loc[corrige, "estado"] = "PAGADA"
    MU.insert(0, "id", np.arange(1, len(MU) + 1, dtype="int32"))
    MU["fecha"] = MU.fecha.values.astype("datetime64[D]")
    guardar(MU, "multas", salida)
    print(f"multas {len(MU):,} {time.time()-t0:.0f}s")

    # ------------------ guardar VEHICULO al final (ya con todo calculado)
    V = V.drop(columns=["_digito", "persona_sri", "fecha_compra"])
    cols = ["id","placa","marca","modelo","anio","color","tipo","clase","subcategoria","cilindraje","combustible",
            "servicio","pasajeros","tonelaje","avaluo","numero_chasis","numero_motor","provincia","canton","propietario_id","origen"]
    guardar(V[cols], "vehiculo", salida)
    print(f"LISTO en {time.time()-t0:.0f}s")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inec", required=True); ap.add_argument("--sri", required=True)
    ap.add_argument("--salida", default="data/sintetico"); ap.add_argument("--semilla", type=int, default=2026)
    a = ap.parse_args()
    main(a.inec, a.sri, a.salida, a.semilla)
