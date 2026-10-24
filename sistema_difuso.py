# -*- coding: utf-8 -*-
"""
Especificacion declarativa del sistema de inferencia difusa RCI-01 y
extraccion de la trazabilidad completa del proceso.

Este modulo es la unica fuente de verdad del componente inteligente:

  1. La ESPECIFICACION declara las variables linguisticas (con sus valores
     linguisticos: palabras y no numeros), las funciones de pertenencia y
     las reglas con sus conectivos logicos difusos (AND / OR).
  2. El motor skfuzzy se CONSTRUYE a partir de esa especificacion, de modo
     que el resultado que calcula el motor y el grafico que se dibuja en
     el panel provienen exactamente de la misma declaracion.
  3. Una vez ejecutado el motor se extrae la traza del proceso en las tres
     etapas del metodo de Mamdani:
        - Fuzzificacion   : grado de pertenencia de cada valor linguistico.
        - Motor de inferencia : fuerza de disparo de cada regla (agregacion,
          activacion y acumulacion).
        - Defuzzificacion : salida crisp por el metodo del centroide.

Se apoya en la API interna de skfuzzy 0.5.x:
    variable.input[sim]                  -> valor crudo de entrada
    termino.membership_value[sim]        -> grado de pertenencia
    regla.aggregate_firing[sim]          -> fuerza de disparo (agregacion)
    c.activation[sim]                    -> activacion del consecuente
    termino.cuts[sim][etiqueta_regla]    -> corte por regla
    consecutivo.output[sim]              -> salida crisp (centroide)
"""

import threading

import numpy as np
import skfuzzy as fuzzy
from skfuzzy import control as ctrl
from skfuzzy.control.term import Term, TermAggregate

# Numero de puntos con los que se muestrea cada curva de pertenencia para
# el desenho en el panel. Es suficiente para que la grafica sea fiel y
# mantiene el tamano de la respuesta JSON razonable.
PUNTOS_CURVA = 101

# ---------------------------------------------------------------------------
# 1. Especificacion declarativa
# ---------------------------------------------------------------------------

ESPECIFICACION = {
    "codigo": "RCI-01",
    "nombre": "Evaluador Inteligente de Confiabilidad",
    "descripcion": (
        "Evalua a un prospecto para la compra de un telefono movil a credito "
        "mediante logica difusa de Mamdani."
    ),
    "metodo_defuzzificacion": "Centroide",
    "umbral_aprobacion": 42,
    "variables": {
        "ingreso": {
            "titulo": "Ingresos mensuales",
            "descripcion": "Ingreso mensual del prospecto declarado en el perfil.",
            "unidad": "$",
            "tipo": "antecedente",
            "rango": [0, 1000],
            "terminos": [
                {
                    "nombre": "bajo",
                    "etiqueta": "Bajo",
                    "descripcion": "Ingresos insuficientes para la cuota.",
                    "forma": "trapmf",
                    "parametros": [0, 0, 150, 250],
                },
                {
                    "nombre": "medio",
                    "etiqueta": "Medio",
                    "descripcion": "Ingresos que cubren la cuota con esfuerzo.",
                    "forma": "trimf",
                    "parametros": [200, 400, 600],
                },
                {
                    "nombre": "alto",
                    "etiqueta": "Alto",
                    "descripcion": "Ingresos holgados para la cuota.",
                    "forma": "trapmf",
                    "parametros": [500, 700, 1000, 1000],
                },
            ],
        },
        "historial": {
            "titulo": "Historial crediticio",
            "descripcion": "Puntaje de historial crediticio (0 a 10) escalado a 100.",
            "unidad": "pts",
            "tipo": "antecedente",
            "rango": [0, 100],
            "terminos": [
                {
                    "nombre": "malo",
                    "etiqueta": "Malo",
                    "descripcion": "Historial con antecedentes de mora.",
                    "forma": "trimf",
                    "parametros": [-50, 0, 50],
                },
                {
                    "nombre": "regular",
                    "etiqueta": "Regular",
                    "descripcion": "Historial sin antecedentes relevantes.",
                    "forma": "trimf",
                    "parametros": [0, 50, 100],
                },
                {
                    "nombre": "bueno",
                    "etiqueta": "Bueno",
                    "descripcion": "Historial con pagos puntuales.",
                    "forma": "trimf",
                    "parametros": [50, 100, 150],
                },
            ],
        },
        "confiabilidad": {
            "titulo": "Nivel de confiabilidad",
            "descripcion": "Nivel de confiabilidad asignado al prospecto.",
            "unidad": "%",
            "tipo": "consecuente",
            "rango": [0, 100],
            "terminos": [
                {
                    "nombre": "baja",
                    "etiqueta": "Baja",
                    "descripcion": "Confiabilidad baja: no se recomienda el credito.",
                    "forma": "trimf",
                    "parametros": [0, 0, 45],
                },
                {
                    "nombre": "media",
                    "etiqueta": "Media",
                    "descripcion": "Confiabilidad media: credito viable con garantias.",
                    "forma": "trimf",
                    "parametros": [35, 55, 75],
                },
                {
                    "nombre": "alta",
                    "etiqueta": "Alta",
                    "descripcion": "Confiabilidad alta: credito aprobado.",
                    "forma": "trimf",
                    "parametros": [60, 100, 100],
                },
            ],
        },
        "cuotas": {
            "titulo": "Numero de cuotas",
            "descripcion": "Numero de cuotas que el sistema sugiere para el plan.",
            "unidad": "cuotas",
            "tipo": "consecuente",
            "rango": [1, 12],
            "terminos": [
                {
                    "nombre": "pocas",
                    "etiqueta": "Pocas",
                    "descripcion": "Plan a pocas cuotas.",
                    "forma": "trimf",
                    "parametros": [1, 1, 4],
                },
                {
                    "nombre": "promedio",
                    "etiqueta": "Promedio",
                    "descripcion": "Plan a un numero intermedio de cuotas.",
                    "forma": "trimf",
                    "parametros": [3, 6, 9],
                },
                {
                    "nombre": "muchas",
                    "etiqueta": "Muchas",
                    "descripcion": "Plan a muchas cuotas.",
                    "forma": "trimf",
                    "parametros": [7, 12, 12],
                },
            ],
        },
    },
    "reglas": [
        {
            "id": "R1",
            "nombre": "Ingresos altos e historial bueno",
            "descripcion": (
                "Si el ingreso mensual es alto y el historial crediticio es bueno, "
                "el prospecto es altamente confiable y se le sugieren muchas cuotas."
            ),
            "conectivo": "AND",
            "antecedentes": [["ingreso", "alto"], ["historial", "bueno"]],
            "consecuentes": [["confiabilidad", "alta"], ["cuotas", "muchas"]],
        },
        {
            "id": "R2",
            "nombre": "Ingresos medios o historial regular",
            "descripcion": (
                "Si el ingreso mensual es medio o el historial crediticio es regular, "
                "el prospecto tiene confiabilidad media y un plan promedio."
            ),
            "conectivo": "OR",
            "antecedentes": [["ingreso", "medio"], ["historial", "regular"]],
            "consecuentes": [["confiabilidad", "media"], ["cuotas", "promedio"]],
        },
        {
            "id": "R3",
            "nombre": "Ingresos bajos o historial malo",
            "descripcion": (
                "Si el ingreso mensual es bajo o el historial crediticio es malo, "
                "el prospecto tiene confiabilidad baja y se le sugieren pocas cuotas."
            ),
            "conectivo": "OR",
            "antecedentes": [["ingreso", "bajo"], ["historial", "malo"]],
            "consecuentes": [["confiabilidad", "baja"], ["cuotas", "pocas"]],
        },
    ],
}

# Simbolos de los conectivos logicos difusos para la presentacion.
SIMBOLO_CONECTIVO = {"AND": "AND (min)", "OR": "OR (max)"}


# ---------------------------------------------------------------------------
# 2. Construccion del motor a partir de la especificacion
# ---------------------------------------------------------------------------

def _funcion_pertenencia(termino, universo):
    """Devuelve la funcion de pertenencia declarada para un termino."""
    forma = termino["forma"]
    parametros = termino["parametros"]

    if forma == "trapmf":
        return fuzzy.trapmf(universo, parametros)
    if forma == "trimf":
        return fuzzy.trimf(universo, parametros)

    raise ValueError("Forma de pertenencia no soportada: %s" % forma)


def _construir_variables():
    variables = {}

    for clave, spec in ESPECIFICACION["variables"].items():
        minimo, maximo = spec["rango"]
        universo = np.arange(minimo, maximo + 1, 1)

        if spec["tipo"] == "antecedente":
            variable = ctrl.Antecedent(universo, clave)
        else:
            variable = ctrl.Consequent(universo, clave)

        for termino in spec["terminos"]:
            variable[termino["nombre"]] = _funcion_pertenencia(termino, variable.universe)

        variables[clave] = variable

    return variables


def _construir_reglas(variables):
    reglas = []

    for spec in ESPECIFICACION["reglas"]:
        conectivo = spec["conectivo"]
        if conectivo not in ("AND", "OR"):
            raise ValueError("Conectivo no soportado: %s" % conectivo)

        antecedente = None
        for variable, termino in spec["antecedentes"]:
            pieza = variables[variable][termino]
            antecedente = pieza if antecedente is None else (
                antecedente & pieza if conectivo == "AND" else antecedente | pieza
            )

        consecuentes = [variables[variable][termino] for variable, termino in spec["consecuentes"]]

        reglas.append(ctrl.Rule(antecedente, consecuentes, label=spec["id"]))

    return reglas


def _construir_motor():
    variables = _construir_variables()
    reglas = _construir_reglas(variables)
    sistema = ctrl.ControlSystem(reglas)

    return {
        "sistema": sistema,
        "variables": variables,
        "antecedentes": list(sistema.antecedents),
        "consecuentes": list(sistema.consequents),
    }


MOTOR = _construir_motor()
_CANDADO = threading.Lock()


# ---------------------------------------------------------------------------
# 3. Utilidades de trazado
# ---------------------------------------------------------------------------

def _muestrear(universo, valores, puntos=PUNTOS_CURVA):
    """Reduce una curva a una cantidad fija de puntos para enviarla al panel."""
    total = len(universo)
    indices = np.unique(
        np.linspace(0, total - 1, min(puntos, total)).astype(int)
    )

    return [
        [round(float(universo[i]), 4), round(float(valores[i]), 4)]
        for i in indices
    ]


def _anotaciones(universo, valores):
    """Soporte y nucleo de un conjunto difuso (se derivan de la curva real)."""
    valores = np.asarray(valores, dtype=float)
    universo = np.asarray(universo, dtype=float)

    positivos = np.nonzero(valores > 0)[0]
    if positivos.size == 0:
        return {"pico": None, "soporte": None, "nucleo": None}

    soporte = [float(universo[positivos[0]]), float(universo[positivos[-1]])]

    plenos = np.nonzero(valores >= 1 - 1e-9)[0]
    if plenos.size:
        nucleo = [float(universo[plenos[0]]), float(universo[plenos[-1]])]
    else:
        nucleo = None

    pico = float(universo[int(np.argmax(valores))])

    return {"pico": round(pico, 4), "soporte": soporte, "nucleo": nucleo}


def _memberships_antecedente(variable, simulador):
    """Grado de pertenencia de cada valor linguistico de un antecedente."""
    resultado = {}

    for nombre, termino in variable.terms.items():
        valor = termino.membership_value[simulador]
        resultado[nombre] = 0.0 if valor is None else round(float(valor), 4)

    return resultado


def _estructura_conectivo(nodo):
    """Describe el arbol de conectivos (AND / OR) del antecedente de una regla."""
    if isinstance(nodo, Term):
        return {
            "tipo": "termino",
            "variable": nodo.parent.label,
            "termino": nodo.label,
        }

    if isinstance(nodo, TermAggregate):
        estructura = {
            "tipo": "conectivo",
            "conectivo": "AND" if nodo.kind == "and" else "OR",
            "simbolo": "∧" if nodo.kind == "and" else "∨",
        }

        estructura["izquierda"] = _estructura_conectivo(nodo.term1)
        if nodo.kind != "not" and nodo.term2 is not None:
            estructura["derecha"] = _estructura_conectivo(nodo.term2)

        return estructura

    return {"tipo": "desconocido"}


def _curva_consecuente(variable, termino, agregado):
    """Curva recortada por el metodo de Mamdani (clipping)."""
    universo = variable.universe
    base = np.asarray(termino.mf, dtype=float)

    if agregado is None:
        recortada = base
    else:
        recortada = np.minimum(base, float(agregado))

    return _muestrear(universo, recortada)


# ---------------------------------------------------------------------------
# 4. Traza del proceso
# ---------------------------------------------------------------------------

def _etiqueta_termino(variable, termino):
    """Nombre legible del valor linguistico de un termino."""
    for spec in ESPECIFICACION["variables"].get(variable, {}).get("terminos", []):
        if spec["nombre"] == termino:
            return spec["etiqueta"]

    return termino.capitalize()


def _descripcion_termino(variable, termino):
    for spec in ESPECIFICACION["variables"].get(variable, {}).get("terminos", []):
        if spec["nombre"] == termino:
            return spec["descripcion"]

    return ""


def construir_traz(simulador, entradas):
    """Construye la traza completa del proceso difuso para el panel."""
    variables_payload = []
    fuzzificacion = []

    for clave, spec in ESPECIFICACION["variables"].items():
        variable = MOTOR["variables"][clave]
        es_entrada = spec["tipo"] == "antecedente"

        terminos_payload = []

        for termino_spec in spec["terminos"]:
            termino = variable.terms[termino_spec["nombre"]]

            curva = np.asarray(termino.mf, dtype=float)

            if es_entrada:
                grado = termino.membership_value[simulador]
                activacion = 0.0 if grado is None else round(float(grado), 4)
            else:
                agregado = termino.membership_value[simulador]
                activacion = 0.0 if agregado is None else round(float(agregado), 4)

            item = {
                "nombre": termino_spec["nombre"],
                "etiqueta": termino_spec["etiqueta"],
                "descripcion": termino_spec["descripcion"],
                "forma": termino_spec["forma"],
                "parametros": termino_spec["parametros"],
                "curva": _muestrear(variable.universe, curva),
                "anotaciones": _anotaciones(variable.universe, curva),
                "activacion": activacion,
            }

            if not es_entrada:
                item["curva_recortada"] = _curva_consecuente(
                    variable, termino, activacion
                )
                item["cortes"] = {
                    etiqueta: round(float(valor), 4)
                    for etiqueta, valor in (termino.cuts[simulador] or {}).items()
                }

            terminos_payload.append(item)

        if es_entrada:
            valor_crudo = variable.input[simulador]
            valor_crudo = None if valor_crudo is None else float(valor_crudo)
        else:
            valor_crudo = variable.output[simulador]
            valor_crudo = None if valor_crudo is None else float(valor_crudo)

        variables_payload.append({
            "clave": clave,
            "titulo": spec["titulo"],
            "descripcion": spec["descripcion"],
            "unidad": spec["unidad"],
            "tipo": spec["tipo"],
            "rango": spec["rango"],
            "valor": None if valor_crudo is None else round(valor_crudo, 4),
            "terminos": terminos_payload,
        })

    # --- Fuzzificacion -------------------------------------------------
    for variable in MOTOR["antecedentes"]:
        spec = ESPECIFICACION["variables"][variable.label]
        memberships = _memberships_antecedente(variable, simulador)

        dominante = max(memberships, key=lambda clave: memberships[clave]) if memberships else None

        fuzzificacion.append({
            "variable": variable.label,
            "titulo": spec["titulo"],
            "valor_crisp": None if variable.input[simulador] is None
                          else round(float(variable.input[simulador]), 4),
            "unidad": spec["unidad"],
            "rango": spec["rango"],
            "grados": [
                {
                    "termino": nombre,
                    "etiqueta": _etiqueta_termino(variable.label, nombre),
                    "grado": memberships[nombre],
                    "descripcion": _descripcion_termino(variable.label, nombre),
                }
                for nombre in variable.terms
            ],
            "dominante": dominante,
            "etiqueta_dominante": (
                _etiqueta_termino(variable.label, dominante) if dominante else None
            ),
        })

    # --- Motor de inferencia --------------------------------------------
    reglas_payload = []
    disparadas = 0
    fuerza_maxima = 0.0

    for regla in MOTOR["sistema"].rules:
        spec_regla = None
        for candidato in ESPECIFICACION["reglas"]:
            if candidato["id"] == regla.label:
                spec_regla = candidato
                break

        fuerza = regla.aggregate_firing[simulador]
        fuerza = 0.0 if fuerza is None else round(float(fuerza), 4)

        if fuerza > 0:
            disparadas += 1
        fuerza_maxima = max(fuerza_maxima, fuerza)

        antecedentes = []
        for termino in regla.antecedent_terms:
            grado = termino.membership_value[simulador]
            grado = 0.0 if grado is None else round(float(grado), 4)
            antecedentes.append({
                "variable": termino.parent.label,
                "termino": termino.label,
                "etiqueta": _etiqueta_termino(termino.parent.label, termino.label),
                "grado": grado,
            })

        consecuentes = []
        for pieza in regla.consequent:
            activacion = pieza.activation[simulador]
            activacion = 0.0 if activacion is None else round(float(activacion), 4)
            consecuentes.append({
                "variable": pieza.term.parent.label,
                "termino": pieza.term.label,
                "etiqueta": _etiqueta_termino(pieza.term.parent.label, pieza.term.label),
                "activacion": activacion,
                "peso": float(pieza.weight),
            })

        reglas_payload.append({
            "id": regla.label,
            "nombre": spec_regla["nombre"] if spec_regla else regla.label,
            "descripcion": spec_regla["descripcion"] if spec_regla else "",
            "conectivo": spec_regla["conectivo"] if spec_regla else "AND",
            "simbolo": "∧" if (spec_regla and spec_regla["conectivo"] == "AND") else "∨",
            "texto_conectivo": SIMBOLO_CONECTIVO.get(
                spec_regla["conectivo"] if spec_regla else "AND", "AND (min)"
            ),
            "estructura": _estructura_conectivo(regla.antecedent),
            "antecedentes": antecedentes,
            "consecuentes": consecuentes,
            "fuerza": fuerza,
            "disparo": fuerza > 0,
        })

    # --- Defuzzificacion -------------------------------------------------
    defuzzificacion = []
    for variable in MOTOR["consecuentes"]:
        spec = ESPECIFICACION["variables"][variable.label]
        valor = variable.output[simulador]
        defuzzificacion.append({
            "variable": variable.label,
            "titulo": spec["titulo"],
            "unidad": spec["unidad"],
            "rango": spec["rango"],
            "salida_crisp": None if valor is None else round(float(valor), 4),
            "metodo": ESPECIFICACION["metodo_defuzzificacion"],
        })

    score = float(MOTOR["variables"]["confiabilidad"].output[simulador])
    num_cuotas = float(MOTOR["variables"]["cuotas"].output[simulador])

    nivel = "Bajo"
    if score >= 70:
        nivel = "Alto"
    elif 40 <= score < 70:
        nivel = "Medio"

    return {
        "sistema": {
            "codigo": ESPECIFICACION["codigo"],
            "nombre": ESPECIFICACION["nombre"],
            "descripcion": ESPECIFICACION["descripcion"],
            "metodo_defuzzificacion": ESPECIFICACION["metodo_defuzzificacion"],
            "umbral_aprobacion": ESPECIFICACION["umbral_aprobacion"],
        },
        "entrada": entradas,
        "variables": variables_payload,
        "fuzzificacion": fuzzificacion,
        "inferencia": {
            "reglas_total": len(reglas_payload),
            "reglas_disparadas": disparadas,
            "fuerza_maxima": round(fuerza_maxima, 4),
            "reglas": reglas_payload,
        },
        "defuzzificacion": defuzzificacion,
        "resultado": {
            "puntaje_confianza": round(score, 2),
            "nivel_riesgo": nivel,
            "cuotas_recomendadas": int(round(num_cuotas)),
            "aprobado": "SI" if score >= ESPECIFICACION["umbral_aprobacion"] else "NO",
        },
    }


# ---------------------------------------------------------------------------
# 5. API publica del modulo
# ---------------------------------------------------------------------------

def simular(ingreso, historial):
    """Ejecuta el sistema difuso y devuelve la traza completa del proceso."""
    ingreso = float(ingreso)
    historial = float(historial)

    rango_ingreso = ESPECIFICACION["variables"]["ingreso"]["rango"]
    rango_historial = ESPECIFICACION["variables"]["historial"]["rango"]

    ingreso = min(max(ingreso, rango_ingreso[0]), rango_ingreso[1])
    historial = min(max(historial, rango_historial[0]), rango_historial[1])

    with _CANDADO:
        simulador = ctrl.ControlSystemSimulation(MOTOR["sistema"])
        simulador.input["ingreso"] = ingreso
        simulador.input["historial"] = historial
        simulador.compute()

        return construir_traz(simulador, {"ingreso": ingreso, "historial": historial})


def especificacion_visual():
    """Estructura estatica del sistema (variables, valores y reglas)."""
    variables = []

    for clave, spec in ESPECIFICACION["variables"].items():
        variables.append({
            "clave": clave,
            "titulo": spec["titulo"],
            "descripcion": spec["descripcion"],
            "unidad": spec["unidad"],
            "tipo": spec["tipo"],
            "rango": spec["rango"],
            "terminos": [
                {
                    "nombre": termino["nombre"],
                    "etiqueta": termino["etiqueta"],
                    "descripcion": termino["descripcion"],
                    "forma": termino["forma"],
                    "parametros": termino["parametros"],
                }
                for termino in spec["terminos"]
            ],
        })

    return {
        "sistema": {
            "codigo": ESPECIFICACION["codigo"],
            "nombre": ESPECIFICACION["nombre"],
            "descripcion": ESPECIFICACION["descripcion"],
            "metodo_defuzzificacion": ESPECIFICACION["metodo_defuzzificacion"],
        },
        "variables": variables,
        "reglas": [
            {
                "id": regla["id"],
                "nombre": regla["nombre"],
                "descripcion": regla["descripcion"],
                "conectivo": regla["conectivo"],
                "simbolo": "∧" if regla["conectivo"] == "AND" else "∨",
                "texto_conectivo": SIMBOLO_CONECTIVO[regla["conectivo"]],
                "antecedentes": [
                    {
                        "variable": variable,
                        "termino": termino,
                        "etiqueta": _etiqueta_termino(variable, termino),
                    }
                    for variable, termino in regla["antecedentes"]
                ],
                "consecuentes": [
                    {
                        "variable": variable,
                        "termino": termino,
                        "etiqueta": _etiqueta_termino(variable, termino),
                    }
                    for variable, termino in regla["consecuentes"]
                ],
            }
            for regla in ESPECIFICACION["reglas"]
        ],
    }
