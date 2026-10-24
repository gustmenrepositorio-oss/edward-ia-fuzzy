# -*- coding: utf-8 -*-
"""
Microservicio de Evaluacion Inteligente de Confiabilidad (RCI-01).

Componente inteligente de logica difusa que evalua a un prospecto para la
compra de un telefono movil a credito. El motor (scikit-fuzzy) y su
especificacion declarativa viven en `sistema_difuso.py`; este archivo es la
capa HTTP.

Endpoints
---------
GET  /salud      -> estado del servicio.
GET  /sistema    -> anatomia del sistema: variables linguisticas, valores
                    linguisticos y reglas con sus conectivos.
POST /proceso    -> traza completa del proceso difuso para unos valores dados
                    (fuzzificacion, motor de inferencia, defuzzificacion).
                    Es lo que alimenta el panel grafico en tiempo real.
POST /evaluar    -> evaluacion de un prospecto por cedula (consulta el perfil
                    financiero) + la misma traza del proceso.

Configuracion por variables de entorno (opcionales):
    IA_DB_HOST, IA_DB_PORT, IA_DB_USER, IA_DB_PASSWORD, IA_DB_NAME
    IA_HOST, IA_PORT
"""

import os

import mysql.connector
from flask import Flask, jsonify, request
from flask_cors import CORS

import sistema_difuso

app = Flask(__name__)
CORS(app)

DB_HOST = os.environ.get('IA_DB_HOST', '127.0.0.1')
DB_PORT = int(os.environ.get('IA_DB_PORT', '3306'))
DB_USER = os.environ.get('IA_DB_USER', 'root')
DB_PASSWORD = os.environ.get('IA_DB_PASSWORD', '')
DB_NAME = os.environ.get('IA_DB_NAME', 'sistema_edward')


def _conectar():
    """Conexion a la base de datos del sistema."""
    return mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )


def _perfil_financiero(cedula):
    """Lee el perfil financiero del prospecto."""
    conn = _conectar()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT ingresos_mensuales, score_credito "
            "FROM perfiles_financiamiento WHERE cedula_persona = %s",
            (cedula,)
        )
        return cursor.fetchone()
    finally:
        try:
            cursor.close()
        finally:
            conn.close()


@app.route('/salud', methods=['GET'])
def salud():
    """Estado del microservicio."""
    return jsonify({
        "ok": True,
        "servicio": sistema_difuso.ESPECIFICACION["codigo"],
        "nombre": sistema_difuso.ESPECIFICACION["nombre"],
    })


@app.route('/sistema', methods=['GET'])
def obtener_sistema():
    """Anatomia del sistema difuso: variables, valores linguisticos y reglas."""
    return jsonify(sistema_difuso.especificacion_visual())


@app.route('/proceso', methods=['POST'])
def proceso_difuso():
    """
    Traza del proceso difuso para unos valores de entrada.

    Es el endpoint que consume el panel grafico: el usuario mueve los
    controles de entrada y el panel se redibuja con la fuzzificacion, el
    disparo de cada regla y la defuzzificacion.
    """
    datos = request.get_json(silent=True) or {}

    try:
        ingreso = float(datos.get('ingreso', 0))
        historial = float(datos.get('historial', 0))
    except (TypeError, ValueError):
        return jsonify({"error": "Los valores de entrada deben ser numericos."}), 400

    try:
        return jsonify(sistema_difuso.simular(ingreso, historial))
    except Exception as error:  # pragma: no cover - red de seguridad
        return jsonify({"error": str(error)}), 400


@app.route('/evaluar', methods=['POST'])
def evaluar_cliente():
    """Evalua al prospecto de una cedula y devuelve la traza del proceso."""
    datos = request.get_json(silent=True) or {}
    cedula = datos.get('cedula')

    if not cedula:
        return jsonify({"error": "Debe indicar la cedula del prospecto."}), 400

    try:
        perfil = _perfil_financiero(cedula)
    except Exception as error:
        return jsonify({"error": "No se pudo leer el perfil financiero: %s" % error}), 500

    if not perfil:
        return jsonify({"error": "No se encontro perfil financiero"}), 404

    try:
        ingreso = float(perfil['ingresos_mensuales'])
        historial = float(perfil['score_credito']) * 10
    except (TypeError, ValueError):
        return jsonify({"error": "El perfil financiero tiene datos incompletos."}), 400

    try:
        traza = sistema_difuso.simular(ingreso, historial)
    except Exception as error:  # pragma: no cover - red de seguridad
        return jsonify({"error": str(error)}), 400

    respuesta = dict(traza["resultado"])
    respuesta["proceso"] = traza

    return jsonify(respuesta)


if __name__ == '__main__':
    host = os.environ.get('IA_HOST', '127.0.0.1')
    port = int(os.environ.get('IA_PORT', '5000'))

    print("=" * 60)
    print("  %s - %s" % (
        sistema_difuso.ESPECIFICACION["codigo"],
        sistema_difuso.ESPECIFICACION["nombre"]
    ))
    print("  Escuchando en http://%s:%d" % (host, port))
    print("  Base de datos: %s/%s" % (DB_HOST, DB_NAME))
    print("=" * 60)

    app.run(host=host, port=port, debug=False)
