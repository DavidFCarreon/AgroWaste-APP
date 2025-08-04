from dotenv import load_dotenv
import os
from openai import OpenAI
import json

load_dotenv()

client = OpenAI()  # toma OPENAI_API_KEY del entorno automáticamente


def obtain_features(product_name: str):
    system_prompt = (
        "Eres un experto en nutrición con acceso a datos confiables. "
        "Cuando un usuario menciona un producto, debes devolver los valores reales de un análisis proximal básico para ese producto. "
        "Siempre usa la función 'obtener_caracteristicas' para estructurar los valores, "
        "y no incluyas explicaciones ni respuestas narrativas.")
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": product_name}
        ],
        functions=[
                    {"name": "obtener_caracteristicas",
                    "description": "Obtiene los valores de un análisis proximal básico de un alimento",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Humedad": {"type": "number", "description": "Contenido de humedad (%)", "minimum": 0, "maximum": 100},
                            "Proteína": {"type": "number", "description": "Contenido de proteína (%)", "minimum": 0, "maximum": 100},
                            "Grasa": {"type": "number", "description": "Contenido de grasa (%)", "minimum": 0, "maximum": 100},
                            "Carbohidratos Totales": {"type": "number", "description": "Contenido total de carbohidratos (%)", "minimum": 0, "maximum": 100},
                            "Azúcares": {"type": "number", "description": "Contenido de azúcares (%)", "minimum": 0, "maximum": 100},
                            "Fibra Dietética": {"type": "number", "description": "Contenido de fibra dietética (%)", "minimum": 0, "maximum": 100},
                            "Fibra Cruda": {"type": "number", "description": "Contenido de fibra cruda (%)", "minimum": 0, "maximum": 100},
                            "Cenizas": {"type": "number", "description": "Contenido de cenizas (%)", "minimum": 0, "maximum": 100},
                        },
                        "required": ["Humedad", "Proteína", "Grasa", "Carbohidratos Totales",
                                    "Azúcares", "Fibra Dietética", "Fibra Cruda", "Cenizas"],
                        },
                    }
                ],
        function_call="auto")

    func_call = completion.choices[0].message.function_call
    arguments_json = func_call.arguments
    args = json.loads(arguments_json)

    return{
        "Moisture": float(args["Humedad"]),
        "Protein": float(args["Proteína"]),
        "Fat": float(args["Grasa"]),
        "Total Carbohydrates": float(args["Carbohidratos Totales"]),
        "Sugars": float(args["Azúcares"]),
        "Dietary Fiber": float(args["Fibra Dietética"]),
        "Crude Fiber": float(args["Fibra Cruda"]),
        "Ash": float(args["Cenizas"])
    }




def classify_frap(FRAP_value: float) -> str:
    if FRAP_value < 1:
        return "bajo"
    elif 1 <= FRAP_value <= 10:
        return "moderado"
    else:
        return "alto"

def obtain_comments(FRAP_value: float, product_name: str) -> str:
    # Validaciones simples
    if not isinstance(FRAP_value, (int, float)) or FRAP_value < 0:
        raise ValueError("FRAP_value debe ser un número no negativo.")
    if not isinstance(product_name, str) or not product_name.strip():
        raise ValueError("product_name debe ser un string no vacío.")

    frap_level = classify_frap(FRAP_value)

    system_prompt = (
        "Eres un experto en nutrición con acceso a datos confiables. "
        "Cuando un usuario menciona un producto junto a su capacidad antioxidante obtenida por el ensayo FRAP "
        "(poder antioxidante reductor férrico), debes devolver un comentario con sus posibles usos en la industria. "
        "Un nivel FRAP bajo (<1 µmol Fe²⁺/g) indica baja capacidad antioxidante, moderado (1-10 µmol Fe²⁺/g) indica capacidad media, "
        "y alto (>10 µmol Fe²⁺/g) indica alta capacidad antioxidante. "
        "Ajusta tu comentario y recomendaciones según este nivel. "
        "Siempre utiliza la función 'obtener_usos' para estructurar la respuesta."
    )

    user_message = (
        f"Producto: {product_name}. Valor FRAP: {FRAP_value} µmol Fe²⁺/g "
        f"(nivel {frap_level})."
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",  # o gpt-4.1 si tienes acceso
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        functions=[
            {
                "name": "obtener_usos",
                "description": "Genera comentario de un alimento con sus posibles usos en la industria basados en su capacidad antioxidante.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "Usos": {
                            "type": "string",
                            "description": "Posibles usos del alimento en la industria basados en su capacidad antioxidante (FRAP).",
                            "maxLength": 500
                        }
                    },
                    "required": ["Usos"]
                }
            }
        ],
        function_call={"name": "obtener_usos"}  # Forzar llamada a función
    )

    func_call = completion.choices[0].message.function_call
    if func_call is None:
        raise ValueError("El modelo no devolvió una llamada a función.")

    arguments_json = func_call.arguments
    if not arguments_json:
        raise ValueError("No se encontraron argumentos en la función llamada.")

    args = json.loads(arguments_json)
    return args.get("Usos", "No se pudo generar un comentario.")


# AGREGAR FUNCION PARA DEVOLVER CANTIDAD ESTIMADA DE TONELADAS DE RESUDIOS MUNDIAL ANUAL


# Ejemplos de prueba:
#if __name__ == "__main__":
#    for frap_val in [0.1, 5, 15]:
#        print(f"\nFRAP={frap_val}")
#        comment = obtain_comments(FRAP_value=frap_val, product_name="Banana")
#        print(comment)
