from dotenv import load_dotenv
import os
from openai import OpenAI
import json

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        "Total_Carbohydrates": float(args["Carbohidratos Totales"]),
        "Sugars": float(args["Azúcares"]),
        "Dietary_Fiber": float(args["Fibra Dietética"]),
        "Crude_Fiber": float(args["Fibra Cruda"]),
        "Ash": float(args["Cenizas"])
    }




def classify_frap(FRAP_value: float) -> str:
    if FRAP_value < 2:
        return "bajo"
    elif 2 <= FRAP_value <= 10:
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

    system_prompt = (f"""Un investigador está estudiando opciones para la valorización del siguiente residuo agroindustrial: {FRAP_value}.
                Se realizaron análisis de actividad antioxidante medida mediante FRAP, y de acuerdo con una clasificación de potencial antioxidante, el residuo clasificó como {frap_level}.
                ¿Qué recomendación le harías al investigador respecto a la dirección hacia la cual debería enfocar sus estudios para el desarrollo de alguna estrategia tecnológica para la valorización del residuo, tomando en cuenta la clasificación?
                Genera la recomendación con un lenguaje y formato adecuado para incluirla en un reporte técnico dirigido a un investigador con nivel de estudios de doctorado."""
                )

    user_message = (
        f"Producto: {product_name}. Valor FRAP: {FRAP_value} µmol Fe²⁺/g "
        f"(nivel {frap_level})."
    )

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
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



# Ejemplos de prueba:
#if __name__ == "__main__":
#    for frap_val in [0.1, 5, 15]:
#        print(f"\nFRAP={frap_val}")
#        comment = obtain_comments(FRAP_value=frap_val, product_name="Banana")
#        print(comment)
