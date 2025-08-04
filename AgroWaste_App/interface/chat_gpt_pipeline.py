from dotenv import load_dotenv
import os
from openai import OpenAI
import json

load_dotenv()

client = OpenAI()  # toma OPENAI_API_KEY del entorno automáticamente


def obtain_features(food_name: str):
    system_prompt = (
        "Eres un experto en nutrición con acceso a datos confiables. "
        "Cuando un usuario menciona un alimento, debes devolver los valores reales de un análisis proximal básico para ese producto. "
        "Siempre usa la función 'obtener_caracteristicas' para estructurar los valores, "
        "y no incluyas explicaciones ni respuestas narrativas.")

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": food_name}
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

print(obtain_features("avena"))  # Ejemplo de uso
