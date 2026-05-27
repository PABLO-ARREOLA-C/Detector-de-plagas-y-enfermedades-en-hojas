import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, request, jsonify
import json
import os

from disease_info import DISEASE_INFO

app = Flask(__name__)

# =========================
# 📦 RUTAS
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASSES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "classes.json"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "modelo_multiplanta2.pth"
)

# =========================
# 📦 CARGAR CLASES
# =========================

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    CLASSES = json.load(f)

# =========================
# 🤖 CONFIGURAR DISPOSITIVO
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =========================
# 🤖 CARGAR MODELO
# =========================

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASSES)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

# =========================
# 🔄 TRANSFORMACIONES
# =========================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )

])

# =========================
# 🌿 VALIDAR HOJA
# =========================

def es_hoja_valida(imagen):

    img = imagen.convert("RGB")

    pixels = list(img.getdata())

    verdes = 0

    total = len(pixels)

    for r, g, b in pixels:

        if g > r * 1.1 and g > b * 1.1:
            verdes += 1

    porcentaje_verde = verdes / total

    return porcentaje_verde > 0.12

# =========================
# 🌱 SEPARAR CLASE
# =========================

def separar_clase(nombre):

    partes = nombre.split("___")

    planta = partes[0].replace("_", " ")

    enfermedad = (
        partes[1].replace("_", " ")
        if len(partes) > 1
        else "Desconocido"
    )

    return planta, enfermedad

# =========================
# 🌐 HOME
# =========================

@app.route("/")
def index():

    return render_template("index.html")

# =========================
# 🔥 PREDICCIÓN
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # =========================
        # VALIDAR ARCHIVO
        # =========================

        if "file" not in request.files:

            return jsonify({
                "error": "No se envió ninguna imagen."
            }), 400

        file = request.files["file"]

        if file.filename == "":

            return jsonify({
                "error": "No se seleccionó ningún archivo."
            }), 400

        # =========================
        # ABRIR IMAGEN
        # =========================

        img = Image.open(file).convert("RGB")

        # =========================
        # VALIDAR HOJA
        # =========================

        if not es_hoja_valida(img):

            return jsonify({
                "error": "La imagen no parece una hoja de planta válida."
            }), 400

        # =========================
        # TRANSFORMAR IMAGEN
        # =========================

        img_tensor = transform(img)

        img_tensor = img_tensor.unsqueeze(0).to(device)

        # =========================
        # IA PREDICCIÓN
        # =========================

        with torch.no_grad():

            outputs = model(img_tensor)

            probs = torch.nn.functional.softmax(
                outputs[0],
                dim=0
            )

            confidence, pred = torch.max(probs, 0)

        # =========================
        # CLASE PREDICHA
        # =========================

        clase = CLASSES[pred.item()]

        # =========================
        # OBTENER INFORMACIÓN
        # =========================

        info = DISEASE_INFO.get(clase)

        # =========================
        # SI NO EXISTE EN EL DICCIONARIO
        # =========================

        if not info:

            planta, enfermedad = separar_clase(clase)

            info = {

                "planta": planta,

                "diagnostico": enfermedad,

                "descripcion":
                    "Información no disponible.",

                "tratamiento":
                    "Consultar especialista agrícola.",

                "prevencion":
                    "Realizar monitoreo constante.",

                "nivel":
                    "Desconocido"

            }

        # =========================
        # RESPUESTA JSON
        # =========================

        return jsonify({

            "planta":
                info["planta"],

            "diagnostico":
                info["diagnostico"],

            "descripcion":
                info["descripcion"],

            "tratamiento":
                info["tratamiento"],

            "prevencion":
                info["prevencion"],

            "nivel":
                info["nivel"],

            "confianza":
                round(float(confidence.item()) * 100, 2)

        })

    except Exception as e:

        return jsonify({

            "error": str(e)

        }), 500

# =========================
# 🚀 EJECUTAR
# =========================

if __name__ == "__main__":

    app.run(
        debug=True
    )