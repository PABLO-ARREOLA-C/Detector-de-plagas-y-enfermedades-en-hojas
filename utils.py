def traducir_clase(nombre_clase):
    traducciones = {
        "Tomato___healthy": "Tomate sano",
        "Tomato___Late_blight": "Tizón tardío en tomate",
        "Tomato___Early_blight": "Tizón temprano en tomate",

        "Soybean___healthy": "Frijol sano",

        "Coffee___healthy": "Café sano",
        "Coffee___rust": "Roya del café",

        "Mango___healthy": "Mango sano",
        "Mango___disease": "Enfermedad en mango",

        "Coconut___healthy": "Coco sano",
        "Coconut___disease": "Enfermedad en coco",
    }

    return traducciones.get(nombre_clase, nombre_clase)