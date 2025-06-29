import os

# Definir carpetas
folders = [
    "data/raw",
    "data/interim",
    "data/processed",
    "notebooks",
    "src/data",
    "src/features",
    "src/models",
    "src/visualization",
    "models",
    "reports/figures",
    "tests"
]

# Crear carpetas
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Crear archivos base
open("README.md", "w").close()
open("requirements.txt", "w").close()
open(".gitignore", "w").close()

print("Estructura de carpetas creada con éxito.")
