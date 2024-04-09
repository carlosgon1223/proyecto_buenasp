import os
import radon
from radon.cli.harvest import CCHarvester

def obtener_complejidad_por_archivo(ruta_repositorio):
    # Lista para almacenar la complejidad ciclomática de cada archivo
    complejidad_por_archivo = {}

    for ruta, directorios, archivos in os.walk(ruta_repositorio):
        for archivo in archivos:
            if archivo.endswith('.py'):
                ruta_completa = os.path.join(ruta, archivo)
                with open(ruta_completa, 'r', encoding='utf-8') as f:
                    código = f.read()
                complejidad = radon.complexity_cc_rank(código)
                complejidad_por_archivo[ruta_completa] = complejidad

    return complejidad_por_archivo

ruta_repositorio = '/Users/carlosgon/Desktop/entrega1_proyecto_BP/scikit-learn'  # Cambia la ruta al directorio de tu repositorio
complejidad_por_archivo = obtener_complejidad_por_archivo(ruta_repositorio)

with open('output2.txt', 'w') as f:
    for archivo, complejidad in complejidad_por_archivo.items():
        f.write(f"{archivo} {complejidad}\n")
