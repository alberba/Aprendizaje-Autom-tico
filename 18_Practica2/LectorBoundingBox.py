import numpy as np

# Ruta al archivo .txt
bounding_box_file = 'C:/Users/Angel/Desktop/mat_extracted_txt/buddha_0001_boundingbox.txt'

# Leer el contenido del archivo y convertirlo a un array de coordenadas
with open(bounding_box_file, 'r') as f:
    content = f.read().strip()
    bounding_box = np.array(eval(content))  # Convertir texto a array NumPy

print("Bounding Box:", bounding_box)
