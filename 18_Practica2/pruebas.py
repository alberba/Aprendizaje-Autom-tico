import os
import random
import shutil

# Define las proporciones para train, val y test
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

OUTPUT_DIR = "C:/Albert/01. Proyectos/Aprendizaje Automático/18_Practica2/datasets/BuddhaDalmatian-4/"  # Directorio donde se crearán las carpetas de salida

# Directorios principales con las imágenes y etiquetas de entrada
IMAGES_DIR = "C:/Albert/01. Proyectos/Aprendizaje Automático/18_Practica2/datasets/BuddhaDalmatian-4/trai/images"  # Carpeta con las imágenes
LABELS_DIR = "C:/Albert/01. Proyectos/Aprendizaje Automático/18_Practica2/datasets/BuddhaDalmatian-4/trai/labels"  # Carpeta con las etiquetas
OUTPUT_DIR = "C:/Albert/01. Proyectos/Aprendizaje Automático/18_Practica2/datasets/BuddhaDalmatian-4/"  # Directorio donde se crearán las carpetas de salida

# Asegúrate de que las carpetas de salida existan
train_images_dir = os.path.join(OUTPUT_DIR, "train/images")
val_images_dir = os.path.join(OUTPUT_DIR, "valid/images")
test_images_dir = os.path.join(OUTPUT_DIR, "test/images")

train_labels_dir = os.path.join(OUTPUT_DIR, "train/labels")
val_labels_dir = os.path.join(OUTPUT_DIR, "valid/labels")
test_labels_dir = os.path.join(OUTPUT_DIR, "test/labels")

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)

os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)
os.makedirs(test_labels_dir, exist_ok=True)

# Listar todas las imágenes en la carpeta de entrada
images = [f for f in os.listdir(IMAGES_DIR) if os.path.isfile(os.path.join(IMAGES_DIR, f))]

# Mezclar las imágenes de forma aleatoria
random.shuffle(images)

# Calcular cuántas imágenes corresponden a cada conjunto
total_images = len(images)
train_count = int(total_images * TRAIN_SPLIT)
val_count = int(total_images * VAL_SPLIT)

# Dividir las imágenes en los conjuntos
train_images = images[:train_count]
val_images = images[train_count:train_count + val_count]
test_images = images[train_count + val_count:]

# Mover las imágenes y sus etiquetas correspondientes
def move_images_and_labels(image_list, images_source_dir, labels_source_dir, images_dest_dir, labels_dest_dir):
    for image in image_list:
        image_name = os.path.splitext(image)[0]  # Nombre base del archivo sin extensión
        label_file = f"{image_name}.txt"  # Se asume que las etiquetas tienen extensión .txt

        # Rutas de origen
        image_src_path = os.path.join(images_source_dir, image)
        label_src_path = os.path.join(labels_source_dir, label_file)

        # Rutas de destino
        image_dest_path = os.path.join(images_dest_dir, image)
        label_dest_path = os.path.join(labels_dest_dir, label_file)

        # Copiar imagen
        shutil.copy(image_src_path, image_dest_path)

        # Copiar etiqueta si existe
        if os.path.exists(label_src_path):
            shutil.copy(label_src_path, label_dest_path)

move_images_and_labels(train_images, IMAGES_DIR, LABELS_DIR, train_images_dir, train_labels_dir)
move_images_and_labels(val_images, IMAGES_DIR, LABELS_DIR, val_images_dir, val_labels_dir)
move_images_and_labels(test_images, IMAGES_DIR, LABELS_DIR, test_images_dir, test_labels_dir)

print(f"Total imágenes: {total_images}")
print(f"Train: {len(train_images)} imágenes")
print(f"Validation: {len(val_images)} imágenes")
print(f"Test: {len(test_images)} imágenes")
print("División completada. Las imágenes y etiquetas se han movido a las carpetas correspondientes.")