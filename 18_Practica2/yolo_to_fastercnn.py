import os

def yolo_to_faster_rcnn(yolo_bbox, image_width, image_height):
    """
    Convierte una bounding box de formato YOLO a Faster R-CNN.
    """
    class_id, x_center, y_center, width, height = yolo_bbox
    x_min = (x_center - width / 2) * image_width
    y_min = (y_center - height / 2) * image_height
    x_max = (x_center + width / 2) * image_width
    y_max = (y_center + height / 2) * image_height

    return [x_min, y_min, x_max, y_max], class_id

def convert_yolo_annotations_fixed_size(yolo_folder, output_folder, image_width=256, image_height=256):
    """
    Convierte archivos de anotaciones en formato YOLO a Faster R-CNN con un tamaño fijo de imagen.
    """
    os.makedirs(output_folder, exist_ok=True)

    for yolo_file in os.listdir(yolo_folder):
        if yolo_file.endswith('.txt'):
            input_path = os.path.join(yolo_folder, yolo_file)
            output_path = os.path.join(output_folder, yolo_file)

            with open(input_path, 'r') as f:
                yolo_annotations = [list(map(float, line.split())) for line in f.readlines()]

            # Convertir cada anotación
            faster_rcnn_annotations = []
            for yolo_bbox in yolo_annotations:
                faster_rcnn_bbox, class_id = yolo_to_faster_rcnn(yolo_bbox, image_width, image_height)
                faster_rcnn_annotations.append({
                    'class_id': int(class_id),
                    'bbox': faster_rcnn_bbox
                })

            # Guardar el resultado
            with open(output_path, 'w') as f_out:
                for ann in faster_rcnn_annotations:
                    bbox_str = ', '.join(map(str, ann['bbox']))
                    f_out.write(f"{ann['class_id']} {bbox_str}\n")

            print(f"Archivo convertido: {output_path}")

# Comprobar las anotaciones convertidas
def verify_converted_annotations(output_folder):
    """
    Lee y verifica las anotaciones convertidas en formato Faster R-CNN.
    """
    for file_name in os.listdir(output_folder):
        if file_name.endswith('.txt'):
            file_path = os.path.join(output_folder, file_name)
            print(f"\nContenido del archivo {file_name}:")
            with open(file_path, 'r') as f:
                for line in f.readlines():
                    print(line.strip())

# ---- CONFIGURACIÓN ----
# Carpeta con anotaciones en formato YOLO
yolo_folder = './yolo_annotations'  # Cambiar por la ruta a tu carpeta de YOLO

# Carpeta de salida para anotaciones convertidas
output_folder = './faster_rcnn_annotations'  # Cambiar por la ruta donde guardar las anotaciones

# ---- EJECUCIÓN ----
convert_yolo_annotations_fixed_size(yolo_folder, output_folder)

# Verificar las anotaciones convertidas
verify_converted_annotations(output_folder)