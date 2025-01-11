import scipy.io
import os

# Carpetas de entrada y salida
input_folder = r'C:\Users\Angel\Desktop\Aprendizaje Automático\Prácticas\P1\Aprendizaje-Autom-tico\18_Practica2\annotations'         # Carpeta principal que contiene las subcarpetas
output_folder = r'C:\Users\Angel\Desktop\Aprendizaje Automático\Prácticas\P1\Aprendizaje-Autom-tico\18_Practica2\mat_extracted_txt'  # Carpeta de salida para guardar los archivos .txt

# Función para buscar archivos .mat dentro de las subcarpetas de la carpeta principal
def find_mat_files_in_subfolders(input_folder):
    mat_files = []
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.isdir(subfolder_path):  # Verifica que sea una carpeta
            for file in os.listdir(subfolder_path):
                if file.endswith('.mat'):
                    mat_files.append((subfolder, os.path.join(subfolder_path, file)))  # Clase y ruta del archivo
    return mat_files

# Función para convertir bounding box a un formato separado por comas
def format_bounding_box(array):
    return '[' + ', '.join(map(str, array.flatten())) + ']'

# Función para convertir obj_contour a un formato con corchetes y comas
def format_obj_contour(array):
    rows = [', '.join(map(str, row)) for row in array]
    return '[' + ', '.join('[' + row + ']' for row in rows) + ']'

# Función para extraer contenido y guardar en archivos .txt
def extract_and_save(input_folder, output_folder):
    # Crear carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)
    
    # Buscar archivos .mat en las subcarpetas
    mat_files = find_mat_files_in_subfolders(input_folder)
    
    for class_name, file_path in mat_files:
        try:
            # Cargar el archivo .mat
            mat_data = scipy.io.loadmat(file_path)
            
            # Extraer datos
            box_coords = mat_data.get('box_coord', None)
            obj_contour = mat_data.get('obj_contour', None)
            
            # Determinar el identificador desde el nombre del archivo
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            identifier = base_name.split('_')[-1]
            
            # Guardar bounding box (formato separado por comas)
            if box_coords is not None:
                boundingbox_file = os.path.join(output_folder, f"{class_name}_{identifier}_boundingbox.txt")
                with open(boundingbox_file, 'w') as bbox_file:
                    bbox_file.write(format_bounding_box(box_coords))
                #print(f"Bounding box guardado: {boundingbox_file}")
            
            # Guardar object contour (formato con corchetes y comas)
            if obj_contour is not None:
                obj_contour_file = os.path.join(output_folder, f"{class_name}_{identifier}_objcontour.txt")
                with open(obj_contour_file, 'w') as contour_file:
                    contour_file.write(format_obj_contour(obj_contour))
                #print(f"Object contour guardado: {obj_contour_file}")
        
        except Exception as e:
            print(f"Error procesando {file_path}: {e}")

# Ejecutar el proceso
extract_and_save(input_folder, output_folder)

print(f"Archivos procesados y guardados en: {output_folder}")
