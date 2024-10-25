import os
import time
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import xml.etree.ElementTree as etree
import skimage
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC


#Parsetja el fitxer xml i recupera la informació necessaria per trobar la cara de l'animal
#
def extract_xml_annotation(filename):
    """Parse the xml file
    :param filename: str
    :return annotation: diccionari
    """
    z = etree.parse(filename)
    objects = z.findall('./object')
    size = (int(float(z.find('.//width').text)), int(float(z.find('.//height').text)))
    dds = []
    for obj in objects:
        dds.append(obj.find('name').text)
        dds.append([int(float(obj.find('bndbox/xmin').text)),
                                      int(float(obj.find('bndbox/ymin').text)),
                                      int(float(obj.find('bndbox/xmax').text)),
                                      int(float(obj.find('bndbox/ymax').text))])

    return {'size': size, 'informacio': dds}

# Selecciona la cara de l'animal i la transforma a la mida indicat al paràmetre mida_desti
def retall_normalitzat(imatge, dades, mida_desti=(64,64)):
    """
    Extreu la regió de la cara (ROI) i retorna una nova imatge de la mida_destí
    :param imatge: imatge que conté un animal
    :param dades: diccionari extret del xml
    :mida_desti: tupla que conté la mida que obtindrà la cara de l'animal
    """
    x, y, ample, alt = dades['informacio'][1]
    retall = np.copy(imatge[y:alt, x:ample])
    return resize(retall, mida_desti)

# Obtenir les dades de les imatges i les etiquetes
def obtenir_dades(carpeta_imatges, carpeta_anotacions, mida=(64, 64)):
    """Genera la col·lecció de cares d'animals i les corresponents etiquetes
    :param carpeta_imatges: string amb el path a la carpeta d'imatges
    :param carpeta_anotacions: string amb el path a la carpeta d'anotacions
    :param mida: tupla que conté la mida que obtindrà la cara de l'animal
    :return:
        images: numpy array 3D amb la col·lecció de cares
        etiquetes: llista binaria 0 si l'animal és un moix 1 en cas contrari
        image_files: llista d'imatges en el mateix ordre
        annotation_files: llista d'anotacions en el mateix ordre
    """

    # Leer los archivos de la carpeta y ordenar alfabéticamente para garantizar un orden consistente
    image_files = sorted([f for f in os.listdir(carpeta_imatges) if f.endswith('.png')])
    annotation_files = sorted([f.replace('.png', '.xml') for f in image_files])

    # Inicializar matrices de imágenes y etiquetas
    n_elements = len(image_files)
    imatges = np.zeros((mida[0], mida[1], n_elements), dtype=np.float16)
    etiquetes = []

    # Recorre los elementos de las dos carpetas
    for idx, image_name in enumerate(image_files):
        image_path = os.path.join(carpeta_imatges, image_name)
        annotation_path = os.path.join(carpeta_anotacions, annotation_files[idx])

        # Verificar que el archivo de anotación exista
        if not os.path.exists(annotation_path):
            print(f"Advertencia: Anotación no encontrada para la imagen {image_name}")
            continue

        # Cargar imagen y anotación
        imatge = imread(image_path, as_gray=True)
        anotacions = extract_xml_annotation(annotation_path)

        # Procesar la imagen
        cara_animal = retall_normalitzat(imatge, anotacions, mida)
        tipus_animal = anotacions["informacio"][0]
        imatges[:, :, idx] = cara_animal
        etiquetes.append(0 if tipus_animal == "cat" else 1)

    # Devolver las rutas completas
    image_paths = [os.path.join(carpeta_imatges, img) for img in image_files]
    annotation_paths = [os.path.join(carpeta_anotacions, ann) for ann in annotation_files]

    return imatges, etiquetes, image_paths, annotation_paths


# Obtenir les dades de les imatges i les etiquetes
def obtenir_dadesBiel(carpeta_imatges, carpeta_anotacions, mida=(64, 64)):
    """Genera la col·lecció de cares d'animals i les corresponents etiquetes
    :param carpeta_imatges: string amb el path a la carpeta d'imatges
    :param carpeta_anotacions: string amb el path a la carpeta d'anotacions
    :param mida: tupla que conté la mida que obtindrà la cara de l'animal
    :return:
        images: numpy array 3D amb la col·lecció de cares
        etiquetes: llista binaria 0 si l'animal és un moix 1 en cas contrari
    """

    n_elements = len([entry for entry in os.listdir(carpeta_imatges) if os.path.isfile(os.path.join(carpeta_imatges, entry))])
    # Una matriu 3D: mida x mida x nombre d'imatges
    imatges = np.zeros((mida[0], mida[1], n_elements), dtype=np.float16)
    # Una llista d'etiquetes
    etiquetes = [0] * n_elements

    #  Recorre els elements de les dues carpetes: llegeix una imatge i obté la informació interessant del xml
    with os.scandir(carpeta_imatges) as elements:

        for idx, element in enumerate(elements):
            nom = element.name.split(".")
            nom_fitxer = nom[0] + ".xml"
            imatge = imread(carpeta_imatges + os.sep + element.name, as_gray=True)
            anotacions = extract_xml_annotation(carpeta_anotacions + os.sep + nom_fitxer)

            cara_animal = retall_normalitzat(imatge, anotacions, mida)
            tipus_animal = anotacions["informacio"][0]

            imatges[:, :, idx] = cara_animal
            etiquetes[idx] = 0 if tipus_animal == "cat" else 1

            # Imprimir información adicional para la muestra 3093
            if nom_fitxer == "Cats_Test3093.xml":
                print(f"Muestra {idx}: Tipo de animal: {tipus_animal}, Etiqueta asignada: {etiquetes[idx]}")
                print(f"Contenido del archivo XML para la muestra 3093: {anotacions}")
                print(nom_fitxer)
                print(etiquetes[idx])

    return imatges, etiquetes


# ---------------------------------------------------------------------------------------------
# -                               FUNCIONES A IMPLEMENTAR                                     -
# ---------------------------------------------------------------------------------------------

def obtenirHoG(imagen, ppc = (8, 8), cpb = (2, 2), o = 9):
    """ Función para extraer características HoG de una sola imagen y mostrarla. """
    
    # Extraer HoG de la imagen
    caracteristicas, imagenHOG = skimage.feature.hog(imagen, 
                                                     pixels_per_cell=ppc, 
                                                     cells_per_block= cpb, 
                                                     orientations = o,       # Sirve para dividir el rango de ángulos en subrangos
                                                     feature_vector = True,  # Devuelve un vector de características
                                                     visualize = True)       # Devuelve la imagen HoG
    
    # Mostrar imagen original y HoG
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax[0].imshow(imagen)
    ax[0].set_title('Imagen Original')
    
    ax[1].imshow(imagenHOG)
    ax[1].set_title('Imagen HOG')
    
    plt.show()

    return caracteristicas  # Devuelve las características HoG como vector


def configuracionHoG(imatges):
    """ Función para probar diferentes configuraciones de HoG en un conjunto de imágenes. """
    
    # Definir las configuraciones de HoG:
    hog_configs = [
        #{'ppc': (4, 4), 'cpb': (2, 2), 'o': 9},
        #{'ppc': (4, 4), 'cpb': (8, 8), 'o': 9},
        #{'ppc': (6, 6), 'cpb': (2, 2), 'o': 9},
        {'ppc': (8, 8), 'cpb': (2, 2), 'o': 9}
        #{'ppc': (8, 8), 'cpb': (2, 2), 'o': 18}
    ]
    
    # Probar cada configuración en todas las imágenes
    for config in hog_configs:
        
        features_list = []
        
        for i in range(imatges.shape[2]): # Iterar sobre todas las imágenes
            # Obtener características HoG para cada imagen
            caracteristiques = obtenirHoG(imatges[:,:,i], 
                                          ppc = config['ppc'], 
                                          cpb = config['cpb'], 
                                          o = config['o'])
            
            features_list.append(caracteristiques)
        
        # Convertir la lista de características en un array numpy
        caracteristiques_hog = np.array(features_list)
        
        # Guardar las características de cada configuración en un archivo
        filename = f"caracteristiques_hog_ppc{config['ppc'][0]}_cpb{config['cpb'][0]}_o{config['o']}.npy"
        np.save(filename, caracteristiques_hog)
        print(f"Guardadas todas las características en {filename}")


def entrenamiento_SVM(caract, etiq):
    """ Función para entrenar un modelo SVM con diferentes kernels. """

    caracteristicas = np.load(caract)
    etiquetas = np.load(etiq)

    # Separación de los datos en entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(caracteristicas, etiquetas, test_size=0.2, random_state=42)

    # Estandarización de los datos:
    scaler = StandardScaler()
    X_transformed = scaler.fit_transform(X_train)
    X_test_transformed = scaler.transform(X_test)

    param_kernels = {
        'rbf': {'kernel': ['rbf'], 'C': [3.5, 3], 'gamma': ['scale', 'auto'], 'tol': [1.5, 1], 'max_iter': [2000, 2500]},
        #'linear': {'kernel': ['linear'], 'C': [9e-5, 1e-4, 11e-5, 0.001, 0.01], 'tol': [1.5, 1, 0.1, 1e-2], 'max_iter': [250, 500, 1000, 2000]},
        #'poly': {'kernel': ['poly'], 'C': [0.85, 1], 'degree': [1, 2], 'gamma': ['scale', 'auto'], 'coef0': [0.1, 0.15, 0.2], 'max_iter': [1500, 1000], 'tol': [0.75, 0.5]},
    }



    """ Revisar si esto hace falta o no
    # Llamada a la función para entrenar un modelo SVM específico
    #params_debug = {'kernel': 'linear', 'C': 0.01, 'gamma': 'scale', 'tol': 1e-2}
    #make_fix_model(X_transformed, y_train, X_test_transformed, y_test, params_debug)
    """


    best_models = {}
    for kernel in param_kernels.keys():
        print(f"Entrenando modelo SVM con kernel {kernel}")
        start_time = time.time()

        svm = SVC()
        grid_search = GridSearchCV(svm, param_kernels[kernel], cv=5, n_jobs=-1)

        grid_search.fit(X_transformed, y_train)

        best_models[kernel] = grid_search.best_estimator_
        print(f"Entrenamiento finalizado en {time.time() - start_time:.2f} segundos")
        print(f"Mejores parámetros encontrados: {grid_search.best_params_}")
        print(f"Mejor precisión encontrada: {grid_search.best_score_}") # Esto no es accuracy o si?
        print("---------------------------------------------------")

    for kernel, model in best_models.items():
        y_pred = model.predict(X_test_transformed)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Precisión del modelo SVM con kernel {kernel}: {accuracy:.3f}")
        print("---------------------------------------------------")
    
    return


def contar_muestras(data):
    """ Cuenta el número de muestras de cada clase. """
    
    clases, conteo = np.unique(data, return_counts=True)
    
    print("Número de muestras por clase (0 -> Gato, 1 -> Perro):")
    for clase, num in zip(clases, conteo):
        print(f"Clase {clase}: {num} ")

    print("\n")
    return
    

def process_images():
    """ Función para procesar X cantidad de imágenes y visualizar las características HoG. """

    image_files = [f"gatigos/images/Cats_Test{i}.png" for i in range(3)]
    annotation_files = [f"gatigos/annotations/Cats_Test{i}.xml" for i in range(3)]
    mida = (64, 64)

    for img_file, ann_file in zip(image_files, annotation_files):
        imagen = imread(img_file, as_gray=True)
        c = retall_normalitzat(imagen, extract_xml_annotation(ann_file), mida)
        
        obtenirHoG(c, ppc=(8,8), cpb=(2,2), o = 18)

    return


def obtener_datos_y_hog():
    """ Función para obtener los datos, extraer características HoG y guardarlas en un archivo. """
    carpeta_images = "gatigos/images"  # NO ES POT MODIFICAR
    carpeta_anotacions = "gatigos/annotations"  # NO ES POT MODIFICAR
    mida = (64, 64)  # DEFINEIX LA MIDA, ES RECOMANA COMENÇAR AMB 64x64
    
    #imatges, etiquetes = obtenir_dades(carpeta_images, carpeta_anotacions, mida)
    #np.save("etiquetas.npy", etiquetes)

    # Obtener imágenes, etiquetas y listas de nombres de archivos
    imatges, etiquetes, image_files, annotation_files = obtenir_dades(carpeta_images, carpeta_anotacions, mida)

    # Guardar etiquetas y nombres de archivo para sincronización en el procesamiento
    np.save("etiquetas.npy", etiquetes)
    np.save("image_files.npy", image_files)
    np.save("annotation_files.npy", annotation_files)

    # Generar y guardar características HoG
    #configuracionHoG(imatges)

    return


def train_and_evaluate_fix_model(X_transformed, y_train, X_test_transformed, y_test, params):
    """ Función para entrenar un modelo SVM con parámetros fijos y evaluarlo. """
    
    svm = SVC(**params) # Los '**' sirven para desempaquetar el diccionario de parámetros
    svm.fit(X_transformed, y_train)
    y_pred = svm.predict(X_test_transformed)

    # Cálculo de las métricas 'Precision' - 'Recall' - 'F1' - 'Accuracy'
    precision, sensibilidad, f1, exactitud = calcular_metricas(y_test, y_pred)
    # Realmente accuracy no lo pide.

    print(f"Exactitud: {exactitud:.3f}")
    print(f"Precisión: {precision:.3f}")
    print(f"Sensibilidad: {sensibilidad:.3f}")
    print(f"F1: {f1:.3f}\n")

    # Evaluar si el modelo está sobreajustando o subajustando
    y_train_pred = svm.predict(X_transformed)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = exactitud  # Ya calculada anteriormente

    # Evaluar si el modelo está sobreajustando o subajustando
    y_train_pred = svm.predict(X_transformed)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = exactitud  # Ya calculada anteriormente

    print(f"-- Testeo por si hay overfitting o underfitting --")
    print(f"Accuracy en entrenamiento: {train_accuracy:.3f}")
    print(f"Accuracy en prueba: {test_accuracy:.3f}")

    diferencia = abs(train_accuracy - test_accuracy)

    if train_accuracy > 0.95 and test_accuracy < 0.80:
        print(f"Diferencia: {diferencia:.3f} - El modelo podría estar sobreajustando (overfitting).")
    
    elif train_accuracy < 0.80 and test_accuracy < 0.80:
        print(f"Diferencia: {diferencia:.3f} - El modelo podría estar subajustando (underfitting).")
    
    else:
        print(f"Diferencia: {diferencia:.3f} - El modelo parece tener un buen equilibrio entre entrenamiento y prueba.")
    
    print("-------------------------------------------------------------------------------------------------\n")

    return y_pred, precision, sensibilidad, f1, exactitud


def calcular_metricas(y_test, y_pred):
    """ Función para calcular la precisión, sensibilidad, F1 y exactitud de cada modelo. """

    precision = precision_score(y_test, y_pred, average = "weighted")
    sensibilidad = recall_score(y_test, y_pred, average = "weighted")
    f1 = f1_score(y_test, y_pred, average = "weighted")
    
    exactitud = accuracy_score(y_test, y_pred)
    
    return precision, sensibilidad, f1, exactitud


def construir_tabla_y_grafico(resultados):
    """ Construye una tabla comparativa y un gráfico con los resultados obtenidos. """
    # Crear la tabla comparativa
    df_resultados = pd.DataFrame(resultados, columns=['Modelo', 'Precisión', 'Sensibilidad', 'F1', 'Exactitud'])
    print(df_resultados)

    # Crear el gráfico
    ax = df_resultados.set_index('Modelo').plot(kind='bar', figsize=(10, 6))
    plt.title('Comparación de Modelos SVM')
    plt.ylabel('Puntuación')
    plt.xticks(rotation=0)
    plt.legend(loc='lower right')

    # Añadir etiquetas con los valores a las barras:
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x(), p.get_height() * 1.01))

    plt.show()
    return


def cargar_Imagenes(indices, img_list, ann_list, y_true, y_pred, titulo):
    """ Función para cargar y mostrar imágenes clasificadas correctamente e incorrectamente. """
    
    fig, axes = plt.subplots(1, len(indices), figsize=(15, 6), sharex=True, sharey=True)
    fig.suptitle(titulo)

    for i, idx in enumerate(indices):
        imagen = imread(img_list[idx], as_gray=True)
        c = retall_normalitzat(imagen, extract_xml_annotation(ann_list[idx]), (64, 64))
        axes[i].imshow(c)
        axes[i].set_title(f'Pred: {y_pred[idx]} - Real: {y_true[idx]}')
        axes[i].axis('off')

    plt.show()
    return


def imagenes_Clasificadas(y_test, y_pred, img_test, ann_test, nImagenes=5, kernel=''):
    """Función para mostrar imágenes clasificadas correctamente e incorrectamente."""

    # Obtener índices de imágenes clasificadas correctamente e incorrectamente:
    correctas = np.where(y_test == y_pred)[0][:nImagenes]
    incorrectas = np.where(y_test != y_pred)[0][:nImagenes]

    # Mostrar las imágenes
    cargar_Imagenes(correctas, img_test, ann_test, y_test, y_pred, f"Imágenes Bien Clasificadas con kernel {kernel}")
    cargar_Imagenes(incorrectas, img_test, ann_test, y_test, y_pred, f"Imágenes Mal Clasificadas con kernel {kernel}")

    return



def main():

    """ Ya hemos obtenido los HoG de las imágenes y guardado las características 
    process_images() --> ¿lo dejo o lo quito?
    obtener_datos_y_hog()
    """

    obtener_datos_y_hog()

    # Cargar características y etiquetas en el orden correcto:
    caracteristicas = np.load("caracteristiques_hog_ppc8_cpb2_o9.npy")
    etiquetas = np.load("etiquetas.npy")
    image_files = np.load("image_files.npy")
    annotation_files = np.load("annotation_files.npy")

    # Separación de los datos en entrenamiento y prueba manteniendo sincronización
    X_train, X_test, y_train, y_test, img_train, img_test, ann_train, ann_test = train_test_split(
        caracteristicas, etiquetas, image_files, annotation_files, test_size=0.2, random_state=42)

    # Confirmación de que los datos están sincronizados
    for i, (y, img, ann) in enumerate(zip(y_test, img_test, ann_test)):
        if "Cats_Test3093" in ann:
            print(f"Verificación en conjunto de prueba: Índice {i} - Etiqueta en y_test: {y} - Imagen: {img} - Anotación: {ann}")

    # Estandarización de los datos:
    scaler = StandardScaler()
    X_transformed = scaler.fit_transform(X_train)
    X_test_transformed = scaler.transform(X_test)

    # Contar número de muestras del conjunto de Test para saber si hay que balancear las clases:
    contar_muestras(y_test)

    """ Ya hemos obtenido los mejores parámetros para cada kernel y los hemos guardado 
    entrenamiento_SVM(caracteristicas, etiquetas)
    """

    # Mejores parámetros para cada kernel:
    best_params_poly = {
        'kernel': 'poly',
        'C': 1,
        'degree': 2,
        'gamma': 'auto',
        'coef0': 0.1,
        'tol': 0.5,
        'max_iter': 1500
    }
    
    best_params_linear = {
        'kernel': 'linear',
        'C': 0.0001,
        'tol': 1,
        'max_iter': 500
    }
    
    best_params_rbf = {
        'kernel': 'rbf',
        'C': 3,
        'gamma': 'scale',
        'tol': 1,
        'max_iter': 2000
    }

    # Definir los modelos y sus parámetros
    modelos = [
        ('Linear', best_params_linear),
        ('Poly', best_params_poly),
        ('RBF', best_params_rbf)
    ]

    # Crear tabla comparativa y gráfico con los resultados obtenidos:
    resultados = []

    # Evaluación de los modelos SVM obtenidos con los mejores parámetros
    for nombre, params in modelos:
        print(f"----- Evaluando Modelo SVM con kernel {nombre} -----")
        y_pred, precision, sensibilidad, f1, exactitud = train_and_evaluate_fix_model(X_transformed, y_train, X_test_transformed, y_test, params)
        resultados.append([nombre, precision, sensibilidad, f1, exactitud])

        # Mostrar imágenes clasificadas correctamente e incorrectamente
        imagenes_Clasificadas(y_test, y_pred, img_test, ann_test, nImagenes=8, kernel=nombre)
    
    # Construir tabla comparativa y gráfico con los resultados obtenidos:
    construir_tabla_y_grafico(resultados)


if __name__ == "__main__":

    main()