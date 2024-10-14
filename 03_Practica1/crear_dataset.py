import os
import time
from matplotlib import pyplot as plt
import numpy as np
import xml.etree.ElementTree as etree
import skimage
from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import StandardScaler

# Importar librerías necesarias:
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC


# HOla

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


def obtenir_dades(carpeta_imatges, carpeta_anotacions, mida=(64, 64)):
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

    return imatges, etiquetes


# ---------------------------------------------------------------------------------------------
# -                               FUNCIONES A IMPLEMENTAR                                     -
# ---------------------------------------------------------------------------------------------

def obtenirHoG(imagen, ppc = (4, 4), cpb = (2, 2), o = 9):
    """ Función para extraer características HoG de una sola imagen y mostrarla. """
    
    # Extraer HoG de la imagen
    caracteristicas, imagenHOG = skimage.feature.hog(imagen, 
                                                     pixels_per_cell=ppc, 
                                                     cells_per_block= cpb, 
                                                     orientations = o,       # Sirve para dividir el rango de ángulos en subrangos
                                                     feature_vector = True,  # Devuelve un vector de características
                                                     visualize = True)       # Devuelve la imagen HoG
    
    # Mostrar imagen original y HoG
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    ax[0].imshow(imagen)
    ax[0].set_title('Imagen Original')
    
    ax[1].imshow(imagenHOG)
    ax[1].set_title('Imagen HOG')
    
    plt.show()

    return caracteristicas  # Devuelve las características HoG como vector


def configuracionsHoG(imatges):
    """ Función para probar diferentes configuraciones de HoG en un conjunto de imágenes. """
    
    # Definir las configuraciones de HoG:
    hog_configs = [
        {'ppc': (4, 4), 'cpb': (2, 2), 'o': 9},
        {'ppc': (4, 4), 'cpb': (8, 8), 'o': 9},
        {'ppc': (6, 6), 'cpb': (2, 2), 'o': 9},
        {'ppc': (8, 8), 'cpb': (4, 4), 'o': 12}
    ]
    
    # Probar cada configuración en todas las imágenes
    for config in hog_configs:
        
        features_list = []
        for i in range(imatges.shape[2]): # Iterar sobre todas las imágenes
        #for i in range(3):  
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
        #np.save(filename, caracteristiques_hog)
        print(f"Guardadas todas las características en {filename}")

def entrenamiento_SVM():
    """ Función para entrenar un modelo SVM con diferentes kernels. """

    caracteristicas = np.load("caracteristiques_hog_ppc4_cpb2_o9.npy")
    etiquetas = np.load("etiquetas.npy")

    # Separación de los datos en entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(caracteristicas, etiquetas, test_size=0.2, random_state=42)

    # Els dos algorismes es beneficien d'estandaritzar les dades
    scaler = StandardScaler()
    X_transformed = scaler.fit_transform(X_train)
    X_test_transformed = scaler.transform(X_test)

    param_kernels = {
        'rbf': {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto']},
        'linear': {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
        'poly': {'kernel': ['poly'], 'C': [0.1, 1, 10, 100], 'degree': [2, 3, 4], 'gamma': ['scale', 'auto']},
    }

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
        print(f"Mejor precisión encontrada: {grid_search.best_score_}")
        print("---------------------------------------------------")

    for kernel, model in best_models.items():
        y_pred = model.predict(X_test_transformed)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Precisión del modelo SVM con kernel {kernel}: {accuracy:.2f}")
        print("---------------------------------------------------")
    return

def main():
    """
    carpeta_images = "gatigos/images"  # NO ES POT MODIFICAR
    carpeta_anotacions = "gatigos/annotations"  # NO ES POT MODIFICAR
    mida = (64, 64)  # DEFINEIX LA MIDA, ES RECOMANA COMENÇAR AMB 64x64
    
    imatges, etiquetes = obtenir_dades(carpeta_images, carpeta_anotacions, mida)

    configuracionsHoG(imatges)
    np.save("etiquetas.npy", etiquetes)
    """

    # TODO: Entrenamiento modelo SVM con 3 kernels (lineal, polinómico y RBF) y 3 configuraciones de HoG
    
    entrenamiento_SVM()

    # TODO: Validación y test de los modelos
    

if __name__ == "__main__":

    main()









