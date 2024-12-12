import scipy.io

# Cargar el archivo .mat
mat_file = './18_Practica2/annotation_0001.mat'
data = scipy.io.loadmat(mat_file)

# Mostrar las claves del diccionario
print(data.keys())