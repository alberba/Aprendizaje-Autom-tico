from torchvision import datasets, transforms, models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import os
import torch.nn as nn
import torch.optim as optim
import torch
from PIL import Image
import numpy as np


# ---- CONFIGURACIÓN ----
# Rutas de las carpetas principales
ruta_base = 'C:/Users/Angel/Desktop/Aprendizaje Automático/Prácticas/P1/Aprendizaje-Autom-tico/18_Practica2/images'
clases_de_interes = ['buddha', 'dalmatian']

# Transformaciones para el dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0., 0., 0.], std=[1., 1., 1.])
])

# Custom Dataset para imágenes y bounding boxes
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, base_folder, classes, transform=None):
        self.base_folder = base_folder
        self.classes = classes
        self.transform = transform
        self.data = []

        # Recorrer cada clase y cargar imágenes y bounding boxes
        for class_idx, class_name in enumerate(classes):
            image_folder = os.path.join(base_folder, class_name)
            print(image_folder)
            bbox_folder = os.path.join(base_folder, f"{class_name}_bbox")
            print(bbox_folder)

            image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
            for image_file in image_files:
                image_path = os.path.join(image_folder, image_file)
                bbox_path = os.path.join(bbox_folder, f"{os.path.splitext(image_file)[0]}_boundingbox.txt")
                
                # Verificar si el archivo de bounding box existe
                if os.path.exists(bbox_path):
                    self.data.append((image_path, bbox_path, class_idx))

        print(f"Clase '{class_name}': {len(self.data)} imágenes cargadas.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, bbox_path, label = self.data[idx]

        # Cargar imagen
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Cargar bounding box
        with open(bbox_path, 'r') as f:
            box = eval(f.read().strip())  # Leer y convertir a lista

        # Crear target
        target = {
            'boxes': torch.tensor([box], dtype=torch.float32),
            'labels': torch.tensor([label], dtype=torch.int64)  # Índice de la clase
        }
        return image, target


# Crear el dataset para ambas clases
dataset = CustomDataset(ruta_base, clases_de_interes, transform=transform)

# Dividir en entrenamiento y prueba
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
test_loader = DataLoader(test_data, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# ---- CONFIGURACIÓN Y ENTRENAMIENTO DE FASTER R-CNN ----
# Modelo Faster R-CNN
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = len(clases_de_interes) + 1  # Número de clases + fondo
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = nn.Linear(in_features, num_classes)

# Configuración de entrenamiento
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Entrenamiento
num_epochs = 5
model.train()

for epoch in range(num_epochs):
    epoch_loss = 0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Calcular pérdida
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        epoch_loss += losses.item()

        # Optimización
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}")

# ---- EVALUACIÓN ----
model.eval()

# Probar una imagen de prueba
test_image_path = '../dataPractica/caltech101/101_ObjectCategories/buddha/example.jpg'
image = Image.open(test_image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

# Predicción
with torch.no_grad():
    predictions = model(image_tensor)

# Mostrar resultados
print("Bounding Boxes:", predictions[0]['boxes'])
print("Labels:", predictions[0]['labels'])
print("Scores:", predictions[0]['scores'])
