import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# =========================
# CONFIGURACIÓN
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "data", "plantvillage")
)

MODEL_PATH = os.path.join(BASE_DIR, "models", "modelo_multiplanta.pth")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "classes.json")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando:", device)

# =========================
# TRANSFORMACIONES
# =========================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =========================
# DATASET
# =========================

print("Cargando dataset...")
dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)

class_names = dataset.classes
num_classes = len(class_names)

print("Clases detectadas:", num_classes)

# Guardar clases (CRÍTICO para Flask)
os.makedirs(os.path.dirname(CLASSES_PATH), exist_ok=True)
with open(CLASSES_PATH, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)

# =========================
# SPLIT TRAIN / VAL
# =========================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0  # Windows safe
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# =========================
# MODELO (RESNET18 CONSISTENTE)
# =========================

print("Cargando modelo ResNet18...")

model = models.resnet18(weights="DEFAULT")

# 🔥 IMPORTANTE: capa final dinámica
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)

# =========================
# LOSS Y OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
# ENTRENAMIENTO
# =========================

print("Iniciando entrenamiento...")

for epoch in range(EPOCHS):

    # TRAIN
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total

    # VALIDATION
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {train_loss:.4f} "
          f"Train Acc: {train_acc:.2f}% "
          f"Val Acc: {val_acc:.2f}%")

# =========================
# GUARDAR MODELO
# =========================

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

torch.save(model.state_dict(), MODEL_PATH)

print("Modelo guardado en:", MODEL_PATH)
print("Entrenamiento completado 🚀")