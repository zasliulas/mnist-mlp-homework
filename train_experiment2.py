
import time
import csv
import random
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from models_experiment2 import MNIST_MLP_Experiment2


# 1. Tekrarlanabilirlik
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# 2. Hiperparametreler
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Kullanılan cihaz:", DEVICE)


# 3. MNIST veri seti
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [55000, 5000],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# 4. Model, loss ve optimizer
model = MNIST_MLP_Experiment2().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# 5. Sonuçları kaydetmek için
results_dir = Path("results/experiment2")
results_dir.mkdir(parents=True, exist_ok=True)

history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": []
}

best_val_accuracy = 0.0
start_time = time.time()


# 6. Eğitim döngüsü
for epoch in range(EPOCHS):
    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        n = labels.size(0)
        train_loss_sum += loss.item() * n
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += n

    train_loss = train_loss_sum / train_total
    train_accuracy = 100 * train_correct / train_total

    # Validation
    model.eval()

    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            n = labels.size(0)
            val_loss_sum += loss.item() * n
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += n

    val_loss = val_loss_sum / val_total
    val_accuracy = 100 * val_correct / val_total

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_accuracy"].append(val_accuracy)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Val Acc: {val_accuracy:.2f}%"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            results_dir / "best_model.pth"
        )


# 7. Süre ve sonuçlar
training_time = time.time() - start_time

print("\nDeney 2 tamamlandı.")
print(f"Eğitim süresi: {training_time:.2f} saniye")
print(f"En iyi validation accuracy: {best_val_accuracy:.2f}%")


# 8. Eğitim geçmişini kaydet
with open(
    results_dir / "training_history.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:
    writer = csv.writer(file)

    writer.writerow([
        "epoch",
        "train_loss",
        "val_loss",
        "train_accuracy",
        "val_accuracy"
    ])

    for i in range(EPOCHS):
        writer.writerow([
            i + 1,
            history["train_loss"][i],
            history["val_loss"][i],
            history["train_accuracy"][i],
            history["val_accuracy"][i]
        ])

print("Eğitim geçmişi results/experiment2/training_history.csv dosyasına kaydedildi.")
