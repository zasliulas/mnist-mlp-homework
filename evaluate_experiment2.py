
import json
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models_experiment2 import MNIST_MLP_Experiment2


# 1. Cihazı belirle
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Kullanılan cihaz:", DEVICE)


# 2. Test veri setini yükle
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)


# 3. Modeli ve en iyi ağırlıklarını yükle
model = MNIST_MLP_Experiment2().to(DEVICE)

model_path = Path("results/experiment2/best_model.pth")

if not model_path.exists():
    raise FileNotFoundError(
        "Model dosyası bulunamadı. Önce train_experiment2.py dosyasını çalıştır."
    )

state_dict = torch.load(
    model_path,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()


# 4. Test değerlendirmesi
criterion = nn.CrossEntropyLoss()

total_loss = 0.0
correct = 0
total = 0

class_correct = [0] * 10
class_total = [0] * 10

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)
        predictions = outputs.argmax(dim=1)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        total += batch_size

        for digit in range(10):
            mask = labels == digit
            class_total[digit] += mask.sum().item()
            class_correct[digit] += (
                (predictions == labels) & mask
            ).sum().item()


test_loss = total_loss / total
test_accuracy = 100.0 * correct / total


# 5. Sonuçları ekrana yazdır
print("\n--- DENEY 2 TEST SONUÇLARI ---")
print("Test örnek sayısı:", total)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Doğru tahmin: {correct}")
print(f"Yanlış tahmin: {total - correct}")

print("\n--- SINIF BAZLI ACCURACY ---")

class_accuracies = {}

for digit in range(10):
    if class_total[digit] > 0:
        accuracy = 100.0 * class_correct[digit] / class_total[digit]
    else:
        accuracy = 0.0

    class_accuracies[str(digit)] = round(accuracy, 2)
    print(f"Rakam {digit}: {accuracy:.2f}%")


# 6. Sonuçları JSON dosyasına kaydet
results_dir = Path("results/experiment2")
results_dir.mkdir(parents=True, exist_ok=True)

results = {
    "test_samples": total,
    "test_loss": round(test_loss, 4),
    "test_accuracy": round(test_accuracy, 2),
    "correct_predictions": correct,
    "incorrect_predictions": total - correct,
    "class_accuracies": class_accuracies
}

with open(
    results_dir / "test_results.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(results, file, indent=4, ensure_ascii=False)

print(
    "\nDeney 2 test sonuçları "
    "results/experiment2/test_results.json dosyasına kaydedildi."
)

