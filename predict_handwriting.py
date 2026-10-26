
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from PIL import Image

from models import MNIST_MLP
from models_experiment2 import MNIST_MLP_Experiment2


# 1. Cihazı belirle
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 2. Dosya yolları
IMAGE_PATH = Path(
    "handwritten_digits/ogrenci_numaram.jpg.jpeg"
)

MODEL1_PATH = Path("results/best_model.pth")
MODEL2_PATH = Path("results/experiment2/best_model.pth")


# 3. Modeli yükleme fonksiyonu
def load_model(model_class, model_path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model dosyası bulunamadı: {model_path.resolve()}"
        )

    model = model_class().to(DEVICE)

    state_dict = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


# 4. Ana program
def main():
    print("Kullanılan cihaz:", DEVICE)

    # Fotoğrafın varlığını kontrol et
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Fotoğraf bulunamadı: {IMAGE_PATH.resolve()}"
        )

    # Fotoğrafı aç
    image = Image.open(IMAGE_PATH).convert("RGB")

    print("Fotoğraf başarıyla açıldı.")
    print("Fotoğraf boyutu:", image.size)

    # Deney 1 modelini yükle
    model1 = load_model(
        MNIST_MLP,
        MODEL1_PATH
    )
    print("Deney 1 modeli başarıyla yüklendi.")

    # Deney 2 modelini yükle
    model2 = load_model(
        MNIST_MLP_Experiment2,
        MODEL2_PATH
    )
    print("Deney 2 modeli başarıyla yüklendi.")

    # Fotoğrafı göster
    plt.figure(figsize=(14, 7))
    plt.imshow(image)
    plt.title("El Yazısıyla Öğrenci Numarası: 23120205091")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print("\nÖn kontroller başarıyla tamamlandı.")
    print("Bir sonraki aşama: Rakamları ayırma ve tahmin.")


if __name__ == "__main__":
    main()

