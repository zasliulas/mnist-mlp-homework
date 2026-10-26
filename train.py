
import torch

from prepare_data import prepare_data
from models import MNIST_MLP
from training_utils import train_model


def main():
    # 1. Veri setlerini hazırla
    train_loader, val_loader, test_loader = prepare_data(
        show_image=False
    )

    # 2. Modeli oluştur
    model = MNIST_MLP()

    print("\nModel 1: 784 -> 256 -> 128 -> 10")
    print(
        f"Parametre sayısı: "
        f"{sum(p.numel() for p in model.parameters())}"
    )

    # 3. Kullanılacak cihazı belirle
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Kullanılan cihaz: {device}")

    # 4. Modeli eğit
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        learning_rate=0.001,
        output_dir="results",
        device=device
    )

    # 5. Sonuçları yazdır
    print("\nEğitim tamamlandı.")
    print(
        f"En iyi doğrulama başarısı: "
        f"{results['best_val_accuracy']:.4f}"
    )
    print(
        f"Eğitim süresi: "
        f"{results['training_time']:.2f} saniye"
    )
    print(
        f"En iyi model dosyası: "
        f"{results['best_model_path']}"
    )


if __name__ == "__main__":
    main()

