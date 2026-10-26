
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path("results")
csv_path = results_dir / "training_history.csv"

if not csv_path.exists():
    raise FileNotFoundError(
        f"{csv_path} bulunamadı. Dosyanın results klasöründe olduğunu kontrol et."
    )

history = pd.read_csv(csv_path)
print("CSV sütunları:", history.columns.tolist())

# Loss grafiği
plt.figure(figsize=(8, 5))
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model 1 - Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(results_dir / "loss_curve.png")
plt.show()
plt.close()

# Accuracy grafiği
plt.figure(figsize=(8, 5))
plt.plot(history["train_accuracy"], label="Train Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Model 1 - Accuracy Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(results_dir / "accuracy_curve.png")
plt.show()
plt.close()

print("Model 1 grafikleri results klasörüne kaydedildi.")
