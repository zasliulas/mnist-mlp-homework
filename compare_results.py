
import json
import pandas as pd
from pathlib import Path

results_dir = Path("results")


# 1. Test sonuçlarını oku
with open(
    results_dir / "test_results.json",
    "r",
    encoding="utf-8"
) as file:
    model1_test = json.load(file)

with open(
    results_dir / "experiment2" / "test_results.json",
    "r",
    encoding="utf-8"
) as file:
    model2_test = json.load(file)


# 2. Eğitim geçmişlerini oku
model1_history = pd.read_csv(
    results_dir / "training_history.csv"
)

model2_history = pd.read_csv(
    results_dir / "experiment2" / "training_history.csv"
)


# 3. Validation accuracy değerlerini aynı ölçeğe getir
model1_best_val = model1_history["val_accuracy"].max()
model2_best_val = model2_history["val_accuracy"].max()

# Model 1 doğrulukları 0-1 aralığındaysa yüzdeye çevir
if model1_best_val <= 1:
    model1_best_val *= 100

# Model 2 doğrulukları 0-1 aralığındaysa yüzdeye çevir
if model2_best_val <= 1:
    model2_best_val *= 100


# 4. Karşılaştırma tablosunu oluştur
comparison = pd.DataFrame([
    {
        "Model": "Model 1 (256-128)",
        "Parametre Sayısı": 235146,
        "En İyi Validation Accuracy (%)": round(model1_best_val, 2),
        "Test Loss": model1_test["test_loss"],
        "Test Accuracy (%)": model1_test["test_accuracy"],
        "Doğru Tahmin": model1_test["correct_predictions"],
        "Yanlış Tahmin": model1_test["incorrect_predictions"],
        "Eğitim Süresi (sn)": 74.14
    },
    {
        "Model": "Model 2 (128-64)",
        "Parametre Sayısı": 109386,
        "En İyi Validation Accuracy (%)": round(model2_best_val, 2),
        "Test Loss": model2_test["test_loss"],
        "Test Accuracy (%)": model2_test["test_accuracy"],
        "Doğru Tahmin": model2_test["correct_predictions"],
        "Yanlış Tahmin": model2_test["incorrect_predictions"],
        "Eğitim Süresi (sn)": 70.02
    }
])


# 5. Sonuçları ekrana yazdır
print("\n===== MODEL KARŞILAŞTIRMASI =====")
print(comparison.to_string(index=False))


# 6. Karşılaştırmayı CSV dosyasına kaydet
comparison.to_csv(
    results_dir / "model_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nKarşılaştırma tablosu "
    "results/model_comparison.csv dosyasına kaydedildi."
)

