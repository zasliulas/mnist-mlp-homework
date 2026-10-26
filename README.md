# MNIST MLP: El Yazısı Rakam Tanıma

## 1. Projenin Amacı

Bu projede MNIST veri kümesi kullanılarak el yazısı rakamlarını sınıflandırmak amacıyla iki farklı Çok Katmanlı Algılayıcı (Multilayer Perceptron - MLP) modeli geliştirilmiştir. Modellerin performansları doğruluk, kayıp, parametre sayısı ve eğitim süresi açısından karşılaştırılmıştır.

Ayrıca kendi el yazımla yazdığım öğrenci numaramın görseli üzerinde rakam tahmini gerçekleştirilmiştir.

**Öğrenci numarası:** 23120205091

## 2. Kullanılan Modeller

### Model 1
- Mimari: 784 → 256 → 128 → 10
- Parametre sayısı: 235.146
- En iyi doğrulama doğruluğu: %97,80

### Model 2
- Mimari: 784 → 128 → 64 → 10
- Parametre sayısı: 109.386
- En iyi doğrulama doğruluğu: %97,58

Giriş katmanı, 28 × 28 boyutundaki görüntülerin 784 boyutlu vektörlere dönüştürülmesiyle oluşturulmuştur. Çıkış katmanı, 0–9 arasındaki on rakam sınıfını temsil etmektedir.

## 3. Deney Sonuçları

| Ölçüt | Model 1 | Model 2 |
|---|---:|---:|
| Parametre sayısı | 235.146 | 109.386 |
| En iyi validation accuracy | %97,80 | %97,58 |
| Test accuracy | %97,78 | %97,50 |
| Test loss | 0,0983 | 0,0865 |
| Doğru tahmin | 9.778 | 9.750 |
| Yanlış tahmin | 222 | 250 |
| Eğitim süresi | 74,14 sn | 70,02 sn |

Test değerlendirmeleri her iki model için de 10.000 örnek üzerinden yapılmıştır.

## 4. Sonuçların Genel Değerlendirmesi

Model 1, test kümesinde %97,78 doğruluk elde ederek Model 2'nin %97,50 sonucundan daha yüksek bir sınıflandırma başarısı göstermiştir. Model 1 ayrıca 28 daha az yanlış tahmin yapmıştır.

Model 2 ise 109.386 parametreyle daha küçük bir mimariye sahiptir. Model 1'e kıyasla yaklaşık %53,5 daha az parametre kullanmış ve 4,12 saniye daha kısa sürede eğitilmiştir. Bunun yanında Model 2'nin test loss değeri daha düşük çıkmıştır.

Bu sonuçlar, daha küçük bir modelin daha düşük hesaplama maliyetiyle rekabetçi bir doğruluk elde edebileceğini göstermektedir. Genel değerlendirmede Model 1 doğruluk açısından, Model 2 ise model boyutu ve eğitim süresi açısından avantajlıdır.

## 5. El Yazısı Öğrenci Numarası Tahmini

`handwritten_digits/ogrenci_numaram.jpg.jpeg` dosyasında kendi el yazımla yazdığım öğrenci numaram bulunmaktadır.

`predict_handwriting.py` dosyası, kaydedilmiş modelleri kullanarak el yazısı görseli üzerinde tahmin yapılmasını sağlar. Projede bu tahmin işlemi görsel çıktı ile gösterilmektedir.

## 6. Proje Dosyaları

- `models.py`: Model 1 mimarisi
- `models_experiment2.py`: Model 2 mimarisi
- `train.py`: Model 1 eğitim kodu
- `train_experiment2.py`: Model 2 eğitim kodu
- `evaluate.py`: Model 1 değerlendirme kodu
- `evaluate_experiment2.py`: Model 2 değerlendirme kodu
- `compare_results.py`: Deney sonuçlarını karşılaştırma
- `plot_model1.py`: Eğitim grafiklerini oluşturma
- `predict_handwriting.py`: El yazısı rakam tahmini
- `handwritten_digits/`: Öğrenci numarası görseli
- `results/`: Model ağırlıkları, grafikler, eğitim geçmişleri ve test sonuçları

## 7. Kullanılan Teknolojiler

- Python
- PyTorch
- Torchvision
- Matplotlib
- Pillow
- MNIST veri kümesi

## 8. Çalıştırma

Proje klasöründe sanal ortam oluşturulup etkinleştirildikten ve gerekli kütüphaneler kurulduktan sonra ilgili Python dosyaları çalıştırılabilir.

El yazısı tahmini için:

```bash
python predict_handwriting.py
```

Model eğitim ve değerlendirme adımları için ilgili `train_*.py` ve `evaluate*.py` dosyaları kullanılabilir. Mevcut sonuç dosyaları `results/` klasöründe bulunmaktadır.
