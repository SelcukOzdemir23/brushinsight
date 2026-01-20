# 🎨 BrushInsight: AI-Powered Art Curator

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%202.x-orange)
![Gradio](https://img.shields.io/badge/Interface-Gradio-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

> **Canlı Demo:** [Hugging Face Spaces - BrushInsight](https://huggingface.co/spaces/muserrefselcukozdemir/brushinsight)

## 📖 Proje Hakkında (Abstract)

Bu çalışma, **Fırat Üniversitesi Prof. Dr. Sami Ekici'nin "Evrişimsel Sinir Ağları" dersi** kapsamında hazırlanan bir yüksek lisans final projesidir.

**BrushInsight**, derin öğrenme yöntemlerinden biri olan **Transfer Learning (Transfer Öğrenme)** tekniğini kullanarak, sanat tarihi literatüründe en etkili kabul edilen 10 ressamın eserlerini sınıflandırmayı amaçlar. Proje, sadece bir sınıflandırma modeli sunmakla kalmayıp, **Açıklanabilir Yapay Zeka (XAI)** prensipleri doğrultusunda **Grad-CAM (Gradient-weighted Class Activation Mapping)** tekniğini kullanarak modelin karar mekanizmasını görselleştirmekte ve "Kara Kutu" (Black-Box) problemini aşmayı hedeflemektedir.

### 🎯 Temel Amaçlar
1.  **Yüksek Başarı:** ResNet-50 mimarisini kullanarak sanat eseri tanıma görevinde yüksek doğruluk oranı elde etmek.
2.  **Açıklanabilirlik (Explainability):** Modelin, bir eseri neden belirli bir ressama ait olarak sınıflandırdığını ısı haritaları (Heatmaps) ile görselleştirmek.
3.  **Erişilebilirlik:** Gradio tabanlı kullanıcı dostu bir web arayüzü ile modelin herkes tarafından deneyimlenmesini sağlamak.

## 🧠 Teknik Mimari ve Metodoloji

Projede, ImageNet veri seti üzerinde önceden eğitilmiş (pre-trained) **ResNet-50** mimarisi kullanılmıştır.

### 1. Veri Seti (Dataset)
*   **Kaynak:** Kaggle - "Best Artworks of All Time"
*   **Kullanılan Veri:** En çok esere sahip ilk 10 sanatçı filtrelenerek kullanılmıştır. (Örn: Van Gogh, Picasso, Monet...)
*   **Ön İşleme:** Görüntüler 224x224 piksel boyutuna getirilmiş ve ResNet-50 standartlarına göre normalize edilmiştir.

### 2. Model Yapısı (Model Architecture)
*   **Base Model:** ResNet-50 (Weights='imagenet', include_top=False)
*   **Eklenen Katmanlar:**
    *   `GlobalAveragePooling2D`: Özellik haritalarını vektöre dönüştürmek için.
    *   `Dense (512, ReLU)`: Modelin spesifik özellikleri öğrenmesi için.
    *   `Dropout (0.5)`: Aşırı öğrenmeyi (Overfitting) engellemek için.
    *   `Dense (10, Softmax)`: 10 sanatçı sınıfı için olasılıksal çıktı üretmek için.

### 3. Sınıf Dengesizliği ile Mücadele
Veri setindeki dengesizlikleri (Imbalance) yönetmek amacıyla eğitim sırasında **Class Weights (Sınıf Ağırlıklandırma)** yöntemi uygulanmıştır. Bu sayede, az sayıda eseri olan sanatçıların (örn. Michelangelo) model tarafından göz ardı edilmesi engellenmiştir.

## 📊 Sonuçlar

Model eğitimi Google Colab üzerinde GPU hızlandırma kullanılarak gerçekleştirilmiştir.
*   **Eğitim Başarısı (Train Accuracy):** ~%90
*   **Doğrulama Başarısı (Val Accuracy):** ~%83
*   **Test Başarısı:** ~%86

Grad-CAM analizleri, modelin sadece renk dağılımına değil, fırça darbelerine (Brushstrokes) ve kompozisyona odaklandığını göstermiştir.

## 🛠️ Kurulum ve Çalıştırma

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

### Gereksinimler
*   Python 3.8 veya üzeri
*   TensorFlow 2.x
*   Gradio

### Adım 1: Projeyi Klonlayın
```bash
git clone [https://github.com/muserrefselcukozdemir/brushinsight.git](https://github.com/muserrefselcukozdemir/brushinsight.git)
cd brushinsight
```
### Adım 2: Sanal Ortam Oluşturun

```bash
# Linux / Mac
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```
### Adım 3: Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### Adım 4: Uygulamayı Başlatın
Uygulama dosyaları app/ klasörü altındadır. Modeli çalıştırmak için:

```bash
python app/app.py
```
### 📂 Dosya Yapısı

BrushInsight/
├── app/
│   ├── app.py                # Gradio arayüz kodu
│   ├── artist_model.keras    # Eğitilmiş AI modeli
│   └── class_indices.json    # Sanatçı etiketleri
├── notebooks/
│   └── BrushInsight_Training.ipynb  # Eğitim kodları (Colab)
├── models/                   # Model yedekleri
├── requirements.txt          # Kütüphane listesi
└── README.md                 # Proje dökümantasyonu

### 👥 Emeği Geçenler
Bu proje, Müşerref Selçuk Özdemir tarafından hazırlanmıştır.

## Github

[https://github.com/muserrefselcukozdemir/brushinsight](https://github.com/muserrefselcukozdemir/brushinsight)