# 🎨 BrushInsight: Yapay Zeka Destekli Sanat Küratörü

![Durum](https://img.shields.io/badge/Durum-Tamamlandı-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%202.x-orange)
![Gradio](https://img.shields.io/badge/Arayüz-Gradio-yellow)
![Lisans](https://img.shields.io/badge/Lisans-MIT-green)

> **Canlı Demo:** [Hugging Face Spaces - BrushInsight](https://huggingface.co/spaces/muserrefselcukozdemir/brushinsight)

## 📖 Proje Hakkında

Bu çalışma, Fırat Üniversitesi Prof. Dr. Sami Ekici'nin "Evrişimsel Sinir Ağları" dersi kapsamında hazırlanan bir yüksek lisans final projesidir.

**BrushInsight**, derin öğrenme yöntemlerinden **Transfer Learning** tekniğini kullanarak, sanat tarihi literatüründe en etkili kabul edilen 10 ressamın eserlerini sınıflandırmayı amaçlar. Proje, sadece bir sınıflandırma modeli sunmakla kalmaz; **Açıklanabilir Yapay Zeka (XAI)** prensipleri doğrultusunda **Grad-CAM** tekniği ile modelin karar mekanizmasını görselleştirir ve "Kara Kutu" problemini aşmayı hedefler.

### 🎯 Temel Amaçlar
1.  **Yüksek Başarı:** ResNet-50 mimarisi ile sanat eseri tanıma görevinde yüksek doğruluk elde etmek.
2.  **Açıklanabilirlik:** Modelin bir eseri neden belirli bir ressama ait olarak sınıflandırdığını ısı haritaları ile göstermek.
3.  **Erişilebilirlik:** Gradio tabanlı kullanıcı dostu bir arayüz ile modelin herkes tarafından deneyimlenmesini sağlamak.

## 🧠 Teknik Yaklaşım

### 1. Veri Seti
*   **Kaynak:** Kaggle - "Best Artworks of All Time"
*   **Kapsam:** En çok esere sahip ilk 10 sanatçı (Van Gogh, Picasso, Monet vb.)
*   **İşleme:** Görüntüler 224x224 piksel boyutuna getirilmiş ve normalize edilmiştir.

### 2. Model Yapısı
*   **Temel Model:** ResNet-50 (ImageNet ağırlıkları)
*   **Ek Katmanlar:**
    *   `GlobalAveragePooling2D`: Özellik haritalarını vektöre dönüştürmek için.
    *   `Dense (512, ReLU)`: Spesifik özellikleri öğrenmek için.
    *   `Dropout (0.5)`: Aşırı öğrenmeyi (Overfitting) engellemek için.
    *   `Dense (10, Softmax)`: 10 sanatçı sınıfı için çıktı üretmek için.

### 3. Eğitim Detayları
Veri setindeki dengesizlikleri yönetmek amacıyla eğitim sırasında **Class Weights (Sınıf Ağırlıklandırma)** yöntemi uygulanmıştır. Böylece az sayıda eseri olan sanatçıların model tarafından göz ardı edilmesi engellenmiştir.

## 📊 Sonuçlar

Model eğitimi Google Colab üzerinde GPU hızlandırma kullanılarak gerçekleştirilmiştir.

*   **Eğitim Başarısı:** ~%90
*   **Doğrulama Başarısı:** ~%83
*   **Test Başarısı:** ~%86

Grad-CAM analizleri, modelin karar verirken eserlerin genel kompozisyonuna ve fırça darbelerine odaklandığını göstermektedir.

## 🛠️ Kurulum ve Çalıştırma

### Gereksinimler
*   Python 3.8 veya üzeri
*   TensorFlow 2.x
*   Gradio

### Adımlar

1. Projeyi klonlayın:
git clone https://github.com/muserrefselcukozdemir/brushinsight.git
cd brushinsight

2. Sanal ortam oluşturun:
# Linux / Mac
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate

3. Bağımlılıkları yükleyin:
pip install -r requirements.txt

4. Uygulamayı başlatın:
python app/app.py

## 📂 Dosya Yapısı
```bash
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
```
## 👥 Emeği Geçenler

Bu proje **Müşerref Selçuk Özdemir** tarafından hazırlanmıştır.

## 🔗 Bağlantılar

*   [GitHub Reposu](https://github.com/muserrefselcukozdemir/brushinsight)
*   [Hugging Face Demo](https://huggingface.co/spaces/muserrefselcukozdemir/brushinsight)
