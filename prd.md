PROJE PRD: BrushInsight (AI-Powered Art Curator)1. Proje Amacı ve KapsamıBu proje, Fırat Üniversitesi Prof. Dr. Sami Ekici'nin "Evrişimsel Sinir Ağları" dersi kapsamında hazırlanan bir yüksek lisans final çalışmasıdır1. Temel amaç, Transfer Learning (Transfer Öğrenme) yöntemiyle dünyanın en etkili 10 ressamının eserlerini sınıflandırmak ve CNN mimarisinin çalışma mantığını (açıklanabilirlik) interaktif bir arayüzle sunmaktır2.2. Veri Seti DetaylarıKaynak: Kaggle - "Best Artworks of All Time"3.Seçim: En çok eseri olan ilk 10 sanatçı (Filtreleme kodla yapılacaktır).Dosya: Hesaplama maliyetini düşürmek için resized.zip kullanılacaktır4444.Kısıt: Veri artırma (Augmentation) uygulanmayacaktır.3. Teknik Mimari (TensorFlow/Keras)Model: ResNet-50 tabanlı mimari.Derin ağlardaki "Vanishing Gradient" (kaybolan gradyan) problemini aşmak için "Skip Connections" (atlamalı bağlantılar) içeren yapı kullanılacaktır5555.Katman Yapısı:Giriş: 224x224x3 boyutunda görüntüler6.Ara Katmanlar: Slide'larda belirtilen ReLU aktivasyon fonksiyonu ($y = max(0, x)$)7777.Çıkış: 10 nöronlu tam bağlantılı (FC) katman ve olasılıksal tahmin için Softmax aktivasyonu8888.Parametre Yönetimi: CNN'in avantajı olan Parameter Sharing (parametre paylaşımı) ve Sparsity of Connections (bağlantı seyreklik) prensipleri sunumda vurgulanacaktır9.4. Dosya ve Klasör YapısıProje klasörü şu şekilde organize edilmelidir:PlaintextBrushInsight/
├── .venv/                  # Yerel sanal ortam
├── data/                   # Veri çekme ve filtreleme scripti
├── models/                 # Colab'den gelecek 'artist_model.h5'
├── notebooks/              # Google Colab eğitim kodları (ipynb)
├── app/                    # Gradio ve Hugging Face kodları (app.py)
├── requirements.txt        # tensorflow, gradio, matplotlib, opencv-python, kaggle
└── README.md               # Akademik açıklama ve kurulum rehberi
5. Uygulama ve Arayüz (Gradio)Hugging Face Spaces üzerinde çalışacak arayüz şu iki bileşeni içermelidir:Tahmin Paneli: Sanatçı adı ve başarı yüzdesi.Eğitim Köşesi (Slide Entegrasyonu): Kullanıcıya aşağıdaki teknik detayları açıklayan basit notlar:Isı Haritası (Grad-CAM): Modelin resimdeki hangi piksellere/özelliklere odaklandığını gösteren görsel10101010.Aktivasyon Notu: ReLU'nun neden seçildiği (Hızlı yakınsama ve seyrek aktivasyon)11111111.Overfitting Uyarısı: Tam bağlantılı katmanların neden overfitting'e meyilli olduğu açıklaması12.6. İş Akışı (Workflow)Yerel: .venv oluşturulacak ve requirements.txt ile kütüphaneler kurulacak.Colab: Kaggle API üzerinden veri çekilecek, filtreleme yapılacak ve model eğitilip .h5 formatında kaydedilecek.Hugging Face: Eğitilen model ve app.py yüklenerek interaktif arayüz yayına alınacak.Kurulum İçin İlk Adım (Yerel Terminal Komutları)Yerel bilgisayarında şu komutlarla projeyi başlatabilirsin:Bash# Proje klasörü oluştur
mkdir BrushInsight
cd BrushInsight

# Sanal ortam oluştur ve aktif et
python -m venv .venv
# Windows için: .venv\Scripts\activate
# Mac/Linux için: source .venv/bin/activate

# Gerekli dosyaları oluştur
touch requirements.txt README.md
mkdir data models notebooks app
requirements.txt içeriği:Plaintexttensorflow
gradio
matplotlib
opencv-python
pandas
kaggle
Kral, bu PRD ile yerel yapay zekaya "Bana bu PRD'ye uygun Python kodlarını yazar mısın?" dediğinde sana nokta atışı kodları verecektir. Hazırsan ilk kodları (Veri filtreleme veya Colab eğitim kodu) yazdırmaya başlayabiliriz. Hangisinden başlayalım?