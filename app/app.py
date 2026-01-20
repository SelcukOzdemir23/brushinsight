import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import json
import os

# --- Ayarlar ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "artist_model.keras" 
CLASS_INDICES_PATH = "class_indices.json"
IMG_SIZE = (224, 224)

# --- Model ve Sınıf Yükleme ---
def load_resources():
    model, labels = None, []
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model başarıyla yüklendi!")
    except Exception as e:
        print(f"❌ HATA: Model bulunamadı! ({e})")
        
    try:
        with open(CLASS_INDICES_PATH, 'r') as f:
            indices = json.load(f)
            labels = [k for k, v in sorted(indices.items(), key=lambda item: item[1])]
        print(f"✅ Sınıflar yüklendi: {labels}")
    except Exception as e:
        print(f"⚠️ UYARI: Sınıf listesi bulunamadı!")
    
    return model, labels

model, LABELS = load_resources()

# --- Grad-CAM Fonksiyonu (Hata Giderilmiş) ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        # HATA ÇÖZÜMÜ: preds eğer listeyse tensora çeviriyoruz
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
            
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        
        # Slicing hatasını engellemek için tf.gather veya tensor indeksi kullanıyoruz
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    img_h, img_w = img.shape[:2]
    heatmap = cv2.resize(heatmap, (img_w, img_h))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    superimposed_img = heatmap * alpha + img * (1 - alpha)
    return np.clip(superimposed_img, 0, 255).astype("uint8")

# --- Tahmin Fonksiyonu ---
def predict_fn(image):
    if model is None or image is None:
        return None, None, "Hata oluştu!"

    img_resized = cv2.resize(image, IMG_SIZE)
    img_array = np.expand_dims(img_resized, axis=0)
    img_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_array.astype(np.float32))

    preds = model.predict(img_preprocessed)
    top_index = np.argmax(preds[0])
    
    confidences = {LABELS[i]: float(preds[0][i]) for i in range(len(LABELS))}
    predicted_label = LABELS[top_index]
    
    # Isı haritası oluşturma
    heatmap = make_gradcam_heatmap(img_preprocessed, model, "conv5_block3_out", top_index)
    gradcam_img = overlay_heatmap(image, heatmap)
    
    return confidences, gradcam_img, f"Sanatçı Tahmini: {predicted_label}"

# --- Gradio Arayüzü ---
# theme parametresi launch içine taşındı
with gr.Blocks(title="BrushInsight: Sanat Küratörü") as demo:
    gr.Markdown("# 🎨 BrushInsight: AI-Powered Art Curator")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="Sanat Eseri Yükleyin", type="numpy")
            btn = gr.Button("Analiz Et ✨", variant="primary")
            output_label = gr.Label(num_top_classes=3, label="Tahmin Olasılıkları")
            
        with gr.Column():
            gradcam_img = gr.Image(label="Grad-CAM Odak Analizi")
            info_text = gr.Textbox(label="Sonuç", interactive=False)

    btn.click(predict_fn, inputs=input_img, outputs=[output_label, gradcam_img, info_text])

if __name__ == "__main__":
    # Gradio 6.0 uyumluluğu için theme burada
    demo.launch(theme=gr.themes.Soft())