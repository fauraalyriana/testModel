from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

# Inisialisasi Flask
app = Flask(__name__)

# Load model .h5
MODEL_PATH = 'mango_classifier_model.h5'  # Pastikan path file model sesuai
model = tf.keras.models.load_model(MODEL_PATH)

# Definisi kelas label (contoh: mature dan immature)
CLASS_LABELS = ['Immature: wait for a few days', 'Mature: ready to eat']

# Endpoint untuk homepage atau test
@app.route('/')
def home():
    return "Fruit Maturity Detection API is running!"

# Endpoint untuk deteksi kematangan buah
@app.route('/predict', methods=['POST'])
def detect():
    try:
        # Pastikan request berupa file gambar
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']

        # Baca gambar dan ubah ke format yang sesuai
        image = tf.keras.preprocessing.image.load_img(file, target_size=(224, 224))  # Resize ke 224x224
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0) / 255.0  # Normalisasi

        # Prediksi menggunakan model
        predictions = model.predict(image)
        class_index = np.argmax(predictions[0])  # Ambil indeks prediksi tertinggi
        result = CLASS_LABELS[class_index]

        # Return hasil prediksi
        return jsonify({
            "status": "success",
            "result": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Menjalankan Flask di host 0.0.0.0 agar bisa diakses dari eksternal
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
