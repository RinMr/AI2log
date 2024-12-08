from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import base64
import pickle
import time

app = Flask(__name__)

# 初期化
history_log = []
old_history_log = []
MAX_HISTORY_SIZE = 4
CLASS_LABELS = ['Cat: Angry', 'Dog: Angry']

# モデルとエンコーダの読み込み
def load_models():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'models/my_trained_model.keras')
    encoder_path = os.path.join(base_path, 'models/label_encoder.pkl')
    emotion_model_path = os.path.join(base_path, 'models/emotion_model.keras')

    model = load_model(model_path)
    label_encoder = pickle.load(open(encoder_path, 'rb'))
    emotion_model = load_model(emotion_model_path)
    return model, label_encoder, emotion_model

model, label_encoder, emotion_model = load_models()

# 画像前処理
def preprocess_image(image_path):
    IMG_SIZE = 128
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    return np.expand_dims(img_normalized, axis=0)

def predict_image(image_path):
    global history_log, old_history_log

    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)[0]
    emotion_predictions = emotion_model.predict(processed_image)[0]

    all_labels = label_encoder.inverse_transform(np.argsort(predictions)[::-1])
    all_probs = np.sort(predictions)[::-1]

    # Prepare the sorted list of (label, probability) pairs
    predictions_list = [(label, prob) for label, prob in zip(all_labels, all_probs)]

    # Emotion predictions remain the same as before
    emotion_label = 'angry'  # 固定された感情
    emotion_confidence = float(emotion_predictions[0]) * 100  # Convert to float

    emotion_details = {emotion_label: emotion_confidence}

    # Prediction for cat or dog (for the message part)
    IMG_SIZE = 128
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)

    prediction = model.predict(img_expanded)[0]
    all_labels = label_encoder.inverse_transform(np.arange(len(prediction)))  # 全てのラベル
    all_probs = prediction * 100  # 確率を百分率に変換

    top_index = np.argmax(prediction)
    top_label = all_labels[top_index]
    top_prob = all_probs[top_index]

    if 'cat' in top_label and top_prob >= 50:
        message = f"これは猫です。"
    elif 'dog' in top_label and top_prob >= 50:
        message = f"これは犬です。"
    else:
        message = "これは猫でも犬でもありません。"

    # 履歴の更新
    history_log.append({
        'labels': all_labels.tolist(),
        'probs': all_probs.tolist(),
        'message': message,
        'image_path': os.path.basename(image_path)
    })

    if len(history_log) > MAX_HISTORY_SIZE:
        old_history_log.append(history_log.pop(0))

    return predictions_list, message, emotion_details, top_label, top_prob  # 5つの値を返す

@app.route('/old_history')
def show_old_history():
    return render_template('old_history.html', history=old_history_log, zip=zip)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.template_filter('zip')
def zip_filter(a, b):
    return zip(a, b)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)

        confidence, label,predictions_list, message, emotion_details = predict_image(image_path)

        all_labels, all_probs = predict_image(image_path)[0:2]

        _, img_encoded = cv2.imencode('.png', cv2.imread(image_path))
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        return render_template(
            'decision.html',
            message=message,
            image=f'<img src="data:image/png;base64,{img_base64}" alt="予測画像" />',
            emotion_details=emotion_details,
        )
    return render_template('home.html')

@app.route('/home')
def show_home():
    # ホーム画面を表示
    return render_template('home.html', history=history_log, timestamp=int(time.time()), zip=zip)
@app.route('/history')
def show_history():
    return render_template('history.html', history=history_log, timestamp=int(time.time()), zip=zip)

if __name__ == '__main__':
    app.run(debug=True)