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
import matplotlib.pyplot as plt
import time
import pickle

app = Flask(__name__)

history_log = []
old_history_log = []
MAX_HISTORY_SIZE = 4

# モデルとエンコーダの読み込み
def load_models():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'models/my_trained_model.h5')
    encoder_path = os.path.join(base_path, 'models/label_encoder.pkl')
    cache_path = os.path.join(base_path, 'models/feature_cache.pkl')  # 拡張子を修正

    if os.path.exists(model_path) and os.path.exists(encoder_path) and os.path.exists(cache_path):
        print("既存のモデルとエンコーダをロードしています...")
        label_encoder = pickle.load(open(encoder_path, 'rb'))
        model = load_model(model_path)
        feature_cache = pickle.load(open(cache_path, 'rb'))  # 修正された拡張子を使用
    else:
        print("モデルが見つからないため、トレーニングを実行しています...")
        folder = 'C:\\Users\\223204\\Desktop\\all\\allimages'
        images, labels = load_images_from_folder(folder)
        processed_images = preprocess_images(images)

        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        categorical_labels = to_categorical(encoded_labels)

        train_images, val_images, train_labels, val_labels = train_test_split(
            processed_images, categorical_labels, test_size=0.2, random_state=42
        )

        model, history = train_model(train_images, train_labels, val_images, val_labels)

        # トレーニング済みモデルとエンコーダを保存
        model.save(model_path)
        pickle.dump(label_encoder, open(encoder_path, 'wb'))
        feature_cache = generate_feature_cache(processed_images, encoded_labels)
        pickle.dump(feature_cache, open(cache_path, 'wb'))  # 修正された拡張子を使用

    return label_encoder, model, feature_cache

def generate_feature_cache(images, labels):
    feature_cache = {}
    for img, label in zip(images, labels):
        label = str(label)
        features = extract_features(img)
        if label not in feature_cache:
            feature_cache[label] = features
    return feature_cache

# モデルとエンコーダのロードをアプリケーション起動時に実行
label_encoder, model, feature_cache = load_models()

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            label = filename.split('_')[0]
            labels.append(label)
    return images, labels

def preprocess_images(images):
    IMG_SIZE = 128
    processed_images = []
    for img in images:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        processed_images.append(img)
    return np.array(processed_images)

def train_model(train_images, train_labels, val_images, val_labels):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(np.unique(train_labels.argmax(axis=1))), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    history = model.fit(datagen.flow(train_images, train_labels, batch_size=32), epochs=1,
                        validation_data=(val_images, val_labels))
    return model, history

def get_related_images(predicted_label, feature_cache, folder='C:\\Users\\223204\\Desktop\\all\\allimages'):
    related_images = []
    predicted_features = feature_cache.get(predicted_label)  # 特徴量の取得

    if predicted_features is not None:
        # 正規化
        predicted_features = predicted_features / np.linalg.norm(predicted_features)

        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            if os.path.isfile(img_path):  # ファイルが存在するか確認
                img = cv2.imread(img_path)
                if img is not None:  # 正常に読み込まれた場合のみ処理
                    img_resized = cv2.resize(img, (128, 128))
                    img_normalized = img_resized / 255.0
                    img_features = extract_features(img_normalized)
                    img_features = img_features / np.linalg.norm(img_features)  # 正規化

                    # 類似度計算
                    similarity = np.dot(predicted_features, img_features)
                    if similarity > 0.8:  # 類似度が高い画像を追加
                        related_images.append(img_path)

    return related_images

# 画像の特徴量を抽出する関数
def extract_features(image):
    # 実際には事前学習したネットワークなどで特徴量を抽出
    # ここでは簡単にフラット化して返します
    return image.flatten()

@app.route('/history')
def show_history():
    # history.htmlに履歴を渡して表示する
    return render_template('history.html', history=history_log, zip=zip)

def predict_image(image_path, model, label_encoder, feature_cache):
    IMG_SIZE = 128
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)

    prediction = model.predict(img_expanded)[0]
    all_labels = label_encoder.inverse_transform(np.arange(len(prediction)))  # ラベルを取得
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

    # 関連画像を取得
    related_images = get_related_images(top_label, feature_cache)

    # 関連画像をBase64形式に変換
    related_images_base64 = []
    valid_paths = []
    for related_image_path in related_images:
        related_img = cv2.imread(related_image_path)
        if related_img is not None:  # 読み込み成功した画像のみ追加
            _, encoded_img = cv2.imencode('.png', related_img)
            related_images_base64.append(base64.b64encode(encoded_img).decode('utf-8'))
            valid_paths.append(related_image_path)

    if not related_images_base64:
        message += " 関連画像は見つかりませんでした。"

    # 履歴の更新
    global history_log, old_history_log
    history_log.append({
        'labels': all_labels.tolist(),
        'probs': all_probs.tolist(),
        'message': message,
        'image_path': os.path.basename(image_path)
    })

    if len(history_log) > MAX_HISTORY_SIZE:
        old_history_log.append(history_log.pop(0))

    # `related_images_base64` と有効な画像パスをペアで返す
    return all_labels, all_probs, img, message, list(zip(related_images_base64, valid_paths))

@app.route('/old_history')
def show_old_history():
    return render_template('old_history.html', history=old_history_log, zip=zip)

folder = 'C:\\Users\\223204\\Desktop\\all\\allimages'
images, labels = load_images_from_folder(folder)
processed_images = preprocess_images(images)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
categorical_labels = to_categorical(encoded_labels)

train_images, val_images, train_labels, val_labels = train_test_split(
    processed_images, categorical_labels, test_size=0.2, random_state=42)

model, history = train_model(train_images, train_labels, val_images, val_labels)

if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.template_filter('zip')
def zip_filter(a, b):
    return zip(a, b)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        uploaded_image_path = os.path.join('uploads', f.filename)
        f.save(uploaded_image_path)

        all_labels, all_probs, img, message, related_images = predict_image(uploaded_image_path, model, label_encoder, feature_cache)

        _, img_encoded = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        # 関連画像をBase64形式に変換
        related_images_base64 = []
        for related_image in related_images:
            related_img = cv2.imread(related_image)
            _, encoded_img = cv2.imencode('.png', related_img)
            related_images_base64.append(base64.b64encode(encoded_img).decode('utf-8'))

        return render_template(
            'decision.html',
            predictions=zip(all_labels, all_probs),
            message=message,
            image=f'<img src="data:image/png;base64,{img_base64}" alt="予測画像"/>',
            related_images=zip(related_images_base64, related_images),
            history=history_log
        )

    return render_template('home.html', history=history_log, timestamp=int(time.time()), zip=zip)

@app.route('/home')
def show_home():
    # ホーム画面を表示
    return render_template('home.html', history=history_log, timestamp=int(time.time()), zip=zip)

if __name__ == '__main__':
    app.run(debug=True)