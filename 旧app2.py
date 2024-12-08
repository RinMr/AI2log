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
import sys
import locale

# 標準出力エンコーディングを設定
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# ロケールの設定確認
print(f"[DEBUG] ロケール設定: {locale.getpreferredencoding()}")

import logging

# ログ設定
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),  # コンソール出力
        logging.FileHandler("app_debug.log", encoding="utf-8"),  # ファイル出力
    ],
)

# デバッグ用ログ関数
def debug_log(message):
    logging.debug(message)

app = Flask(__name__)

history_log = []
old_history_log = []
MAX_HISTORY_SIZE = 4
CLASS_LABELS = ['Cat: Angry', 'Dog: Angry']

# モデルとエンコーダの読み込み
def debug_log(message):
    print(f"[DEBUG] {message}")

# モデルとエンコーダの読み込み
def load_models():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'models/my_trained_model.keras')
    encoder_path = os.path.join(base_path, 'models/label_encoder.pkl')
    cache_path = os.path.join(base_path, 'models/feature_cache.pkl')
    emotion_model = load_model('models/emotion_model.keras')  # Emotionモデルを読み込む

    if os.path.exists(model_path) and os.path.exists(encoder_path) and os.path.exists(cache_path):
        debug_log("既存のモデルとエンコーダをロードしています...")
        label_encoder = pickle.load(open(encoder_path, 'rb'))
        model = load_model(model_path)
        feature_cache = pickle.load(open(cache_path, 'rb'))  # 修正された拡張子を使用
    else:
        debug_log("モデルが見つからないため、トレーニングを実行しています...")
        folder = 'C:\\Users\\user\\Desktop\\GitHub\\allimages\\allimages2'
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

    debug_log("モデルとエンコーダのロード完了")
    return label_encoder, model, feature_cache, emotion_model  # emotion_modelを返す

# モデルとエンコーダのロードをアプリケーション起動時に実行
label_encoder, model, feature_cache, emotion_model = load_models()  # emotion_modelも受け取る

def generate_feature_cache(images, labels):
    feature_cache = {}
    for img, label in zip(images, labels):
        label = str(label)
        features = extract_features(img)
        if label not in feature_cache:
            feature_cache[label] = features
            debug_log(f"キャッシュ生成: ラベル {label} の特徴量を生成しました")
    return feature_cache

# モデルとエンコーダのロードをアプリケーション起動時に実行
label_encoder, model, feature_cache, emotion_model = load_models()

def load_images_from_folder(folder):
    images = []
    labels = []
    debug_log(f"フォルダから画像を読み込み中: {folder}")
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            label = filename.split('_')[0]
            labels.append(label)
    debug_log(f"画像読み込み完了: {len(images)} 枚の画像を処理")
    return images, labels

def preprocess_images(images):
    IMG_SIZE = 128
    debug_log("画像の前処理を開始")
    processed_images = []
    for img in images:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # 画像のリサイズ
        img = img / 255.0  # 正規化
        processed_images.append(img)
    debug_log(f"画像の前処理完了: {len(processed_images)} 枚")
    return np.array(processed_images)  # ここで (num_samples, 128, 128, 3) の4次元配列を返す

def train_model(train_images, train_labels, val_images, val_labels):
    debug_log("モデルのトレーニングを開始")
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

    # データジェネレータを使って学習
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    # 画像データの形状を確認 (バッチ次元は自動的に追加される)
    train_images = np.array(train_images)  # 形状を (num_samples, 128, 128, 3) に
    val_images = np.array(val_images)  # 同様に

    # データジェネレータを使ってトレーニング
    history = model.fit(datagen.flow(train_images, train_labels, batch_size=32),
                        epochs=1, validation_data=(val_images, val_labels))
    debug_log("モデルのトレーニングが完了")
    return model, history

def get_related_images(predicted_label, feature_cache, folder='C:\\Users\\user\\Desktop\\GitHub\\allimages\\allimages2'):
    related_images = []
    print(f"[DEBUG] 予測ラベル: {predicted_label}")
    print(f"[DEBUG] 指定フォルダ: {folder}")
    
    # キャッシュされた特徴量を取得
    predicted_features = feature_cache.get(predicted_label)
    if predicted_features is not None:
        print(f"[DEBUG] キャッシュから取得した特徴量の長さ: {len(predicted_features)}")
        predicted_features = predicted_features / np.linalg.norm(predicted_features)
    else:
        print(f"[ERROR] キャッシュに予測ラベル '{predicted_label}' が見つかりませんでした。")
        return related_images

    # フォルダ内の画像をループ処理
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        print(f"[DEBUG] 処理中のファイル: {img_path}")
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARNING] 画像の読み込みに失敗しました: {filename}")
                continue
            
            # 画像の前処理
            try:
                img_resized = cv2.resize(img, (128, 128))
                img_normalized = img_resized / 255.0
                img_features = extract_features(img_normalized)
                img_features = img_features / np.linalg.norm(img_features)
            except Exception as e:
                print(f"[ERROR] 画像特徴量の抽出に失敗しました: {filename}. エラー: {e}")
                continue
            
            # 類似度計算
            similarity = np.dot(predicted_features, img_features)
            print(f"[DEBUG] 類似度: {similarity:.4f} (画像: {filename})")

            if similarity > 0.1:
                print(f"[INFO] 関連画像として追加: {filename}")
                related_images.append(img_path)
        else:
            print(f"[WARNING] ファイルが見つかりません: {img_path}")
    
    if not related_images:
        print("[INFO] 関連画像が見つかりませんでした。")
    else:
        print(f"[INFO] 関連画像数: {len(related_images)}")

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

# emotion_modelを使用して感情を予測する処理を追加

def predict_image(image_path, model, label_encoder, feature_cache, emotion_model):
    IMG_SIZE = 128
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0

    # バッチ次元を追加して4次元にする
    img_expanded = np.expand_dims(img_normalized, axis=0)  # shape: (1, IMG_SIZE, IMG_SIZE, 3)

    # 予測
    prediction = model.predict(img_expanded)[0]  # 予測結果を取得
    all_labels = label_encoder.inverse_transform(np.arange(len(prediction)))  # ラベルを取得
    all_probs = prediction * 100  # 確率を百分率に変換

    # 最も確率が高いラベルとその確率を取得
    top_index = np.argmax(prediction)
    top_label = all_labels[top_index]
    top_prob = all_probs[top_index]

   # Emotionに基づく予測結果
    emotion_prediction = emotion_model.predict(img_expanded)  # モデルの予測
    emotion_labels = ['angry']  # 感情ラベルが1つしかない場合
    emotion_label = emotion_labels[0]  # ラベルを直接指定
    emotion_prob = float(emotion_prediction[0][0]) * 100  # 確率を百分率に変換

    emotion_details = {emotion_label: emotion_prediction[0][0]}  # 値を float に変換
    return render_template(
        'result.html',
        message="感情予測の結果",
        emotion=emotion_label,
        emotion_details=emotion_details,
        image=image_base64
    )


    # メッセージを生成
    if 'Cat' in top_label and top_prob >= 50:
        message = f"これは猫です。"
    elif 'Dog' in top_label and top_prob >= 50:
        message = f"これは犬です。"
    else:
        message = "これは猫でも犬でもありません。"

    # Emotionに基づく予測結果
    emotion_details = {
        'filename': os.path.basename(image_path),
        'prediction': top_label,
        'confidence': top_prob,  # 確信度
        'emotion': emotion_label,  # 感情ラベル
        'emotion_confidence': emotion_prob  # 感情の確信度
    }

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

    # related_images_base64 と有効な画像パスをペアで返す
    return all_labels, all_probs, img, message, related_images, emotion_details

@app.route('/old_history')
def show_old_history():
    return render_template('old_history.html', history=old_history_log, zip=zip)

folder = 'C:\\Users\\user\\Desktop\\GitHub\\allimages\\allimages2'
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

        # 予測結果を取得
        all_labels, all_probs, img, message, related_images, emotion_details = predict_image(
            uploaded_image_path, model, label_encoder, feature_cache, emotion_model  # 5つを受け取る
        )

        # 画像をBase64エンコード
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
            related_images=zip(related_images_base64, related_images),  # ここで関連画像をテンプレートに渡す
            history=history_log,
            emotion_details=emotion_details  # emotion_detailsもテンプレートに渡す
        )

    return render_template('home.html', history=history_log, timestamp=int(time.time()), zip=zip)

@app.route('/home')
def show_home():
    # ホーム画面を表示
    return render_template('home.html', history=history_log, timestamp=int(time.time()), zip=zip)

if __name__ == '__main__':
    app.run(debug=True)