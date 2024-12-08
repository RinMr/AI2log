import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle

# 画像サイズとデータセットのディレクトリパス
folder = 'C:\\Users\\user\\Desktop\\GitHub\\allimages\\allimages2'  # 適切なパスに変更してください
IMG_SIZE = 128  # 画像サイズ

# 画像を読み込む関数
def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0  # 正規化
            images.append(img)
            label = filename.split('_')[0]  # ファイル名の先頭部分をラベルとする
            labels.append(label)
    return np.array(images), labels

# モデルを作成する関数
def create_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # 2クラス分類（猫と犬）
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 画像の読み込みと前処理
images, labels = load_images_from_folder(folder)

# ラベルのエンコード
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
categorical_labels = to_categorical(encoded_labels)

# 訓練データと検証データに分割
train_images, val_images, train_labels, val_labels = train_test_split(images, categorical_labels, test_size=0.2, random_state=42)

# モデル作成
model = create_model((IMG_SIZE, IMG_SIZE, 3))

# データ拡張設定
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

# モデル訓練
history = model.fit(datagen.flow(train_images, train_labels, batch_size=32), epochs=10, validation_data=(val_images, val_labels))

# モデル保存
model_path = 'models/my_trained_model.keras'
model.save(model_path)
print(f"モデルを保存しました: {model_path}")

# ラベルエンコーダー保存
encoder_path = 'models/label_encoder.pkl'
with open(encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"ラベルエンコーダーを保存しました: {encoder_path}")

# 特徴量のキャッシュ作成
def extract_features(image):
    return image.flatten()  # ここでは簡単にフラット化して返す

def generate_feature_cache(images, labels):
    feature_cache = {}
    for img, label in zip(images, labels):
        label = str(label)  # ラベルを文字列に変換
        features = extract_features(img)
        if label not in feature_cache:
            feature_cache[label] = features
    return feature_cache

# 特徴量キャッシュを生成して保存
feature_cache = generate_feature_cache(images, encoded_labels)
cache_path = 'models/feature_cache.pkl'
with open(cache_path, 'wb') as f:
    pickle.dump(feature_cache, f)
print(f"特徴量キャッシュを保存しました: {cache_path}")
