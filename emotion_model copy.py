import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# フォルダ構造のパス
BASE_DIR = "C:\\Users\\user\\Desktop\\AI2.ver8.5(log)\\emotion"  # emotionフォルダのパスに変更
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VALIDATION_DIR = os.path.join(BASE_DIR, "validation")

# パラメータ
IMG_SIZE = 128  # 画像サイズ
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 5

# モデル構造
def create_model(input_shape, num_classes):
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
        Dense(num_classes, activation='softmax')  # 出力ノード数はクラス数
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# データ拡張とデータジェネレータ
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'  # クラスのカテゴリカルエンコーディング
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# モデルの作成
model = create_model((IMG_SIZE, IMG_SIZE, 3), NUM_CLASSES)

# モデル保存（Keras形式で保存）
checkpoint = ModelCheckpoint(
    filepath='models/emotion_model.keras',  # 拡張子を .keras に変更
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# モデル訓練
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    callbacks=[checkpoint]
)

print("トレーニング完了！モデルが保存されました。")
