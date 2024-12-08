import os

# 対象ディレクトリを指定
target_directory = 'C:\\Users\\user\\Desktop\\GitHub\\Emotion_archive\\dog_Emotion\\aa'
new_base_name = "angry-dog"
file_extension = ".jpg"

counter = 976
for filename in os.listdir(target_directory):
    old_path = os.path.join(target_directory, filename)
    if os.path.isfile(old_path) and filename.lower().endswith(file_extension):
        new_filename = f"{new_base_name}{counter}{file_extension}"
        new_path = os.path.join(target_directory, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} -> {new_path}")
        counter += 1
