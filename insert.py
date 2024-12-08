import os

# 対象ディレクトリを指定
target_directory = "C:\\Users\\user\\Desktop\\AI2.ver8.5(log)\\emotions_detaset\\cat\\angry"

for filename in os.listdir(target_directory):
    old_path = os.path.join(target_directory, filename)
    if os.path.isfile(old_path):
        new_filename = filename.replace("(", "").replace(")", "") # ()削除
        new_filename = filename.replace(" ", "")                  # 空白削除
        new_path = os.path.join(target_directory, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} -> {new_path}")
