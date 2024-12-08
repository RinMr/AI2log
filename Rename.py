import os

def rename_files(directory, old_part, new_part):
    for filename in os.listdir(directory):
        if old_part in filename:
            new_filename = filename.replace(old_part, new_part)
            old_filepath = os.path.join(directory, filename)
            new_filepath = os.path.join(directory, new_filename)
            os.rename(old_filepath, new_filepath)
            print(f'Renamed: {filename} -> {new_filename}')

# 使用例
directory_path = 'C:\\Users\\user\\Desktop\\GitHub\\Emotion_archive\\dog_Emotion\\angry'
rename_files(directory_path, 'angry-dog_', 'angry-dog')
