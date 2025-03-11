import os
import cv2
import numpy as np
import albumentations as A

# 폴더 경로 설정
base_path = r"C:\Users\souok\Desktop\Aug"
source_folders = ["Fist_LH", "Fist_RH", "Open_LH", "Open_RH"]
target_folders = ["Aug_Fist", "Aug_Fist", "Aug_Open", "Aug_Open"]

transform = A.Compose([
    A.HorizontalFlip(p=1.0),
    A.Rotate(limit=10, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.1, p=0.7),
    A.GaussNoise(var_limit=(10.0, 30.0), p=0.5),
])

# 증강 실행
for src_folder, tgt_folder in zip(source_folders, target_folders):
    src_path = os.path.join(base_path, src_folder)
    tgt_path = os.path.join(base_path, tgt_folder)
    os.makedirs(tgt_path, exist_ok=True)
    
    print(f"Processing {src_folder} -> {tgt_folder}")
    if not os.path.exists(src_path):
        print(f"Source folder {src_path} does not exist!")
        continue

    file_count = 0
    for filename in os.listdir(src_path):
        if filename.endswith(".BMP"):
            file_count += 1
            img_path = os.path.join(src_path, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Failed to load {img_path}")
                continue
            
            print(f"Processing {filename} (File {file_count})")
            # 원본 저장
            original_save_path = os.path.join(tgt_path, filename)
            success = cv2.imwrite(original_save_path, image)
            if success:
                print(f"Saved original: {original_save_path}")
            else:
                print(f"Failed to save original: {original_save_path}")
            
            # 증강 이미지 5개 생성
            for i in range(5):
                augmented = transform(image=image)
                aug_image = augmented["image"]
                if aug_image is not None:
                    aug_filename = f"{filename.split('.')[0]}_aug{i}.BMP"
                    aug_save_path = os.path.join(tgt_path, aug_filename)
                    success = cv2.imwrite(aug_save_path, aug_image)
                    if success:
                        print(f"Saved augmented: {aug_save_path}")
                    else:
                        print(f"Failed to save augmented: {aug_save_path} (Check write permissions or disk space)")
                else:
                    print(f"Augmentation failed for {filename}_aug{i}")

    print(f"Processed {file_count} files in {src_folder}")

print("데이터 증강 완료!")