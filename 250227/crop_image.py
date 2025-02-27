# image cropping code

import cv2

# 이미지 파일 경로
image_path = "./TEST_0002.BMP"

# 이미지 로드
image = cv2.imread(image_path)

# 자를 좌표 설정
x_start, x_end = 311, 765
y_start, y_end = 101, 579

# 이미지 자르기
cropped_image = image[y_start:y_end, x_start:x_end]

# 저장할 파일 경로
cropped_image_path = "./cropped_image.BMP"

# 자른 이미지 저장
cv2.imwrite(cropped_image_path, cropped_image)

# 결과 출력
print(f"가운데 부분을 자른 이미지가 저장되었습니다: {cropped_image_path}")
