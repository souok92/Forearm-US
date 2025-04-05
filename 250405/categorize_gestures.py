import os
import shutil
import re

def categorize_files(input_folder, output_folder):
    """
    입력 폴더의 파일들을 태그와 프레임 번호에 따라 분류하여 출력 폴더에 저장
    
    파일명 형식: name_tag_frame_epik
    태그(0-9) 기준:
        0,1: index
        2,3: mid
        4,5: ring
        6,7: pinky
        8,9: fist
    프레임 번호:
        나머지가 0-9: open
        나머지가 10-18: 해당 태그 동작
    """
    # 출력 폴더 생성
    folders = ['open', 'index', 'mid', 'ring', 'pinky', 'fist']
    for folder in folders:
        folder_path = os.path.join(output_folder, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    
    # 파일명 패턴 컴파일 - n자리 숫자 형식을 처리함
    pattern = re.compile(r'(.+)_(\d{4})_(\d{4})_epik')
    
    # 태그를 카테고리로 매핑
    tag_to_category = {
        '0': 'index', '1': 'index',
        '2': 'mid', '3': 'mid',
        '4': 'ring', '5': 'ring',
        '6': 'pinky', '7': 'pinky',
        '8': 'fist', '9': 'fist'
    }
    
    # 입력 폴더의 모든 파일 처리
    file_count = 0
    for filename in os.listdir(input_folder):
        # 파일명 파싱
        match = pattern.match(filename)
        if match:
            name, tag, frame = match.groups()
            frame = int(frame)
            
            # 태그를 정수로 변환하고 첫 번째 자리만 고려
            tag_num = int(tag) % 10  # 4자리 태그에서 마지막 자리만 사용
            tag_str = str(tag_num)
            
            # 프레임 번호에 따라 분류
            # 18로 나눈 나머지가 0-8면 open, 9-17이면 해당 태그 동작
            remainder = frame % 18
            
            if 0 <= remainder <= 8:
                category = 'open'
            else:  # 9 <= remainder <= 17
                category = tag_to_category.get(tag_str, 'unknown')
                print(f"태그: {tag}, 변환된 태그: {tag_str}, 카테고리: {category}")
            
            # 파일 복사
            src_path = os.path.join(input_folder, filename)
            
            # 소스 파일이 존재하는지 확인
            if not os.path.exists(src_path):
                print(f"경고: 소스 파일이 존재하지 않습니다: {src_path}")
                continue
                
            # _epik를 제거한 새 파일명 생성
            new_filename = filename.replace('_epik', '')
            dst_path = os.path.join(output_folder, category, new_filename)
            
            # 대상 디렉토리가 존재하는지 확인하고 필요하면 생성
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            try:
                shutil.copy2(src_path, dst_path)
                file_count += 1
                print(f"복사됨: {src_path} -> {dst_path}")
            except Exception as e:
                print(f"파일 복사 중 오류 발생: {src_path} -> {dst_path}")
                print(f"오류 내용: {str(e)}")
                continue
    
    return file_count

if __name__ == '__main__':
    # 사용자 입력 받기
    input_folder = r"C:\Users\souok\Desktop\editver"
    output_folder = r"C:\Users\souok\Desktop\cat_gest"
    
    # 경로 정규화 (백슬래시 문제 해결)
    input_folder = os.path.normpath(input_folder)
    output_folder = os.path.normpath(output_folder)
    
    print(f"입력 폴더: {input_folder}")
    print(f"출력 폴더: {output_folder}")
    
    # 입력 폴더와 출력 폴더가 존재하는지 확인
    if not os.path.exists(input_folder):
        print(f"오류: 입력 폴더 '{input_folder}'가 존재하지 않습니다.")
        exit(1)
    
    if not os.path.exists(output_folder):
        print(f"출력 폴더 '{output_folder}'가 존재하지 않아 생성합니다.")
        os.makedirs(output_folder)
    
    # 파일 분류 실행
    try:
        file_count = categorize_files(input_folder, output_folder)
        print(f"완료: {file_count}개 파일이 성공적으로 분류되었습니다.")
    except Exception as e:
        import traceback
        print(f"오류 발생: {str(e)}")
        print(traceback.format_exc())  # 자세한 오류 정보 출력