# Description: Convert AVI files to MP4 files using ffmpeg
# Put converted videos in a new folder named "converted_videos".

import ffmpeg
import os

def convert_ms_cram_to_mp4(input_file, output_file):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
    
    try:
        (
            ffmpeg
            .input(input_file)
            .output(output_file, vcodec='libx264', pix_fmt='yuv420p')
            .run(overwrite_output=True)
        )
        print(f"Conversion successful: {output_file}")
    except ffmpeg.Error as e:
        print("Error occurred:", e)
        print(e.stderr.decode())

def convert_folder_avi_to_mp4(folder_path, output_folder):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' not found.")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file in os.listdir(folder_path):
        if file.lower().endswith(".avi"):
            input_file = os.path.join(folder_path, file)
            output_file = os.path.join(output_folder, os.path.splitext(file)[0] + ".mp4")
            print(f"Converting {input_file} to {output_file}...")
            convert_ms_cram_to_mp4(input_file, output_file)

if __name__ == "__main__":
    folder_path = "./"  # 현재 폴더 내의 모든 AVI 파일 변환
    output_folder = "./converted_videos"  # 변환된 파일 저장 폴더
    convert_folder_avi_to_mp4(folder_path, output_folder)
