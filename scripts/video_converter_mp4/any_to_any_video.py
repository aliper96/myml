import os
from moviepy.editor import *

# Define the input and output folder paths.
input_folder = "./"
output_folder = "./mp4/"

# Define the output video format.
output_format = "mp4"

# Create the output folder if it doesn't exist.
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Set up encoding parameters.
codec = "libx265"
preset = "fast"
threads = 8
audio_codec = "aac"
fps = 30
nvenc = True  # Set to True if using NVIDIA GPU acceleration

# Loop through all video files in the input folder and convert them to the output format.
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith((".mp4", ".avi", ".mov", ".wmv", ".mkv", ".flv", ".webm",".mpg")):
        print(file_name)
        input_file_path = os.path.join(input_folder, file_name)
        output_file_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + "." + output_format)
        
        # Load the input video file and set the output format.
        video = VideoFileClip(input_file_path)
        if nvenc:
            video_format = "h264_nvenc"
        else:
            video_format = codec
        
        # Set the frame rate of the output video file to 30 fps.
        if hasattr(video, 'fps'):
            video = video.set_fps(fps)
        else:
            print(f"Skipping file {file_name}: no valid fps attribute found.")
            continue
        
        # Write the new video file and close the video.
        try:
            video.write_videofile(output_file_path, audio_codec=audio_codec, 
                                  preset=preset, threads=threads, 
                                  codec=video_format)
        except Exception as e:
            print(f"Error writing file {file_name}: {str(e)}")
        finally:
            video.close()
