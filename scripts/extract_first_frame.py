"""
python scripts/extract_first_frame.py \
    --video_dir="path/to/lerobot/video/dir/" \
    --save_dir="results/frames/"
"""

import argparse
import imageio
import os
from tqdm import tqdm


def extract_first_frame(video_path, save_path):
    reader = imageio.get_reader(video_path)
    first_frame = reader.get_data(0)
    reader.close()
    imageio.imwrite(save_path, first_frame)


def get_filename(path):
    return os.path.splitext(os.path.basename(path))[0] + ".png"


def main(args):
    assert (args.video_path is None) != (args.video_dir is None), "Either video_path or video_dir must be provided, but not both."

    os.makedirs(args.save_dir, exist_ok=True)
    if args.video_path:
        extract_first_frame(args.video_path, os.path.join(args.save_dir, get_filename(args.video_path)))
    elif args.video_dir:
        video_files = [f for f in os.listdir(args.video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        for video_file in tqdm(video_files, desc="Extracting first frames"):
            video_path = os.path.join(args.video_dir, video_file)
            save_path = os.path.join(args.save_dir, get_filename(video_file))
            extract_first_frame(video_path, save_path)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract first frames from videos in a directory.")
    parser.add_argument("--video_path", type=str, help="Path to a single video file.")
    parser.add_argument("--video_dir", type=str, help="Path to the directory containing video files.")
    parser.add_argument("--save_dir", type=str, default="first_frames", help="Directory to save the extracted first frames.")
    args = parser.parse_args()
    main(args)