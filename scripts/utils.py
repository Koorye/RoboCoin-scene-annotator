import imageio
import os


def extract_first_frame(video_path, save_path):
    reader = imageio.get_reader(video_path)
    first_frame = reader.get_data(0)
    reader.close()
    imageio.imwrite(save_path, first_frame)


def get_lerobot_root():
    return os.path.expanduser('~/.cache/huggingface/lerobot/')


def get_filename_without_suffix(path):
    return os.path.splitext(os.path.basename(path))[0]


def ensure_dir(path):
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)


def load_image(path):
    return imageio.v2.imread(path)