import os


def parent_dir(directory) -> str:
    return os.path.dirname(directory)


ROOT_DIR = parent_dir(os.path.abspath(os.path.join(parent_dir(__file__))))
IMAGES_DIR = os.path.join(ROOT_DIR, "gpt4o", "images")
EXPORT_DIR = os.path.join(ROOT_DIR, "gpt4o", "export")
LOCAL_EXPORT_DIR = os.path.join(EXPORT_DIR, "local")
GPT4o_EXPORT_DIR = os.path.join(EXPORT_DIR, "gpt4")
