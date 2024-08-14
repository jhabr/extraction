import os


def parent_dir(directory) -> str:
    return os.path.dirname(directory)


ROOT_DIR = parent_dir(os.path.abspath(os.path.join(parent_dir(__file__))))
GPT4o_IMAGES_DIR = os.path.join(ROOT_DIR, "gpt4o", "images")
GPT4o_EXPORT_DIR = os.path.join(ROOT_DIR, "gpt4o", "export")
