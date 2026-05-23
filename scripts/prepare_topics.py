from config_loader import load_env_file
from data_topics import prepare_topic_splits


if __name__ == "__main__":
    load_env_file("evolution")
    paths = prepare_topic_splits()
    for name, path in paths.items():
        print(f"{name}: {path}")
