from keras import models
import os
import argparse


MODEL_DIR = "model"

def get_model_path(name: str):
    models = os.listdir(MODEL_DIR)
    model_names = set(map(lambda x: x.split(".")[0], models))

    if name not in model_names:
        raise ValueError(f"Model {name} not found in {MODEL_DIR}")
    if name in models:
        raise ValueError(f"Model {name} already exported")

    return os.path.join(MODEL_DIR, name + ".keras")


def export_model(name: str, path: str):
    model = models.load_model(path)

    if model is None:
        raise ValueError(f"Model {path} not found")

    model.save(os.path.join(MODEL_DIR, name))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            prog="export.py", 
            description="script to export model in the SavedModel format")
    parser.add_argument('architecture')

    args = parser.parse_args()

    model_path = get_model_path(args.architecture)

    export_model(args.architecture, model_path)

