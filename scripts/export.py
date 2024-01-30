from keras import models
import os
import argparse


MODEL_DIR = "model"

def get_model_path(name: str):
    models = os.listdir(MODEL_DIR)
    model_names = set(map(lambda x: x.split(".")[0], models))

    if name not in model_names:
        raise ValueError(f"Model {name} not found in {MODEL_DIR}")

    return os.path.join(MODEL_DIR, name)


def export_model(path: str):
    model = models.load_model(path + ".keras")

    if model is None:
        raise ValueError(f"Model {path} not found")

    if not os.path.exists(path):
        os.mkdir(path)

    exports = os.listdir(path)
    exports = list(map(int, exports))

    if len(exports) == 0:
        max_export = 0
    else:
        max_export = max(exports)
    
    model.export(os.path.join(path, str(max_export + 1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            prog="export.py", 
            description="script to export model in the SavedModel format")
    parser.add_argument('architecture')

    args = parser.parse_args()

    model_path = get_model_path(args.architecture)

    export_model(model_path)

    print("Export complete.")

