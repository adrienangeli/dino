"""Script for comparing executions so as to find the best model"""
import glob
import json
import os
import shutil
from collections import namedtuple

import fire

def main(
    prediction_dir: str,
) -> None:
    """Compare executions

    Args:
        prediction_dir (str): The path of the directory containing the
        predictions
    """
    if not os.path.isdir(prediction_dir):
        raise Exception("--prediction-dir must be a directory")

    json_files = glob.glob(f"{prediction_dir}/*.json")
    if not json_files:
        raise Exception(f"no .json predictions under {json_files}")

    print(f"Comparing predictions of {len(json_files)} JSON files...")

    prediction_blobs = {}
    for file_path in json_files:
        with open(file_path, "r", encoding="UTF-8") as file:
            prediction_blobs[os.path.basename(file_path)] = json.load(file)

    BestModel = namedtuple("BestModel", "acc, loss, n_hidden_layers")
    best_of_best = BestModel(acc=None, loss=None, n_hidden_layers=None)
    for prediction_filename, blob in prediction_blobs.items():
        loss = blob["loss"]
        acc = blob["accuracy"]
        n_hidden_layers = blob["n_hidden_layers"]
        
        if not best_of_best.acc or acc > best_of_best.acc:
            best_of_best = BestModel(
                acc=acc,
                loss=loss,
                n_hidden_layers=n_hidden_layers,
            )


    output_json_filename = os.path.join(
        os.environ.get("VH_OUTPUTS_DIR", "."),
        f"metrics-nhl-best.json",
    )
    
    with open(output_json_filename, "w", encoding="UTF-8") as output:
        json.dump({"loss": best_of_best.loss, "acc": best_of_best.acc, "n_hidden_layers": best_of_best.n_hidden_layers},
                  output, indent=2)

if __name__ == "__main__":
    fire.Fire(main)
