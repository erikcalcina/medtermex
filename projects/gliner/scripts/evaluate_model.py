import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from gliner import GLiNER
from tqdm import tqdm

from common.evaluate import evaluate_ner_performance


def main(args):
    if not Path(args.data_test_file).exists():
        raise FileNotFoundError(f"Test data file {args.data_test_file} does not exist")

    with open(args.data_test_file, "r", encoding="utf8") as f:
        data = json.load(f)

    # load the device (GPU or CPU)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu"
    )

    model = GLiNER.from_pretrained(args.model_name_or_path)
    model.to(device)

    performances = {
        "exact": {
            "p": [],
            "r": [],
            "f1": [],
        },
        "relaxed": {
            "p": [],
            "r": [],
            "f1": [],
        },
        "overlap": {
            "p": [],
            "r": [],
            "f1": [],
        },
    }

    labels = list(set([e["label"] for e in data for e in e["labels"]]))

    for example in tqdm(data, desc="Processing examples"):
        true_ents = example["labels"]
        pred_ents = model.predict_entities(
            example["text"], labels, threshold=args.threshold
        )

        # evaluate the performance
        for match_type in performances:
            p, r, f1 = evaluate_ner_performance(
                true_ents,
                pred_ents,
                match_type,
            )
            performances[match_type]["p"].append(p)
            performances[match_type]["r"].append(r)
            performances[match_type]["f1"].append(f1)

    for match_type in performances:
        for metric in performances[match_type]:
            performances[match_type][metric] = np.mean(
                np.array(performances[match_type][metric])
            )

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf8") as f:
        json.dump(performances, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_test_file",
        type=str,
        help="path to the test data file",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="path to the pre-trained model or model name",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="path to the output file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="threshold for the prediction (default is 0.5)",
    )
    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="whether to use CPU (default is GPU)",
    )
    args = parser.parse_args()
    main(args)
