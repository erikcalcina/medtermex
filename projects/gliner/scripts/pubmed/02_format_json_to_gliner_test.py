import json
from argparse import ArgumentParser
from pathlib import Path

from nltk.tokenize import sent_tokenize
from tqdm import tqdm


def main(args):
    if not Path(args.input_file).exists():
        raise FileNotFoundError(f"Input file {args.input_file} does not exist")

    with open(args.input_file, "r", encoding="utf8") as f:
        data = json.load(f)

    gliner_data = []
    for example in tqdm(data, desc="Processing examples"):
        sentence_start = 0
        sentence_end = 0
        for sentence in sent_tokenize(example["text"]):
            sentence_start = sentence_end
            sentence_end = sentence_start + len(sentence) + 1
            sentence_entities = []
            for entity in example["entities"]:
                if (
                    entity["text"] in sentence
                    and entity["start"] >= sentence_start
                    and entity["end"] <= sentence_end
                ):
                    sentence_entities.append(
                        {
                            "text": entity["text"],
                            "label": entity["label"].replace("_", " ").capitalize(),
                        }
                    )
            gliner_data.append({"text": sentence, "labels": sentence_entities})

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf8") as f:
        json.dump(gliner_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
