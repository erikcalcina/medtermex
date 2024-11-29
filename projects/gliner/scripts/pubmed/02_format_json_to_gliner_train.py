import json
from argparse import ArgumentParser
from pathlib import Path

from gliner import GLiNER
from nltk.tokenize import sent_tokenize
from tqdm import tqdm


def main(args):
    if not Path(args.input_file).exists():
        raise FileNotFoundError(f"Input file {args.input_file} does not exist")

    with open(args.input_file, "r", encoding="utf8") as f:
        data = json.load(f)

    model = GLiNER.from_pretrained(args.model_name_or_path)

    gliner_data = []
    for example in tqdm(data, desc="Processing examples"):
        sentences = sent_tokenize(example["text"])
        sentence_start = 0
        for sentence in sentences:
            token_info = []
            ws_generator = model.data_processor.words_splitter(sentence)
            for token_idx, token in enumerate(ws_generator):
                token_info.append(
                    (
                        token[0],
                        token[1] + sentence_start,
                        token[2] + sentence_start,
                        token_idx,
                    )
                )
            sentence_start += len(sentence) + 1  # +1 for the space or punctuation

            # prepare tokenized text
            tokenized_text = [t[0] for t in token_info]

            gliner_entities = []
            entities = example["entities"]
            for entity in entities:
                entity_start = entity["start"]
                entity_end = entity["end"]
                entity_tokens = []
                for token_text, token_start, token_end, token_idx in token_info:
                    if token_start >= entity_start and token_end <= entity_end:
                        entity_tokens.append((token_text, token_idx))
                if entity_tokens:
                    gliner_entities.append(
                        [
                            entity_tokens[0][1],
                            entity_tokens[-1][1],
                            entity["label"].replace("_", " ").capitalize(),
                        ]
                    )

            gliner_data.append(
                {"tokenized_text": tokenized_text, "ner": gliner_entities}
            )

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf8") as f:
        json.dump(gliner_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
