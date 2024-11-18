import json


# open and return the json file content
def open_json(file_path: str) -> dict:
    """
    Opens a JSON file and returns its contents as a dictionary.

    Args:
        file_path: The path to the JSON file.

    Returns:
        The contents of the JSON file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(file_path: str, data: dict) -> None:
    """
    Write a dictionary to a JSON file.

    Args:
        file_path: The path to the file where the JSON data will be written.
        data: The dictionary to be written to the JSON file.

    Returns:
        None
    """
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)
