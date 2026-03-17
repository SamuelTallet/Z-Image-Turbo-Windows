from pathlib import Path

from pydantic import TypeAdapter

from .image_model import ImageModel


def get_image_models(json_file: Path) -> list[ImageModel]:
    """Get image models from a JSON file."""
    adapter = TypeAdapter(list[ImageModel])

    with open(json_file, "r") as file:
        models_json = file.read()

    return adapter.validate_json(models_json)
