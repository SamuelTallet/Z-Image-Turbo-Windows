from pathlib import Path

from pydantic import TypeAdapter

from .image_model import ImageModel


def get_image_models(json_file: Path) -> list[ImageModel]:
    """Get image models from a JSON file."""
    adapter = TypeAdapter(list[ImageModel])

    models_json = json_file.read_text(encoding="utf-8")

    return adapter.validate_json(models_json)
