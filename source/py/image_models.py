"""Image models management."""

from pathlib import Path

from pydantic import TypeAdapter

from .image_model import ImageModel


def get_models(json_file: Path) -> list[ImageModel]:
    """Get image models from a JSON file."""
    adapter = TypeAdapter(list[ImageModel])
    models_json = json_file.read_text(encoding="utf-8")

    return adapter.validate_json(models_json)


def find_model(model_id: str, models: list[ImageModel]) -> ImageModel:
    """Find an image model by its ID among a list.
    Raises: StopIteration if not found.
    """
    return next(m for m in models if m.id == model_id)
