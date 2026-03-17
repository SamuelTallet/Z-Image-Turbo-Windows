from pydantic import BaseModel


class ImageModel(BaseModel):
    """An image model."""

    id: str
    """Model ID at Hugging Face.
    Example: Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32
    """

    backup_id: str
    """Backup model ID at Hugging Face."""

    pipeline: str
    """Diffusers pipeline class of this model."""
    # Value is checked on model load.

    extra_features: list[str]
    """Extra features supported by this model."""
