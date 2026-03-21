from pydantic import BaseModel


class ImageModel(BaseModel):
    """An image model."""

    id: str
    """Model ID at Hugging Face.
    Example: Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32
    """

    backup_id: str
    """Backup model ID at Hugging Face."""

    name: str
    """Model name.
    Example: Z-Image Turbo
    """

    family: str
    """Family of this model.
    Example: Z-Image
    """

    pipeline: str
    """Diffusers pipeline class of this model."""

    base_ids: list[str]
    """Known IDs of base image model.
    Example: zimage
    """

    features: list[str]
    """Features supported by this model."""
