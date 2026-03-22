from typing import Literal

from pydantic import BaseModel


class ImageModel(BaseModel):
    """An image model."""

    id: str
    """Model ID at Hugging Face.
    Example: Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32
    """

    backup_id: str | None
    """Backup model ID at Hugging Face."""

    name: str
    """Model name.
    Example: Z-Image Turbo
    """

    codename: str | None
    """Model codename.
    Example: ZiT
    """

    family: str
    """Family of this model.
    Example: Z-Image
    """

    pipeline: str
    """Diffusers pipeline class of this model."""

    required_steps: int
    """Inference steps required for this model."""

    base_ids: list[str]
    """Known IDs of base image model.
    Example: zimage
    """

    features: list[
        Literal[
            "text-to-image",
            "image-to-image",
        ]
    ]
    """Features supported by this model."""

    license_url: str
    """Link to this model license."""
