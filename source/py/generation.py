from pathlib import Path
from time import time

from PIL.Image import Image


class Generation:
    """A generated image, and its associated prompt and settings."""

    model: str
    """Model used to generate image."""

    image: Image
    """Generated image."""

    prompt: str
    """Prompt used to generate image."""

    resolution: str
    """Desired resolution for generated image."""

    seed: int
    """Seed used to generate image."""

    steps: int
    """Number of steps used to generate image."""

    def __init__(
        self,
        model: str,
        image: Image,
        prompt: str,
        resolution: str,
        seed: int,
        steps: int,
    ) -> None:
        """Initialize a generation object."""
        self.model = model
        self.image = image
        self.prompt = prompt
        self.resolution = resolution
        self.seed = seed
        self.steps = steps

    def save(self, output_dir: Path) -> tuple[Path, Path, Path]:
        """Save image, prompt and settings in given output directory.

        Returns:
            Tuple of (image file, prompt file, settings file).
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Including timestamp in filename allows for chronological sort.
        # This also prevents filename conflicts...
        timestamp = int(time())

        image_file = output_dir / f"image-{timestamp}.png"
        self.image.save(image_file)

        prompt_file = output_dir / f"image-{timestamp}.prompt.txt"
        prompt_file.write_text(self.prompt, encoding="utf-8")

        # Using INI format makes obvious this file contains parameters
        # and distinguishes it from prompt.txt in File Explorer.
        settings_file = output_dir / f"image-{timestamp}.settings.ini"
        settings_file.write_text(
            f"Model = {self.model}\n"
            f"Resolution = {self.resolution}\n"
            f"Seed = {self.seed}\n"
            f"Steps = {self.steps}\n"
        )

        return image_file, prompt_file, settings_file
