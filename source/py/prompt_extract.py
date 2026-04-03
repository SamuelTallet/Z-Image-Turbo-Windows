"""Prompt extraction."""

import gradio as gr
from PIL import Image


def extract_prompt_from_image(file_pointer) -> str | None:
    """Extract prompt from image metadata.

    Args:
        file_pointer: File pointer (path, etc) to image.

    Returns:
        Found prompt or `None` otherwise.
    """
    image = Image.open(file_pointer)

    print(image.info)  # DEBUG

    if image.info.get("prompt"):
        return image.info["prompt"]

    return None


def extract_update_prompt(mm_prompt: dict | None) -> dict:
    """Extract prompt from image within multimodal dictionary.

    Args:
        mm_prompt: Multimodal dictionary containing prompt.

    Returns:
        Updated `dict` with found prompt if image attached.
    """
    if not mm_prompt or not mm_prompt.get("files"):
        return gr.skip()

    file_path: str = mm_prompt["files"][0]

    # Empty files to prevent infinite loop.
    mm_prompt["files"] = []

    if prompt := extract_prompt_from_image(file_path):
        mm_prompt["text"] = prompt

    return mm_prompt
