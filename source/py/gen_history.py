"""Generations history."""

from logging import warning
from pathlib import Path

import gradio as gr

PROMPTS_HISTORY_MAX_ROWS = 300
"""Maximum searchable prompts in history frame."""


def get_prompts_history(input_dir: Path) -> list[str]:
    """Get prompts history.

    Args:
        input_dir: Directory containing generated images and their prompts.

    Returns:
        A list of unique prompts, most recent first.
    """
    if not isinstance(input_dir, Path):
        raise TypeError("input_dir must be a Path")

    prompts: list[str] = []  # We don't use a set to keep order.

    recent_images_files: list[Path] = sorted(
        input_dir.glob("image-[0-9]*.png"),  # Pattern: image-{timestamp}.png
        reverse=True,
    )[:PROMPTS_HISTORY_MAX_ROWS]

    # Filesystem is used as a database.
    # This shortcut implies to be cautious about file existence and contents.
    for image_file in recent_images_files:
        prompt_file = input_dir / f"{image_file.stem}.prompt.txt"

        if not prompt_file.exists():
            warning(f"{prompt_file} doesn't exist")
            continue

        try:
            prompt = prompt_file.read_text(encoding="utf-8")
        except Exception as error:
            warning(f"Failed to read {prompt_file}: {error}")
            continue

        if prompt and prompt not in prompts:
            prompts.append(prompt)

    return prompts


def add_prompt_to_history(candidate_prompt: str, history: list[list[str]]):
    """Add a new entry to prompts history frame
    and ensure this frame is visible.

    Args:
        candidate_prompt: Prompt we want to add to history.
        history: Current prompts history.

    Returns:
        Updated history frame value and visibility.
    """
    new_prompt = candidate_prompt.strip()

    if not new_prompt or [new_prompt] in history:
        return gr.skip()

    history.insert(0, [new_prompt])
    return gr.update(value=history, visible=True)


def on_prompts_history_row_select(event: gr.SelectData) -> str:
    """Handle row selection in prompts history frame.

    Returns:
        Selected prompt.
    """
    if not event.value:
        raise RuntimeError("Prompt is not available")

    return event.value
