"""Trigger word helpers."""

import gradio as gr


def update_trigger_word(trigger_words: list, mm_prompt: dict | None) -> str:
    """Update the trigger word in the prompt.

    Args:
        trigger_words: List of [previous, current] trigger words.
        mm_prompt: Multimodal dictionary containing the prompt.

    Returns:
        Updated prompt if provided.
    """
    if not mm_prompt or not mm_prompt.get("text"):
        return gr.skip()

    previous_tw, current_tw = trigger_words
    prompt: str = mm_prompt["text"]

    # Removes the previous trigger word from start of the prompt.
    if previous_tw and prompt.startswith(previous_tw):
        prompt = prompt[len(previous_tw) :].lstrip()

    # Adds the current trigger word to start of the prompt.
    if current_tw:
        prompt = f"{current_tw} {prompt}"

    return prompt


def remove_trigger_word(trigger_words: list, mm_prompt: dict | None) -> tuple:
    """Remove the current trigger word from the prompt.

    Args:
        trigger_words: List of [previous, current] trigger words.
        mm_prompt: Multimodal dictionary containing the prompt.

    Returns:
        Tuple of (empty trigger words list, updated prompt if provided).
    """
    if not mm_prompt or not mm_prompt.get("text"):
        return [None, None], gr.skip()

    _, current_tw = trigger_words

    prompt: str = mm_prompt["text"]

    # Removes the current trigger word from start of the prompt.
    if current_tw and prompt.startswith(current_tw):
        prompt = prompt[len(current_tw) :].lstrip()

    return [None, None], prompt
