"""Trigger word helpers."""


def update_trigger_word(trigger_words: list, prompt: str) -> str:
    """Update the trigger word in the prompt.

    Args:
        trigger_words: List of [previous, current] trigger words.
        prompt: The current prompt as a string.

    Returns:
        Updated prompt.
    """
    previous_tw, current_tw = trigger_words

    # Removes the previous trigger word from start of the prompt.
    if previous_tw and prompt.startswith(previous_tw):
        prompt = prompt[len(previous_tw) :].lstrip()

    # Adds the current trigger word to start of the prompt.
    if current_tw:
        prompt = f"{current_tw} {prompt}"

    return prompt


def remove_trigger_word(trigger_words: list, prompt: str) -> tuple:
    """Remove the current trigger word from the prompt.

    Args:
        trigger_words: List of [previous, current] trigger words.
        prompt: The current prompt as a string.

    Returns:
        Tuple of (empty trigger words list, updated prompt).
    """
    _, current_tw = trigger_words

    # Removes the current trigger word from start of the prompt.
    if current_tw and prompt.startswith(current_tw):
        prompt = prompt[len(current_tw) :].lstrip()

    return [None, None], prompt
