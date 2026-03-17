import json
from pathlib import Path


def get_example_prompts(json_file: Path) -> list[str]:
    """Get example prompts from a JSON file."""
    prompts = json.loads(
        json_file.read_text(encoding="utf-8"),
    )

    return [prompt["text"] for prompt in prompts]
