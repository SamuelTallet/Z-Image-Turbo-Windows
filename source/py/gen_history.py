"""Generations history."""

from pathlib import Path
from sqlite3 import Connection, Cursor, connect

import gradio as gr

SEARCHABLE_PROMPTS = 100
"""Maximum searchable prompts in history frame."""


def _open_prompts_history_db(sqlite_file: Path) -> Connection:
    """Open prompts history database.

    Args:
        sqlite_file: Path to prompts history SQLite database.

    Returns:
        SQLite database connection.
    """
    sqlite_file.parent.mkdir(parents=True, exist_ok=True)

    return connect(sqlite_file)


def _init_prompts_history_table(sqlite_cursor: Cursor) -> None:
    """Initialize prompts history database table if needed.

    Args:
        sqlite_cursor: A SQLite cursor.
    """
    sqlite_cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS prompts (
            id INTEGER PRIMARY KEY,
            prompt TEXT NOT NULL UNIQUE,
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
        );
        """
    )
    sqlite_cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_recent_prompts ON prompts (
            created_at DESC
        );
        """
    )


def get_prompts_history(sqlite_file: Path) -> list[str]:
    """Get prompts history.

    Args:
        sqlite_file: Path to prompts history SQLite database.

    Returns:
        A list of unique prompts, most recent first.
    """
    if not isinstance(sqlite_file, Path):
        raise TypeError("sqlite_file must be a Path")

    sqlite_connection = _open_prompts_history_db(sqlite_file)
    sqlite_cursor = sqlite_connection.cursor()

    _init_prompts_history_table(sqlite_cursor)

    sqlite_result = sqlite_cursor.execute(
        """
        SELECT prompt
        FROM prompts
        ORDER BY created_at DESC
        LIMIT ?;
        """,
        (SEARCHABLE_PROMPTS,),
    )

    prompts: list[str] = []

    for sqlite_row in sqlite_result:
        prompts.append(sqlite_row[0])

    sqlite_connection.close()

    return prompts


def add_prompt_to_history_frame(mm_prompt: dict | None, history: list[list[str]]):
    """Add a new entry to prompts history frame
    and ensure this frame is visible.

    Args:
        mm_prompt: Multimodal dictionary containing the prompt.
        history: Current prompts history.

    Returns:
        Updated history frame value and visibility, if new prompt provided.
    """
    if not mm_prompt or not mm_prompt.get("text"):
        return gr.skip()

    prompt: str = mm_prompt["text"]

    if [prompt] in history:
        return gr.skip()

    history.insert(0, [prompt])
    return gr.update(value=history, visible=True)


def insert_prompt_in_history_db(mm_prompt: dict | None, sqlite_file: Path):
    """Insert a new prompt into dedicated history database.

    Args:
        mm_prompt: Multimodal dictionary containing the prompt.
        sqlite_file: Path to prompts history SQLite database.
    """
    if not mm_prompt or not mm_prompt.get("text"):
        return

    prompt: str = mm_prompt["text"]

    sqlite_connection = _open_prompts_history_db(sqlite_file)
    sqlite_cursor = sqlite_connection.cursor()

    _init_prompts_history_table(sqlite_cursor)

    sqlite_cursor.execute(
        """
        INSERT INTO prompts (prompt)
        VALUES (?)
        ON CONFLICT (prompt) DO NOTHING;
        """,
        (prompt,),
    )

    sqlite_connection.commit()
    sqlite_connection.close()


def on_prompts_history_row_select(event: gr.SelectData) -> str:
    """Handle row selection in prompts history frame.

    Returns:
        Selected prompt.
    """
    if not event.value:
        raise RuntimeError("Prompt is not available")

    return event.value
