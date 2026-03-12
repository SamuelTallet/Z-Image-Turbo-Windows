"""Generations history."""

from pathlib import Path
from sqlite3 import Connection, Cursor, connect

import gradio as gr

PROMPTS_HISTORY_MAX_ROWS = 300
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
        (PROMPTS_HISTORY_MAX_ROWS,),
    )

    prompts: list[str] = []

    for sqlite_row in sqlite_result:
        prompts.append(sqlite_row[0])

    sqlite_connection.close()

    return prompts


def add_prompt_to_history_frame(candidate_prompt: str, history: list[list[str]]):
    """Add a new entry to prompts history frame
    and ensure this frame is visible.

    Args:
        candidate_prompt: Prompt we want to prepend to history frame.
        history: Current prompts history.

    Returns:
        Updated history frame value and visibility.
    """
    new_prompt = candidate_prompt.strip()

    if not new_prompt or [new_prompt] in history:
        return gr.skip()

    history.insert(0, [new_prompt])
    return gr.update(value=history, visible=True)


def insert_prompt_in_history_db(candidate_prompt: str, sqlite_file: Path):
    """Insert a new prompt into dedicated history database.

    Args:
        candidate_prompt: Prompt we want to insert into history database.
        sqlite_file: Path to prompts history SQLite database.
    """
    new_prompt = candidate_prompt.strip()

    if not new_prompt:
        return

    sqlite_connection = _open_prompts_history_db(sqlite_file)
    sqlite_cursor = sqlite_connection.cursor()

    _init_prompts_history_table(sqlite_cursor)

    sqlite_cursor.execute(
        """
        INSERT INTO prompts (prompt)
        VALUES (?)
        ON CONFLICT (prompt) DO NOTHING;
        """,
        (new_prompt,),
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
