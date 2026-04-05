"""Pinpoint solver — uses Groq LLM to guess the category from revealed clues."""

import os
from groq import Groq


def guess_category(clues: list[str], previous_guesses: list[str] | None = None) -> str:
    """Given a list of revealed clue words, guess the common category.

    Args:
        clues: List of revealed clue words.
        previous_guesses: List of previously incorrect guesses to avoid repeating.

    Returns:
        A single category guess string.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable not set. "
            "Get one at https://console.groq.com/keys"
        )

    client = Groq(api_key=api_key)

    clue_list = "\n".join(f"- {c}" for c in clues)

    avoid_text = ""
    if previous_guesses:
        avoid_text = (
            "\n\nThe following guesses were already tried and are WRONG, "
            "do NOT repeat them:\n"
            + "\n".join(f"- {g}" for g in previous_guesses)
        )

    prompt = f"""You are playing a word association game called Pinpoint.
    You are given clue words. They ALL belong to the exact same hidden category.
    Your task: find the ONE category that fits every single clue simultaneously.

    Clues revealed so far:
    {clue_list}
    {avoid_text}

    Rules:
    - The category must explain ALL clues, not just the most recent one.
    - Think about what property or context is shared across EVERY clue.
    - Be specific but concise (1-5 words).
    - Output ONLY the category. No explanation, no punctuation, no extra text.

    Common category types:
    - "___ [word]" or "[word] ___" (compound words or phrases)
    - Things found in a specific place
    - Things that have a specific property
    - Types of a broader concept"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        include_reasoning=False,
        max_completion_tokens=4000
    )

    raw = response.choices[0].message.content.strip()
    return raw.strip('"\'.,!').strip()
