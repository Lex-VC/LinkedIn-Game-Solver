"""Crossclimb solver — uses Groq LLM to solve clues and arrange as word ladder."""

import os
import json
from groq import Groq


def _call_llm(prompt: str, temperature: float = 0.3) -> str:
    """Send a prompt to the Groq LLM and return the raw response text."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable not set. "
            "Get one at https://console.groq.com/keys"
        )
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    msg = response.choices[0].message
    # qwen puts reasoning in a separate field; the answer is in .content
    content = msg.content or ""
    # If content is empty, check for reasoning field as fallback
    if not content.strip():
        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None) or ""
        print(f"  [debug] content was empty, reasoning length={len(reasoning)}")
        content = reasoning
    return content.strip()


def _parse_json(raw: str) -> dict:
    """Extract and parse JSON from an LLM response that may contain markdown
    or <think> tags."""
    import re
    text = raw

    # Strip <think>…</think> blocks (qwen reasoning)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip markdown fences
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    # Last resort: find the first { … } block
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)

    return json.loads(text)


def solve_crossclimb(clues: dict[int, str],
                     word_length: int) -> list[tuple[int, str]]:
    """Solve crossclimb clues and return answers in word-ladder order.

    Args:
        clues:       {row_index: clue_text}  (middle rows only).
        word_length: number of letters per word.

    Returns:
        [(row_index, "WORD"), ...] ordered top-to-bottom as a valid word
        ladder (each consecutive pair differs by exactly one letter).
    """
    clue_list = "\n".join(
        f"  Row {idx}: \"{clue}\"" for idx, clue in sorted(clues.items())
    )

    prompt = f"""You are solving a LinkedIn Crossclimb puzzle.

You have {len(clues)} clues. Each answer is a common English word of exactly \
{word_length} letters. The answers must form a WORD LADDER: every consecutive \
pair of words differs by exactly ONE letter (same length, one letter changed, \
all other positions identical).

Clues (row numbers are identifiers — the final order may differ):
{clue_list}

Instructions:
1. Solve each clue to find a {word_length}-letter word.
2. Arrange the words into a valid word ladder.
3. Return the result as JSON (no extra text).

Output format:
{{
  "reasoning": "brief explanation of your solving process",
  "ladder": [
    {{"row": <row_index>, "word": "<WORD>"}},
    ...
  ]
}}

The "ladder" array must list entries from the TOP of the ladder to the BOTTOM,
so that ladder[i] and ladder[i+1] differ by exactly one letter.
Output ONLY the JSON object — no markdown fences, no commentary. /no_think"""

    raw = _call_llm(prompt, temperature=0.3)
    print(f"--- RAW LLM RESPONSE ---\n{raw}\n--- END ---")
    data = _parse_json(raw)

    result: list[tuple[int, str]] = []
    for entry in data["ladder"]:
        result.append((int(entry["row"]), entry["word"].upper()))

    # Sanity-check: verify the ladder property
    for i in range(len(result) - 1):
        w1 = result[i][1]
        w2 = result[i + 1][1]
        diff = sum(1 for a, b in zip(w1, w2) if a != b)
        if diff != 1:
            print(f"WARNING: ladder[{i}]={w1} → ladder[{i+1}]={w2} "
                  f"differ by {diff} letter(s), expected 1")

    return result


def solve_endpoint(clue: str, word_length: int,
                   adjacent_word: str) -> str:
    """Solve a locked-row clue constrained by the adjacent ladder word.

    The answer must differ from *adjacent_word* by exactly one letter.

    Returns:
        The answer word in uppercase.
    """
    prompt = f"""Solve this word puzzle clue. The answer is a common English \
word of exactly {word_length} letters.

Clue: "{clue}"

CRITICAL CONSTRAINT: The answer must differ from "{adjacent_word}" by exactly \
ONE letter (change one letter, keep all other positions the same).

Think about which single-letter changes to "{adjacent_word}" produce real \
words, then pick the one that matches the clue.

Reply with ONLY the answer word — nothing else. /no_think"""

    raw = _call_llm(prompt, temperature=0.2)
    # Take the first word-like token
    word = raw.split()[0].strip('"\'.,!()').upper()
    return word
