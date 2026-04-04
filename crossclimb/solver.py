"""Crossclimb solver — uses Groq LLM to solve clues and arrange as word ladder."""

import os
import json
from groq import Groq


_MODELS = [
    ("openai/gpt-oss-120b", 6000),
    ("openai/gpt-oss-20b",  6000),
]


def _call_llm(prompt: str, temperature: float = 0.3) -> str:
    """Send a prompt to the Groq LLM and return the raw response text.

    Tries the primary model first; falls back to a smaller model on token
    limit errors (HTTP 413 / rate_limit_exceeded).
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY environment variable not set. "
            "Get one at https://console.groq.com/keys"
        )
    client = Groq(api_key=api_key)

    last_err = None
    for model, max_tokens in _MODELS:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
                include_reasoning=False,
                reasoning_effort="medium",
                temperature=temperature,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            err_str = str(e)
            if "413" in err_str or "rate_limit" in err_str or "tokens" in err_str:
                print(f"  Token limit hit on {model}, falling back...")
                last_err = e
                continue
            raise

    raise last_err


def _parse_json(raw: str) -> dict:
    """Extract and parse JSON from an LLM response that may contain markdown."""
    import re
    text = raw

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
Output ONLY the JSON object — no markdown fences, no commentary."""

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


def solve_endpoints(clue: str, word_length: int,
                    top_adjacent: str, bottom_adjacent: str) -> tuple[str, str]:
    """Solve both locked endpoint rows from a single shared clue.

    The clue describes a two-word phrase where one word is the top locked row
    and the other is the bottom locked row (order may be swapped).
    Each answer must differ from its adjacent ladder word by exactly one letter.

    Returns:
        (top_word, bottom_word) both in uppercase.
    """
    prompt = f"""You are solving the final step of a LinkedIn Crossclimb puzzle.

The top and bottom locked rows each need a {word_length}-letter word.
The clue describes how the two words relate to each other.

Clue: "{clue}"

CRITICAL CONSTRAINTS:
- The TOP word must differ from "{top_adjacent}" by exactly ONE letter \
(change one letter, keep all other positions the same).
- The BOTTOM word must differ from "{bottom_adjacent}" by exactly ONE letter.
- The two words must satisfy the relationship described in the clue.

Think about which single-letter changes to "{top_adjacent}" and \
"{bottom_adjacent}" produce real words, then find the pair that matches \
the clue.

Output ONLY JSON:
{{"top": "<WORD>", "bottom": "<WORD>"}}"""

    raw = _call_llm(prompt, temperature=0.2)
    print(f"--- RAW ENDPOINT RESPONSE ---\n{raw}\n--- END ---")
    data = _parse_json(raw)

    top_word = data["top"].strip().upper()
    bottom_word = data["bottom"].strip().upper()
    return top_word, bottom_word
