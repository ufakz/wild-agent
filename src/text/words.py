def count_words(text: str) -> int:
    return len(text.split())


def max_chars_for_words(max_words: int) -> int:
    """Upper bound on stored sample length (chars) derived from max_words."""
    return min(max_words * 15, 4096)
