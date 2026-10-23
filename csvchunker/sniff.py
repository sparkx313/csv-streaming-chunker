"""Pick a delimiter from the bytes, without assuming the header says so."""

from .records import count_quotes, parse_fields

CANDIDATES = (",", ";", "\t", "|")


def sniff_delimiter(lines, candidates=CANDIDATES, quotechar='"'):
    """Return the candidate whose field count is stable across the sample.

    Counting raw occurrences ranks the busiest character, which is the wrong
    signal: a file of addresses uses commas inside quotes far more than between
    fields. Consistency is what identifies a delimiter - the real one produces
    the same field count on every well-formed row.
    """
    sample = list(lines)
    if not sample:
        return candidates[0]
    best = None
    for candidate in candidates:
        counts = []
        for text in sample:
            body = text.rstrip("\r\n")
            if count_quotes(body, quotechar) % 2:
                continue
            counts.append(len(parse_fields(body, candidate, quotechar)))
        if not counts:
            continue
        mode = max(set(counts), key=counts.count)
        mismatches = sum(1 for count in counts if count != mode)
        # A character that never splits anything is only a fallback: the ranking
        # puts it behind every candidate that produced at least two fields.
        score = (0 if mode >= 2 else 1, mismatches, -mode)
        if best is None or score < best[0]:
            best = (score, candidate)
    return best[1] if best else candidates[0]
