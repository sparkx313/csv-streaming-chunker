"""Split one CSV file into byte ranges that can be read independently.

The trick is that a byte offset is not a record boundary. Seek to the middle of a
file and the first thing you land on is inside a field, possibly inside a quoted
field that started lines earlier. Two rules make the split exact:

* cut forward to the next line terminator and start after it, so no record is
  read twice and none is lost
* only do this when the file has no records that span lines, because a quoted
  newline makes "the next terminator" a position *inside* a record
"""

import os

from .decode import LineSource, byte_encoding, detect_encoding
from .reader import ChunkedCsvReader

_UNIT = {"utf-8": 1, "utf-16": 2, "utf-16-le": 2, "utf-16-be": 2}


def code_unit(encoding):
    return _UNIT.get(encoding, 1)


def align_forward(stream, offset, encoding, limit):
    """First line boundary at or after `offset`, or None if nothing is left.

    Rounding down to a whole code unit first means a cut inside a two-byte
    UTF-16 character still decodes, and the loop means a cut in the middle of a
    short line moves on to the next boundary rather than going backwards.
    """
    if offset >= limit:
        return None
    base = offset - offset % code_unit(encoding)
    stream.seek(base)
    source = LineSource(stream, encoding, 0, start=base, end=limit)
    while True:
        line = source.readline()
        if line is None:
            return None
        if source.position >= offset:
            return source.position


def plan_ranges(path, workers=4, encoding=None, validate=True):
    """Return `(ranges, notes)` where ranges are `(start, end)` byte windows.

    `validate` does a full pass looking for records that span lines. Skipping it
    is faster but the plan is then only correct for a file the caller already
    knows is single-line-per-record, which is what the note says.
    """
    size = os.path.getsize(path)
    with open(path, "rb") as stream:
        head = stream.read(4)
    detected, _bom = detect_encoding(head)
    encoding = encoding or detected
    # Cuts are made in bytes, so the codec with explicit endianness is what
    # decides whether a byte offset is a whole code unit; the logical name is
    # only kept for the notes the caller reads back.
    codec = byte_encoding(encoding, head)
    if workers <= 1 or size == 0:
        return [(0, size)], {"encoding": encoding, "workers": 1, "size": size,
                             "validated": False, "reason": "single range requested"}
    if validate:
        multiline = count_multiline_records(path, encoding=encoding)
        if multiline:
            return [(0, size)], {
                "encoding": encoding,
                "workers": 1,
                "size": size,
                "validated": True,
                "multiline_records": multiline,
                "reason": "records span newlines; line-boundary cuts would split a record",
            }
    cuts = [0]
    with open(path, "rb") as stream:
        targets = [max(1, size * i // workers) for i in range(1, workers)]
        for target in targets:
            cut = align_forward(stream, target, codec, size)
            if cut is None or cut <= cuts[-1]:
                continue
            cuts.append(min(cut, size))
        cuts.append(size)
    boundaries = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)
                  if cuts[i + 1] > cuts[i]]
    return boundaries, {"encoding": encoding, "workers": len(boundaries), "size": size,
                        "validated": bool(validate),
                        "reason": "line-boundary split" if validate else "unvalidated split"}


def count_multiline_records(path, encoding=None, chunk_size=4096):
    """How many records occupy more than one line, i.e. quoted newlines."""
    reader = ChunkedCsvReader(path, chunk_size=chunk_size, encoding=encoding)
    for _ in reader.chunks():
        pass
    return reader.stats["multiline_records"]


def iter_range_chunks(path, start, end, chunk_size=1024, chunk_bytes=None,
                      encoding=None, delimiter=None, header=True, on_ragged=None):
    """Chunks for one byte window, records being owned by the window they start in."""
    reader = ChunkedCsvReader(path, chunk_size=chunk_size, chunk_bytes=chunk_bytes,
                              encoding=encoding, delimiter=delimiter, header=header,
                              on_ragged=on_ragged or "collect")
    return reader.chunks(resume_at=start, stop_at=end)


def iter_chunks_parallel(path, workers=4, chunk_size=1024, encoding=None, delimiter=None):
    """All chunks, in file order, produced from independent byte windows.

    Call `plan_ranges` directly to see why a file ended up with the number of
    windows it got.
    """
    ranges, _notes = plan_ranges(path, workers=workers, encoding=encoding)
    for start, end in ranges:
        for chunk in iter_range_chunks(path, start, end, chunk_size=chunk_size,
                                       encoding=encoding, delimiter=delimiter):
            yield chunk
