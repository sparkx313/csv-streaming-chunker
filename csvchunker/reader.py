"""Row-count and byte-budget chunking over a CSV file, one record at a time."""

import os

from .decode import LineSource, byte_encoding, detect_encoding
from .records import CsvProblem, iter_records
from .sniff import sniff_delimiter

RAGGED_COLLECT = "collect"
RAGGED_PAD = "pad"
RAGGED_RAISE = "raise"


class Chunk:
    """A run of consecutive rows plus the byte range they came from.

    The byte range is the resume point: re-opening the file at `byte_end` and
    reading forward yields exactly the rows after this chunk, with no
    end-of-file scan and no state kept in memory.

    `first_row` is an index *within the pass that produced this chunk*. A resumed
    read starts at 0 again, because the absolute row number can only come from
    reading the prefix that resuming was meant to skip.
    """

    __slots__ = ("rows", "first_row", "byte_start", "byte_end")

    def __init__(self, rows, first_row, byte_start, byte_end):
        self.rows = rows
        self.first_row = first_row
        self.byte_start = byte_start
        self.byte_end = byte_end

    def __len__(self):
        return len(self.rows)

    def __repr__(self):
        return (f"Chunk(rows={len(self.rows)}, rows {self.first_row}"
                f"-{self.first_row + len(self.rows) - 1}, "
                f"bytes {self.byte_start}-{self.byte_end})")


class ChunkedCsvReader:
    def __init__(self, path, chunk_size=1024, chunk_bytes=None, encoding=None,
                 delimiter=None, header=True, as_dicts=False, on_ragged=RAGGED_COLLECT):
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        if chunk_bytes is not None and chunk_bytes < 1:
            raise ValueError("chunk_bytes must be at least 1")
        if on_ragged not in (RAGGED_COLLECT, RAGGED_PAD, RAGGED_RAISE):
            raise ValueError(f"unknown on_ragged policy {on_ragged!r}")
        self.path = path
        self.chunk_size = chunk_size
        self.chunk_bytes = chunk_bytes
        self.encoding = encoding
        self.delimiter = delimiter
        self.header = header
        self.as_dicts = as_dicts
        self.on_ragged = on_ragged
        self.headers = None
        self.stats = {
            "rows": 0,
            "chunks": 0,
            "bytes": 0,
            "multiline_records": 0,
            "empty_records": 0,
            "ragged_short": 0,
            "ragged_long": 0,
            "resumes": 0,
        }

    def _open(self):
        with open(self.path, "rb") as probe:
            head = probe.read(4)
        encoding, bom = detect_encoding(head)
        if self.encoding is not None:
            encoding = self.encoding
        return open(self.path, "rb"), encoding, bom, byte_encoding(encoding, head)

    def _records(self, stream, encoding, bom, byte_enc, start=0, end=None, delimiter=None):
        # The header pass and the sniff both leave the stream somewhere else, so
        # every record walk states where it begins instead of trusting the cursor.
        stream.seek(start)
        lines = LineSource(stream, encoding, bom, start=start, end=end,
                           byte_encoding=byte_enc)
        return iter_records(lines, delimiter or self.delimiter or ",")

    def chunks(self, resume_at=0, stop_at=None):
        """Yield chunks of rows from a byte window.

        `resume_at` is a boundary from a previous Chunk or from `plan_ranges`;
        `stop_at` ends the window at the first record that starts at or beyond
        it, so windows partition a file without overlapping or dropping rows.
        """
        stream, encoding, bom, byte_enc = self._open()
        try:
            if self.delimiter is None:
                self.delimiter = self._sniff(stream, encoding, bom, byte_enc)
                stream.seek(0)
            if resume_at:
                self.stats["resumes"] += 1
                self._ensure_headers(stream, encoding, bom, byte_enc)
            else:
                # The byte-order mark is bytes on disk that no record covers.
                self.stats["bytes"] += bom
            row_index = 0
            pending = []
            pending_start = None
            last_end = bom
            skip_header = self.header and not resume_at
            for record in self._records(stream, encoding, bom, byte_enc, start=resume_at,
                                        delimiter=self.delimiter):
                if stop_at is not None and record.byte_start >= stop_at:
                    break
                if skip_header:
                    self.headers = list(record.fields)
                    skip_header = False
                    last_end = record.byte_end
                    self.stats["bytes"] += record.byte_end - record.byte_start
                    continue
                width = len(self.headers) if self.headers else None
                fields, last_end = self._align(record, width)
                row_index += 1
                self.stats["rows"] += 1
                self.stats["bytes"] += record.byte_end - record.byte_start
                if record.line_count > 1:
                    self.stats["multiline_records"] += 1
                if not record.fields:
                    self.stats["empty_records"] += 1
                if pending_start is None:
                    pending_start = record.byte_start
                pending.append(self._project(fields))
                span = record.byte_end - pending_start
                if len(pending) >= self.chunk_size or (
                    self.chunk_bytes is not None and span >= self.chunk_bytes
                ):
                    self.stats["chunks"] += 1
                    yield Chunk(pending, row_index - len(pending), pending_start,
                                record.byte_end)
                    pending = []
                    pending_start = None
            if pending:
                self.stats["chunks"] += 1
                yield Chunk(pending, row_index - len(pending), pending_start, last_end)
        finally:
            stream.close()

    def _ensure_headers(self, stream, encoding, bom, byte_enc):
        """A resume offset is in the middle of the data, so the header has to come
        from the top of the file."""
        stream.seek(0)
        for record in self._records(stream, encoding, bom, byte_enc, start=0,
                                    delimiter=self.delimiter):
            self.headers = list(record.fields)
            break

    def _project(self, fields):
        if self.as_dicts and self.headers:
            return dict(zip(self.headers, fields))
        return fields

    def _align(self, record, width):
        fields = record.fields
        if width is None or len(fields) == width:
            return fields, record.byte_end
        if len(fields) < width:
            self.stats["ragged_short"] += 1
            if self.on_ragged == RAGGED_RAISE:
                raise CsvProblem(
                    f"row at byte {record.byte_start} has {len(fields)} fields, "
                    f"header has {width}")
            if self.on_ragged == RAGGED_PAD:
                return fields + [None] * (width - len(fields)), record.byte_end
            return fields, record.byte_end
        self.stats["ragged_long"] += 1
        if self.on_ragged == RAGGED_RAISE:
            raise CsvProblem(
                f"row at byte {record.byte_start} has {len(fields)} fields, "
                f"header has {width}")
        if self.on_ragged == RAGGED_PAD:
            return fields[:width], record.byte_end
        return fields, record.byte_end

    def _sniff(self, stream, encoding, bom, byte_enc):
        stream.seek(0)
        lines = LineSource(stream, encoding, bom, byte_encoding=byte_enc)
        sample = []
        for _ in range(32):
            line = lines.readline()
            if line is None:
                break
            sample.append(line[0])
        if not sample:
            return ","
        return sniff_delimiter(sample, quotechar='"')

    def rows(self, resume_at=0):
        for chunk in self.chunks(resume_at=resume_at):
            for row in chunk.rows:
                yield row

    def read_all(self, resume_at=0):
        return list(self.rows(resume_at=resume_at))

    def byte_length(self):
        return os.path.getsize(self.path)
