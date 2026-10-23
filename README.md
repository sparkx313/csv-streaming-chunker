# csv-streaming-chunker

Reads a CSV file of any size in chunks and hands back the **byte offset** each chunk
ended at, so the next run can resume there, or four processes can each take a slice of
one file and produce exactly the rows a single sequential read would have.

No dependencies. Standard library only, Python 3.9+ (measured on CPython 3.11.9,
Windows x64).

```python
from csvchunker import ChunkedCsvReader, plan_ranges

reader = ChunkedCsvReader("orders.csv", chunk_size=5000)
for chunk in reader.chunks():
    process(chunk.rows)
    checkpoint = chunk.byte_end          # store this

# next run, or next worker
ChunkedCsvReader("orders.csv", chunk_size=5000).chunks(resume_at=checkpoint)

ranges, notes = plan_ranges("orders.csv", workers=4)   # [(0, 5888992), ...]
```

A `Chunk` is `(rows, first_row, byte_start, byte_end)`. The byte window is the point
of the whole library; the rows are what you came for.

## The three problems this exists to solve

**1. A byte offset is not a record boundary.** Seek to the middle of a file and the
first thing you land on is inside a field — possibly inside a quoted field that began
three lines earlier. `align_forward` cuts *forward* to the next line terminator and
starts after it, so no record is read twice and none is lost.

**2. Cutting at line boundaries is only legal if no record spans a line.** One quoted
newline anywhere in the file turns "the next terminator" into a position *inside* a
record. So `plan_ranges` validates first — a full pass counting multi-line records —
and **refuses to shard** a file where the cuts would corrupt data:

```
    no quoted newlines: validate 1.286s -> 4 windows, skip validation 0.001s -> 4 windows
    records span lines: validate 0.717s -> 1 windows, skip validation 0.001s -> 4 windows
  refused because: records span newlines; line-boundary cuts would split a record
```

`validate=False` is available when the caller already knows the file is one record per
line, and the note says so. The safe default costs a full pass; the number above is
that price, and it is honest about being O(file).

**3. Byte offsets must survive decoding.** Universal newlines are handled explicitly
(`\r\n`, `\n`, bare `\r`) instead of asking Python to translate them, because
translation moves the offsets. A line's byte width is recovered by re-encoding it, and
for UTF-16 that re-encode must use an explicit endianness — `str.encode("utf-16")`
prepends a BOM, which would invent two bytes that are not in the file. That is what
`byte_encoding()` in `csvchunker/decode.py:34` is for, and why a UTF-16 window starting
mid-file decodes at all: the codec can't wait for a BOM that only exists at byte 0.

Decoding is a `codecs` incremental decoder over block reads, so a multi-byte character
cut in half by the block boundary is not a crash, and a line longer than a block is not
a special case.

## Measured behaviour

`python benchmarks/run_benchmarks.py` reproduces everything below, on a freshly
generated 23.6 MB / 200,000-row / 4-column file.

### Peak allocations, streaming vs holding

`tracemalloc`, Python-side allocations only:

```
     rows  file (MB)    streaming     read_all    ratio
    20000        2.3        497K       7904K      16x
   200000       23.6        506K      79321K     157x
```

The streaming column is the claim: **497K → 506K for 10× the file**, i.e. flat, because
one chunk of 500 rows plus one 32 KB read block is all that exists at a time. The
`read_all` column is the control that proves the measurement means something: it grows
10.0× for a 10× file (7.9 MB → 79.3 MB). If the generator were quietly buffering, both
columns would scale together.

### Resume cost

Reading the last 10% of a 23.6 MB file two ways: seek to the checkpoint, or scan from
the top and throw rows away.

```
  rows after boundary: 20000, read 20000 via resume_at, 20000 via scan
  resume_at          : 0.540s (37,015 rows/s)
  scan and discard   : 5.266s
  speed-up           : 9.7x
```

9.7× for skipping 90% of the work — the speed-up tracks the fraction skipped, not the
file size, which is the property that makes it usable on a 40 GB file. Both paths return
the identical 20,000 rows, which is the correctness half of the sentence.

### Throughput against `csv`

```
  rows read        : ours 200000, csv 200000
  time             : ours 5.942s, csv 0.337s
  rows/s           : ours     33,656   csv    593,278
  csv is 17.6x faster (C tokenizer vs Python)
```

Python's `csv` module is a C extension. This is pure Python doing quote-parity record
assembly and byte accounting on every line, so it is 17.6× slower per row and that is
the cost of the three features above. If you just need rows, use `csv`. If you need
resumable, sharded, offset-exact reads of a file that does not fit in memory, `csv`
cannot do it.

### Byte-range partitioning

```
  plan       : 4 windows, line-boundary split
  boundaries : [0, 5888992, 11777832, 17666712] ... True
  contiguous : True (no gap, no overlap)
    window         0-  5888992:   51150 rows
    window   5888992- 11777832:   50702 rows
    window  11777832- 17666712:   49074 rows
    window  17666712- 23555592:   49074 rows
  split vs whole: 200000 rows vs 200000, equal=True
```

Windows are unequal in row count — cuts are at byte one-quarters, so rows land wherever
the text falls. The invariant that matters is `equal=True`: sharded reads produce the
same rows, in the same order, as one sequential read.

Note there is no parallel *speedup* claimed here: `iter_chunks_parallel` walks the
windows in file order so the comparison against a single read is exact. Handing each
window to a process is the intended use, and the offsets are what make it safe.

### Delimiter sniffing

Quote-aware, by field-count consistency across the first lines. Each file below contains
quoted `has,comma` values, so a naive counter gets the `,` case wrong:

```
    ',': headers ['a', 'b', 'c'], 200 rows, 3 fields each, 21 quoted 'has,comma' values survived intact
    ';': headers ['a', 'b', 'c'], 200 rows, 3 fields each, 21 quoted 'has,comma' values survived intact
   '\t': headers ['a', 'b', 'c'], 200 rows, 3 fields each, 19 quoted 'has,comma' values survived intact
    '|': headers ['a', 'b', 'c'], 200 rows, 3 fields each, 26 quoted 'has,comma' values survived intact
```

## Two deliberate differences from `csv`

**An unclosed quote raises.** `csv` treats a quote that never closes as "the field runs
to the end of the input": `""""",x` becomes the single field `"",x`. A chunker cannot
afford that leniency — the worker that hits it silently swallows every record after it,
which is precisely the failure this library is built to prevent. So `CsvProblem` is
raised, and `test_diverges_from_stdlib_on_a_quote_that_never_closes` records `csv`'s
answer alongside the rejection so the difference stays visible instead of drifting into
a bug.

**`Chunk.first_row` counts rows within the pass.** A resumed read starts at 0 again,
because the absolute row number is only knowable by reading the prefix that resuming was
invented to skip. Byte offsets *are* absolute — that is the guarantee the resume API
makes, and the test asserts both halves.

## What it handles

- Encodings: UTF-8 (BOM or not), UTF-16 LE/BE by BOM, any code page on explicit request
  (`encoding="cp1252"`); a UTF-8-unsafe code page raises `UnicodeDecodeError` rather than
  silently replacing bytes.
- Line endings: `\n`, `\r\n`, lone `\r`, including a `\r` that sits exactly on a block
  boundary and might be the first half of a `\r\n`.
- Fields: quoted delimiters, `""` escapes, empty fields (`a,,` → `['a', '', '']`),
  trailing records with no final newline (flagged `unterminated`).
- Ragged rows: `on_ragged="collect"` (pass through), `"pad"` (None-fill or truncate to
  header width), `"raise"`.
- Chunking: by row count, by byte budget, or both; `stats` reports rows, chunks, bytes,
  multi-line records, empty records, ragged counts and resumes.

## Tests

```
python -m unittest discover -s tests -p "test_*.py" -t tests

Ran 41 tests in 0.2s

OK
```

The load-bearing ones: a corpus compared field-for-field against `csv`; chunk byte
ranges asserted contiguous and complete against the file size; `parallel == sequential`
on a multi-worker plan; CRLF and UTF-16 files split *inside* a line ending and still
recovered; resume asserted row-identical to a full read; the refusal to shard a file with
quoted newlines.

## Layout

```
csvchunker/decode.py     block reads, incremental decode, exact line byte widths
csvchunker/records.py    quote-parity record assembly, field parsing
csvchunker/sniff.py      delimiter detection
csvchunker/reader.py     ChunkedCsvReader, Chunk, ragged-row policies
csvchunker/ranges.py     align_forward, plan_ranges, per-window iteration
tests/test_chunker.py    41 tests
benchmarks/              memory, throughput, resume, partitioning, validation, sniffing
```

## Limits

- No writer, no dtype inference, no `dict`-per-row beyond `as_dicts=True`, no gzip: a
  compressed file has no seekable byte offsets, which is the whole premise.
- `chunk_bytes` is approximate by construction — a byte budget can only be checked once
  a record is complete, so a chunk can overshoot by one record.
- Multi-line detection is quote-parity, a heuristic: a stray unbalanced quote in an
  otherwise plain field makes a line look like it continues.
- Single-threaded by design. The library produces offsets; sharding them across workers
  is the caller's scheduler.
