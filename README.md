# csv-streaming-chunker

Reads a CSV file of any size in chunks and hands back the **byte offset** each chunk
ended at, so the next run can resume there, or four processes can each take a slice of
one file and produce exactly the rows a single sequential read would have.

No dependencies. Standard library only, Python 3.9+ (measured on CPython 3.11.9,
Windows x64).

```python
import os, tempfile
from csvchunker import ChunkedCsvReader, plan_ranges, iter_range_chunks

path = os.path.join(tempfile.mkdtemp(), "orders.csv")
with open(path, "w", newline="", encoding="utf-8") as handle:
    handle.write("order_id,amount\n")
    for i in range(1, 2001):
        handle.write("%d,%d\n" % (i, i * 7 % 10000))

checkpoints = [c.byte_end for c in ChunkedCsvReader(path, chunk_size=500).chunks()]
print(checkpoints)                    # [4251, 8752, 13665, 18594] - last is the file size

tail = list(ChunkedCsvReader(path, chunk_size=500).chunks(resume_at=checkpoints[1]))
print(len(tail), sum(len(c) for c in tail), tail[0].first_row)      # 2 1000 0

ranges, notes = plan_ranges(path, workers=4)
print(ranges)                         # [(0, 4656), (4656, 9302), (9302, 13953), (13953, 18594)]
print(notes["reason"], notes["validated"])   # line-boundary split True

window = ranges[2]
rows = list(iter_range_chunks(path, window[0], window[1], chunk_size=500))
print(window, sum(len(c) for c in rows), rows[0].rows[0])           # (9302, 13953) 477 ['1056', '7392']
print(sum(sum(len(c) for c in iter_range_chunks(path, s, e, chunk_size=500))
          for s, e in ranges))        # 2000 - the four windows are the whole file
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

```text
    no quoted newlines: validate 0.555s -> 4 windows, skip 0.001s -> 4 windows
    records span lines: validate 0.236s -> 1 windows, skip 0.000s -> 4 windows
  refused because: records span newlines; line-boundary cuts would split a
                    record
```

`validate=False` is available when the caller already knows the file is one record per
line, and the note says so. The safe default costs a full pass; the number above is
that price on a 23.6 MB file, and it is honest about being O(file).

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

```text
       rows   file MB   streaming    read_all   ratio
      20000       2.3        497K       7904K     16x
     200000      23.6        506K      79321K    157x
```

The streaming column is the claim: **497K → 506K for 10× the file**, i.e. flat, because
one chunk of 500 rows plus one 32 KB read block is all that exists at a time. The
`read_all` column is the control that proves the measurement means something: it grows
10.0× for a 10× file (7.9 MB → 79.3 MB). If the generator were quietly buffering, both
columns would scale together.

### Resume cost

Reading the last 10% of a 23.6 MB file two ways: seek to the checkpoint, or scan from
the top and throw rows away.

```text
  rows after boundary: 20000, read 20000 via resume_at, 20000 via scan
  resume_at  best-of-5 : 0.456s (43,821 rows/s)
  scan+discard best     : 4.496s
  noise floor: resume_at 13%, scan 6%
  measured speed-up 9.9x for skipping 90% of the rows (ideal from bytes
  would be 9.8x); correctness first: both paths hand back the same
  20000 rows, so the seek is not reading less data by accident. The
  speed-up is 885% of the scan's own time against pass-pass floors
  of 13% and 6%, so the ordering is far outside noise.
```

9.9× for skipping 90% of the work, against the 9.8× the byte ratio alone predicts.
The seek side's own pass-to-pass floor on this run is 13%, so the measured and the
predicted speed-ups are not distinguishable — the seek does the work the byte
arithmetic says it must, and no more. The speed-up tracks the fraction skipped rather
than the file size, which
is the property that makes it usable on a 40 GB file. Both paths return the identical
20,000 rows, which is the correctness half of the sentence.

### Throughput against `csv`

```text
  rows read : ours 200000, csv 200000 data rows (+1 header)
  best time : ours 4.498s, csv 0.283s over the 200000 rows
  rows/s    : ours     44,467   csv      706,046
  pass-pass noise floor: ours 15%, csv 5%
```

Python's `csv` module is a C extension. This is pure Python doing quote-parity record
assembly and byte accounting on every line, so it is 15.9× slower per row and that is
the cost of the three features above — our own column's pass-to-pass floor on this run
is 15%, so a 1488% gap is not something the load moved. If you just need rows, use
`csv`. If you need
resumable, sharded, offset-exact reads of a file that does not fit in memory, `csv`
cannot do it.

### Byte-range partitioning

```text
  plan       : 4 windows, reason=line-boundary split
  boundaries : [0, 5888992, 11777832, 17666712] ... ends at file size=True
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

```text
     ',' headers ['a', 'b', 'c'], 200 rows x 3 fields, 21 quoted cells intact
     ';' headers ['a', 'b', 'c'], 200 rows x 3 fields, 21 quoted cells intact
    '\t' headers ['a', 'b', 'c'], 200 rows x 3 fields, 19 quoted cells intact
     '|' headers ['a', 'b', 'c'], 200 rows x 3 fields, 26 quoted cells intact
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
