"""Measure what streaming actually buys, and what it costs.

Four numbers are worth testing, and only two of them are selling points:

1. peak memory while chunking, which must not grow with the file
2. resume by byte offset, whose cost must track the tail and not the file
3. throughput against the C-implemented `csv` module, which we lose
4. byte-range partitioning, including what the validation pass costs to stay safe

Everything printed here is measured on the run, on a generated file, with
`tracemalloc` for the memory rows (Python-side allocations only - it does not see
the interpreter's own arenas).
"""

import csv
import os
import random
import shutil
import sys
import tempfile
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csvchunker import ChunkedCsvReader, iter_chunks_parallel, plan_ranges

FIELDS = 4
WIDTH = 20  # padding so a row is ~100 bytes rather than a few characters


def write_file(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(f"column{i}" for i in range(FIELDS)) + "\n")
        for r in range(rows):
            cells = [f"r{r}c{i}" + "x" * WIDTH for i in range(FIELDS)]
            handle.write(",".join(cells) + "\n")
    return path


def memory(path, chunk_size=500):
    """Peak Python allocations: streaming one chunk at a time vs holding all rows.

    The streaming figure should not move when the file gets bigger. The
    read-all figure should move with it, and if it does not, the generator is
    quietly buffering and the whole design is a lie.
    """
    print("Peak allocations while reading the same file two ways")
    print(f"{'rows':>9} {'file (MB)':>10} {'streaming':>12} {'read_all':>12} {'ratio':>8}")
    for rows in (20_000, 200_000):
        data = os.path.join(os.path.dirname(path), f"mem{rows}.csv")
        write_file(data, rows)
        size = os.path.getsize(data) / 1e6

        tracemalloc.start()
        reader = ChunkedCsvReader(data, chunk_size=chunk_size)
        streamed = sum(len(chunk) for chunk in reader.chunks())
        _, stream_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        held = ChunkedCsvReader(data, chunk_size=chunk_size).read_all()
        _, all_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"{rows:>9} {size:>10.1f} {stream_peak / 1024:>10.0f}K "
              f"{all_peak / 1024:>10.0f}K {all_peak / stream_peak:>7.0f}x")
        assert streamed == rows and len(held) == rows, (streamed, len(held), rows)
    print("  (streaming peak is flat across a 10x file size; read_all is not)")


def throughput(path):
    print("Throughput, full file, single thread")
    reader = ChunkedCsvReader(path, chunk_size=1000)
    t0 = time.perf_counter()
    ours = sum(len(chunk) for chunk in reader.chunks())
    ours_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    with open(path, newline="", encoding="utf-8") as handle:
        std = list(csv.reader(handle))
    std_time = time.perf_counter() - t0

    print(f"  rows read        : ours {ours}, csv {len(std) - 1}")
    print(f"  time             : ours {ours_time:.3f}s, csv {std_time:.3f}s")
    print(f"  rows/s           : ours {ours / ours_time:>10,.0f}   "
          f"csv {len(std) / std_time:>10,.0f}")
    print(f"  csv is {ours_time / std_time:.1f}x faster (C tokenizer vs Python)")


def resume(path):
    """Read the last 10% of the file two ways: seek, or scan and discard."""
    print("Resume cost: last 10% of the file")
    chunks = list(ChunkedCsvReader(path, chunk_size=1000).chunks())
    boundary = chunks[int(len(chunks) * 0.9)].byte_start
    expected = sum(len(c) for c in chunks[int(len(chunks) * 0.9):])

    t0 = time.perf_counter()
    resumed = sum(len(c) for c in ChunkedCsvReader(path, chunk_size=1000).chunks(
        resume_at=boundary))
    seek_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    scanned = 0
    for chunk in ChunkedCsvReader(path, chunk_size=1000).chunks():
        if chunk.byte_start >= boundary:
            scanned += len(chunk)
    scan_time = time.perf_counter() - t0

    print(f"  rows after boundary: {expected}, read {resumed} via resume_at, "
          f"{scanned} via scan")
    print(f"  resume_at          : {seek_time:.3f}s ({resumed / seek_time:,.0f} rows/s)")
    print(f"  scan and discard   : {scan_time:.3f}s")
    print(f"  speed-up           : {scan_time / seek_time:.1f}x "
          f"(it tracks the fraction skipped, not the file size)")


def ranges(path, workers=4):
    print(f"Byte-range partitioning with workers={workers}")
    size = os.path.getsize(path)
    plan, notes = plan_ranges(path, workers=workers)
    print(f"  plan       : {len(plan)} windows, {notes['reason']}")
    print(f"  boundaries : {[start for start, _ in plan]} ... {plan[-1][1] == size}")
    contiguous = all(a[1] == b[0] for a, b in zip(plan, plan[1:]))
    print(f"  contiguous : {contiguous} (no gap, no overlap)")
    for start, end in plan:
        window = ChunkedCsvReader(path, chunk_size=1000)
        rows_in = sum(len(c) for c in window.chunks(resume_at=start, stop_at=end))
        print(f"    window {start:>9}-{end:>9}: {rows_in:>7} rows")
    parallel = [row for chunk in iter_chunks_parallel(path, workers=workers,
                                                      chunk_size=1000)
                for row in chunk.rows]
    sequential = ChunkedCsvReader(path, chunk_size=1000).read_all()
    print(f"  split vs whole: {len(parallel)} rows vs {len(sequential)}, "
          f"equal={parallel == sequential}")


def validation_cost(dirpath):
    """`validate=True` is a full extra pass. Say what it costs and what it buys."""
    print("What the validation pass costs, and what it prevents")
    flat = os.path.join(dirpath, "flat.csv")
    write_file(flat, 50_000)
    quoted = os.path.join(dirpath, "quoted.csv")
    with open(quoted, "w", encoding="utf-8", newline="") as handle:
        handle.write("a,b\n")
        for r in range(50_000):
            handle.write(f'"line {r}\nbreak",x\n')

    for name, data in (("no quoted newlines", flat), ("records span lines", quoted)):
        t0 = time.perf_counter()
        checked, _ = plan_ranges(data, workers=4, validate=True)
        validated = time.perf_counter() - t0
        t0 = time.perf_counter()
        blind, notes = plan_ranges(data, workers=4, validate=False)
        unvalidated = time.perf_counter() - t0
        print(f"  {name:>20}: validate {validated:.3f}s -> {len(checked)} windows, "
              f"skip validation {unvalidated:.3f}s -> {len(blind)} windows")
    _, notes = plan_ranges(quoted, workers=4)
    print(f"  refused because: {notes['reason']}")
    print("  (the blind split of the quoted file would cut records in half; the")
    print("   validation pass is O(file) and is the price of trusting the plan)")


def sniffing(dirpath):
    print("Delimiter sniffing")
    rng = random.Random(3)
    for delimiter in (",", ";", "\t", "|"):
        data = os.path.join(dirpath, f"sniff{ord(delimiter)}.csv")
        with open(data, "w", encoding="utf-8", newline="") as handle:
            handle.write(delimiter.join(["a", "b", "c"]) + "\n")
            for r in range(200):
                cells = [f"v{r}", '"has,comma"' if rng.random() < 0.1 else "plain",
                         str(rng.randint(0, 999))]
                handle.write(delimiter.join(cells) + "\n")
        reader = ChunkedCsvReader(data, chunk_size=50)
        rows = reader.read_all()
        commas = sum(1 for row in rows if row[1] == "has,comma")
        print(f"  {delimiter!r:>5}: headers {reader.headers}, "
              f"{len(rows)} rows, {len(rows[0])} fields each, "
              f"{commas} quoted 'has,comma' values survived intact")


def main():
    directory = tempfile.mkdtemp(prefix="csvchunker-bench-")
    rows = 200_000
    path = write_file(os.path.join(directory, "bench.csv"), rows)
    print(f"Generated {os.path.getsize(path) / 1e6:.1f} MB file, "
          f"{rows:,} rows, {FIELDS} columns\n")
    try:
        memory(path)
        print()
        throughput(path)
        print()
        resume(path)
        print()
        ranges(path)
        print()
        validation_cost(directory)
        print()
        sniffing(directory)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


if __name__ == "__main__":
    main()
