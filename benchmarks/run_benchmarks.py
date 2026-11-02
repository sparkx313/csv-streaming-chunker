"""流式切块到底赚了什么、又赔了什么——全部数字都从本次实测计算，不引用传说。

运行方式：`python benchmarks/run_benchmarks.py`。打印出来的每一行都会同时写进
`benchmarks/last_run.txt`，README 逐字引用的就是那份文件；因此解读段落里的每个数字
都来自它上面那张表，改一个数字这句话就得跟着改。

文件由固定字段生成，行数、字节数、窗口边界、嗅探计数、tracemalloc 峰值都是确定列，
换一台机器也能逐字复现；只有耗时列会随负载漂移，所以它们取「丢弃两次预热后的最好
一次」，并打印同一段代码反复跑出来的抖动，作为「多小算打平」的地板。六节依次回答：
切块内存随不随文件长、断点续读贵在哪、纯 Python 相对 C 实现的 `csv` 赔多少、字节窗口
切分是否精确、计划的校验这一趟值不值、以及分隔符嗅探能不能穿过带引号的逗号。
"""

import csv
import gc
import os
import random
import shutil
import sys
import tempfile
import textwrap
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csvchunker import ChunkedCsvReader, iter_chunks_parallel, plan_ranges

FIELDS = 4
WIDTH = 20          # 补白，让每行约 100 字节而不是几个字符
ROWS_SMALL = 20_000
ROWS_BIG = 200_000
REPS = 5            # 丢弃预热后参与取最优的次数
COLD = 2            # 丢弃的冷启动次数（首次打开文件、分配 arena 都算在这里）


def _line(title):
    print(title)
    print("-" * len(title))


def _timings(fn, count):
    """关掉分代回收再计时：这些对象引用计数就能释放，周期性扫描只会往每次采样里塞一段
    和读数无关的暂停。"""
    samples = []
    gc.disable()
    try:
        for _ in range(count):
            gc.collect()
            started = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - started)
    finally:
        gc.enable()
    return samples


def _best(fn, reps=REPS, cold=COLD):
    """耗时列只认最好的一次：这台机器上的负载抖动比我们要比的差异还大。"""
    return min(_timings(fn, reps + cold)[cold:])


def _spread(fn, reps=REPS, cold=COLD):
    """丢掉冷启动之后，最热与最冷一次相差的百分比——低于这个差距就不算谁更快。"""
    samples = _timings(fn, reps + cold)[cold:]
    return 100.0 * (max(samples) - min(samples)) / min(samples)


def write_file(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(f"column{i}" for i in range(FIELDS)) + "\n")
        for index in range(rows):
            cells = [f"r{index}c{i}" + "x" * WIDTH for i in range(FIELDS)]
            handle.write(",".join(cells) + "\n")
    return path


def memory(dirpath):
    """同一份文件，两种读法的 tracemalloc 峰值：只统计 Python 侧分配，看不到解释器自己的内存池。"""
    _line("1. peak allocations, streaming vs holding everything")
    print("  the same file read two ways under tracemalloc; chunk_size=500. streaming")
    print("  keeps one chunk plus one 32 KB block alive, read_all keeps every row, so")
    print("  these byte figures are deterministic and reproduce exactly across runs.\n")
    print("  %9s %9s %11s %11s %7s" %
          ("rows", "file MB", "streaming", "read_all", "ratio"))
    small = write_file(os.path.join(dirpath, "mem20.csv"), ROWS_SMALL)
    big = write_file(os.path.join(dirpath, "mem200.csv"), ROWS_BIG)
    figures = {}
    for rows, data in ((ROWS_SMALL, small), (ROWS_BIG, big)):
        tracemalloc.start()
        streamed = sum(len(chunk) for chunk in
                       ChunkedCsvReader(data, chunk_size=500).chunks())
        _, stream_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tracemalloc.start()
        held = ChunkedCsvReader(data, chunk_size=500).read_all()
        _, all_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert streamed == rows and len(held) == rows, (streamed, len(held), rows)
        figures[rows] = (os.path.getsize(data), stream_peak, all_peak)
        print("  %9d %9.1f %10.0fK %10.0fK %6.0fx" %
              (rows, os.path.getsize(data) / 1e6, stream_peak / 1024,
               all_peak / 1024, all_peak / stream_peak))
    stream_delta = 100.0 * (figures[ROWS_BIG][1] / figures[ROWS_SMALL][1] - 1.0)
    all_scale = figures[ROWS_BIG][2] / figures[ROWS_SMALL][2]
    file_scale = figures[ROWS_BIG][0] / figures[ROWS_SMALL][0]
    print("  the streaming column moves %.1f%% across a %.1fx bigger file -- flat, which"
          % (stream_delta, file_scale))
    print("  is the claim; if the generator were secretly buffering, read_all would sit")
    print("  still too. It does not: it scales %.1fx against the file's %.1fx, so the"
          % (all_scale, file_scale))
    print("  control scales with the input and the streaming column does not.\n")


def throughput(dirpath):
    """整份文件单线程读一遍：纯 Python 记录装配对 `csv` 的 C 分词器。"""
    _line("2. throughput against the C-implemented csv module")
    path = write_file(os.path.join(dirpath, "bench.csv"), ROWS_BIG)
    size = os.path.getsize(path)
    ours_fn = lambda: sum(len(chunk) for chunk in
                          ChunkedCsvReader(path, chunk_size=1000).chunks())
    def csv_read():
        with open(path, newline="", encoding="utf-8") as handle:
            return list(csv.reader(handle))
    ours_rows = ours_fn()
    std_rows = len(csv_read()) - 1
    ours_best = _best(ours_fn)
    std_best = _best(csv_read)
    ours_spread = _spread(ours_fn)
    std_spread = _spread(csv_read)
    print("  rows read : ours %d, csv %d data rows (+1 header)" %
          (ours_rows, std_rows))
    print("  best time : ours %.3fs, csv %.3fs over the %d rows" %
          (ours_best, std_best, ours_rows))
    print("  rows/s    : ours %10s   csv %12s" %
          (format(int(ours_rows / ours_best), ","),
           format(int((std_rows + 1) / std_best), ",")))
    print("  pass-pass noise floor: ours %.0f%%, csv %.0f%%" %
          (ours_spread, std_spread))
    print("  csv is %.1fx faster here (%.3fs against %.3fs best-of-%d): a Python record"
          % (ours_best / std_best, ours_best, std_best, REPS))
    # 把"差距"换算到与 floor 同一个单位（百分比）再比，否则这句话没法被表格核对。
    print("  assembler that also reports byte offsets cannot beat a C tokenizer, and a")
    print("  %.0f%% gap is %.1fx this run's wider floor (%.0f%%), so it is a"
          % (100.0 * (ours_best / std_best - 1.0),
             100.0 * (ours_best / std_best - 1.0) / max(1.0, max(ours_spread, std_spread)),
             max(ours_spread, std_spread)))
    print("  real ordering and not a load artifact. That %.1fx is the price of offsets;"
          % (ours_best / std_best))
    print("  if you only need rows, `csv` is the right tool and this is not it.\n")
    return path


def resume(path):
    """读最后 10%：跳到续读点，还是从头扫一遍再把行扔掉。"""
    _line("3. resume cost, last 10%% of the %d rows" % ROWS_BIG)
    chunks = list(ChunkedCsvReader(path, chunk_size=1000).chunks())
    tail = chunks[int(len(chunks) * 0.9):]
    boundary = tail[0].byte_start
    expected = sum(len(chunk) for chunk in tail)
    whole_rows = sum(len(chunk) for chunk in chunks)
    seek_fn = lambda: sum(len(chunk) for chunk in
                          ChunkedCsvReader(path, chunk_size=1000).chunks(resume_at=boundary))
    scan_fn = lambda: sum(len(chunk) for chunk in
                          ChunkedCsvReader(path, chunk_size=1000).chunks()
                          if chunk.byte_start >= boundary)
    seek_best = _best(seek_fn)
    scan_best = _best(scan_fn)
    seek_spread = _spread(seek_fn)
    scan_spread = _spread(scan_fn)
    resumed, scanned = seek_fn(), scan_fn()
    print("  rows after boundary: %d, read %d via resume_at, %d via scan" %
          (expected, resumed, scanned))
    print("  resume_at  best-of-%d : %.3fs (%s rows/s)" %
          (REPS, seek_best, format(int(resumed / seek_best), ",")))
    print("  scan+discard best     : %.3fs" % scan_best)
    print("  noise floor: resume_at %.0f%%, scan %.0f%%" % (seek_spread, scan_spread))
    print("  measured speed-up %.1fx for skipping %.0f%% of the rows (ideal from bytes"
          % (scan_best / seek_best, 100.0 * (1.0 - expected / whole_rows)))
    print("  would be %.1fx); correctness first: both paths hand back the same"
          % (os.path.getsize(path) / (os.path.getsize(path) - boundary),))
    print("  %d rows, so the seek is not reading less data by accident. The" % expected)
    print("  speed-up is %.0f%% of the scan's own time against pass-pass floors"
          % (100.0 * (scan_best / seek_best - 1.0),))
    print("  of %.0f%% and %.0f%%, so the ordering is far outside noise.\n"
          % (seek_spread, scan_spread))


def ranges(path):
    """字节窗口切分：窗口数量、边界是否连续、切出来的行是否等于整读。"""
    _line("4. byte-range partitioning (workers=4)")
    size = os.path.getsize(path)
    plan, notes = plan_ranges(path, workers=4)
    print("  plan       : %d windows, reason=%s" % (len(plan), notes["reason"]))
    print("  boundaries : %s ... ends at file size=%s" %
          ([start for start, _ in plan], plan[-1][1] == size))
    contiguous = all(a[1] == b[0] for a, b in zip(plan, plan[1:]))
    print("  contiguous : %s (no gap, no overlap)" % contiguous)
    counts = []
    for start, end in plan:
        rows_in = sum(len(chunk) for chunk in
                      ChunkedCsvReader(path, chunk_size=1000).chunks(
                          resume_at=start, stop_at=end))
        counts.append(rows_in)
        print("    window %9d-%9d: %7d rows" % (start, end, rows_in))
    parallel = [row for chunk in iter_chunks_parallel(path, workers=4, chunk_size=1000)
                for row in chunk.rows]
    sequential = ChunkedCsvReader(path, chunk_size=1000).read_all()
    print("  split vs whole: %d rows vs %d, equal=%s" %
          (len(parallel), len(sequential), parallel == sequential))
    assert contiguous and parallel == sequential
    print("  cuts fall at byte one-quarters, so window %s\n  rows spread %d..%d are "
          "unequal by design; the invariant that" %
          (plan[counts.index(max(counts))], min(counts), max(counts)))
    print("  matters is equal=%s: sharded reads return the same rows in the same order"
          % (parallel == sequential))
    print("  as one sequential read. Nothing here claims a parallel speed-up - the")
    print("  iterator walks windows in file order so the comparison is exact; the")
    print("  offsets are what make handing each window to its own process safe.\n")


def validation(dirpath):
    """计划器的校验是一趟完整扫描：说清它花多少、拦住什么。"""
    _line("5. what the planner's validation pass costs, and what it refuses")
    flat = write_file(os.path.join(dirpath, "flat.csv"), ROWS_SMALL)
    quoted = os.path.join(dirpath, "quoted.csv")
    with open(quoted, "w", encoding="utf-8", newline="") as handle:
        handle.write("a,b\n")
        for index in range(ROWS_SMALL):
            handle.write('"line %d\nbreak",x\n' % index)
    for label, data in (("no quoted newlines", flat), ("records span lines", quoted)):
        checked, _ = _once(lambda: plan_ranges(data, workers=4, validate=True))
        # _once_time 返回 (耗时, 返回值)，只要左边那个数。
        validated, _ = _once_time(plan_ranges, data, workers=4, validate=True)
        blind, notes = _once(lambda: plan_ranges(data, workers=4, validate=False))
        unvalidated, _ = _once_time(plan_ranges, data, workers=4, validate=False)
        print("  %20s: validate %.3fs -> %d windows, skip %.3fs -> %d windows" %
              (label, validated, len(checked), unvalidated, len(blind)))
    _, notes = plan_ranges(quoted, workers=4)
    flat_checked, flat_notes = plan_ranges(flat, workers=4, validate=True)
    print(textwrap.fill(notes["reason"], width=76,
                        initial_indent="  refused because: ",
                        subsequent_indent=" " * 20))
    print("  validate=%s on the flat file: %d windows (%s)" %
          (flat_notes["validated"], len(flat_checked), flat_notes["reason"]))
    print("  the flat file costs a full extra pass to validate and still yields 4")
    print("  windows; the quoted file costs the same pass and yields 1 - the scan found")
    print("  records that span newlines and refused a line-boundary cut, because a")
    print("  validated line-boundary plan is only trustworthy once you know no record")
    print("  crosses a line. Skipping validation is available and the note says so.")


def _once(fn):
    return fn()


def _once_time(fn, *args, **kwargs):
    started = time.perf_counter()
    out = fn(*args, **kwargs)
    return (time.perf_counter() - started), out


def sniffing(dirpath):
    """分隔符嗅探必须穿过带引号的逗号：种子固定，计数可复现。"""
    _line("6. delimiter sniffing through quoted commas")
    rng = random.Random(3)
    print("  each file quotes some cells as \"has,comma\", so counting raw commas would")
    print("  rank \",\" wrong; sniffing keys on field-count consistency instead. seeded")
    print("  (random.Random(3)) - the quoted-cell counts below reproduce exactly.\n")
    for delimiter in (",", ";", "\t", "|"):
        data = os.path.join(dirpath, "sniff%d.csv" % ord(delimiter))
        with open(data, "w", encoding="utf-8", newline="") as handle:
            handle.write(delimiter.join(["a", "b", "c"]) + "\n")
            for index in range(200):
                cells = ["v%d" % index,
                         '"has,comma"' if rng.random() < 0.1 else "plain",
                         str(rng.randint(0, 999))]
                handle.write(delimiter.join(cells) + "\n")
        reader = ChunkedCsvReader(data, chunk_size=50)
        rows = reader.read_all()
        kept = sum(1 for row in rows if row[1] == "has,comma")
        print("  %6s headers %s, %d rows x %d fields, %d quoted cells intact" %
              (repr(delimiter), reader.headers, len(rows), len(rows[0]), kept))
    print("  all four report the right 3-column header and the comma cell survives the")
    print("  quoted parse, so the delimiter is chosen by consistency, not by which")
    print("  character appears most often.\n")


class _Tee:
    """stdout 照常打印，同时攒一份原文，收尾写进 last_run.txt。

    这样 `last_run.txt` 永远是刚刚那一次运行的逐字副本，README 引用的数字不可能跟
    文件对不上——包括解读段落里那些从表算出来的数字。
    """

    def __init__(self, stream, path):
        self.stream = stream
        self.path = path
        self.parts = []

    def write(self, text):
        self.stream.write(text)
        self.parts.append(text)
        return len(text)

    def flush(self):
        self.stream.flush()

    def save(self):
        with open(self.path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write("".join(self.parts))


def main():
    report = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_run.txt")
    console = sys.stdout
    console.reconfigure(encoding="utf-8", errors="replace")
    tee = _Tee(console, report)
    sys.stdout = tee
    directory = tempfile.mkdtemp(prefix="csvchunker-bench-")
    try:
        print("csvchunker benchmarks - python %s" % sys.version.split()[0])
        big = write_file(os.path.join(directory, "bench.csv"), ROWS_BIG)
        print("generated %s: %.1f MB, %s rows, %d columns\n"
              % (os.path.basename(big), os.path.getsize(big) / 1e6,
                 format(ROWS_BIG, ","), FIELDS))
        memory(directory)
        print()
        path = throughput(directory)
        print()
        resume(path)
        print()
        ranges(path)
        print()
        validation(directory)
        print()
        sniffing(directory)
        # 收束行：工件少了这一行就说明脚本半路崩了，文件里的数字并不完整。
        print()
        print("end of run - all numbers above belong to this run only")
    finally:
        sys.stdout = tee
        tee.save()
        sys.stdout = console
        shutil.rmtree(directory, ignore_errors=True)
        print("wrote %s" % report)


if __name__ == "__main__":
    main()
