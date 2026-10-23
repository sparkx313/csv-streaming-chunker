import csv
import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csvchunker import (
    CsvProblem,
    ChunkedCsvReader,
    RAGGED_PAD,
    RAGGED_RAISE,
    align_forward,
    count_multiline_records,
    count_quotes,
    detect_encoding,
    iter_chunks_parallel,
    parse_fields,
    plan_ranges,
    sniff_delimiter,
)


def write(name, data, encoding="utf-8"):
    path = os.path.join(tempfile.mkdtemp(prefix="csvchunker-"), name)
    mode = "wb"
    with open(path, mode) as handle:
        handle.write(data if isinstance(data, bytes) else data.encode(encoding))
    return path


class TestFieldParsing(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(parse_fields("a,b,c"), ["a", "b", "c"])

    def test_empty_and_trailing(self):
        self.assertEqual(parse_fields("a,,b,"), ["a", "", "b", ""])
        self.assertEqual(parse_fields(""), [])

    def test_quoted_delimiter_and_newline(self):
        self.assertEqual(parse_fields('"a,b",c'), ["a,b", "c"])
        self.assertEqual(parse_fields('"a\nb",c'), ["a\nb", "c"])

    def test_doubled_quote_is_literal(self):
        self.assertEqual(parse_fields('"he said ""hi""",2'), ['he said "hi"', "2"])
        self.assertEqual(parse_fields('"""quoted"""'), ['"quoted"'])

    def test_agrees_with_stdlib_on_a_corpus(self):
        corpus = [
            "a,b,c",
            '"a","b","c"',
            'a,"b,c",d',
            'a,,',
            '"multi\nline",b',
            "  spaced , fields ",
            'quote"inside,plain',
            'trailing-quote"',
        ]
        for line in corpus:
            with self.subTest(line=line):
                ours = parse_fields(line)
                theirs = next(csv.reader(io.StringIO(line)))
                self.assertEqual(ours, theirs)

    def test_diverges_from_stdlib_on_a_quote_that_never_closes(self):
        # stdlib csv treats an unclosed quote as "the field runs to the end of
        # whatever it was given": ['""""",x'] becomes one field. A byte-range
        # chunker cannot afford that leniency - the worker that hits it swallows
        # every record after it, which is exactly the failure this library exists
        # to prevent. So these two lines are a documented difference, not a bug,
        # and the stdlib answer is recorded here so the difference stays visible.
        for line in ['""""",x', '"abc,def']:
            with self.subTest(line=line):
                with self.assertRaises(CsvProblem):
                    parse_fields(line)
                self.assertEqual(next(csv.reader(io.StringIO(line))),
                                 ['"",x'] if line == '""""",x' else ['abc,def'])

    def test_unterminated_quote_is_reported(self):
        with self.assertRaises(CsvProblem):
            parse_fields('"abc,def')

    def test_quote_parity(self):
        self.assertEqual(count_quotes('"a""b",c'), 2)
        self.assertEqual(count_quotes('"a,b'), 1)
        self.assertEqual(count_quotes("plain"), 0)


class TestLineSplitting(unittest.TestCase):
    def source(self, text, encoding="utf-8"):
        from csvchunker import LineSource

        return LineSource(io.BytesIO(text.encode(encoding)), encoding)

    def drain(self, source):
        out = []
        while True:
            line = source.readline()
            if line is None:
                return out
            out.append(line)

    def test_crlf_lf_and_cr(self):
        from csvchunker import LineSource

        for terminator in ("\n", "\r\n", "\r"):
            with self.subTest(terminator=terminator):
                source = LineSource(io.BytesIO(f"a{terminator}b".encode()), "utf-8")
                lines = self.drain(source)
                self.assertEqual([text.rstrip("\r\n") for text, _, _ in lines], ["a", "b"])

    def test_lone_trailing_cr_is_not_a_phantom_line(self):
        from csvchunker import LineSource

        source = LineSource(io.BytesIO("a\rb\r".encode()), "utf-8")
        self.assertEqual([text for text, _, _ in self.drain(source)], ["a\r", "b\r"])

    def test_line_longer_than_block(self):
        text = "x" * 5000 + "\n"
        from csvchunker import LineSource

        source = LineSource(io.BytesIO(text.encode()), "utf-8", block_bytes=7)
        lines = list(iter(lambda: source.readline(), None))
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0][1], len(text))

    def test_multibyte_character_split_across_blocks(self):
        from csvchunker import LineSource

        text = "é,ü\n"
        source = LineSource(io.BytesIO(text.encode("utf-8")), "utf-8", block_bytes=3)
        line = source.readline()
        self.assertEqual(line[0], text)
        self.assertEqual(line[1], len(text.encode("utf-8")))

    def test_offsets_are_byte_offsets_not_character_offsets(self):
        text = "ünïcode,ř\nnext,2\n"
        encoded = text.encode("utf-8")
        from csvchunker import LineSource

        source = LineSource(io.BytesIO(encoded), "utf-8")
        first = source.readline()
        self.assertEqual(first[1], len(encoded.split(b"\n")[0]) + 1)
        second = source.readline()
        self.assertEqual(second[2], first[1])

    def test_bom_detection(self):
        self.assertEqual(detect_encoding(b"\xef\xbb\xbfa,b"), ("utf-8", 3))
        self.assertEqual(detect_encoding(b"\xff\xfet"), ("utf-16", 2))
        self.assertEqual(detect_encoding(b"a,b"), ("utf-8", 0))

    def test_utf8_bom_is_stripped_and_counted(self):
        path = write("bom.csv", "é,x\n1,2\n".encode("utf-8-sig"))
        reader = ChunkedCsvReader(path, chunk_size=10)
        self.assertEqual(reader.read_all(), [["1", "2"]])
        self.assertEqual(reader.headers, ["é", "x"])
        self.assertEqual(reader.stats["bytes"], os.path.getsize(path))

    def test_utf16_file(self):
        path = write("u16.csv", "a,b\n1,2\n".encode("utf-16"))
        reader = ChunkedCsvReader(path, chunk_size=10)
        self.assertEqual(reader.read_all(), [["1", "2"]])
        self.assertEqual(reader.stats["bytes"], os.path.getsize(path))

    def test_cp1252_needs_an_explicit_encoding(self):
        path = write("win.csv", "naïve,x\n1,2\n".encode("cp1252"))
        with self.assertRaises(UnicodeDecodeError):
            ChunkedCsvReader(path, chunk_size=10).read_all()
        reader = ChunkedCsvReader(path, chunk_size=10, encoding="cp1252")
        self.assertEqual(reader.read_all(), [["1", "2"]])
        self.assertEqual(reader.headers, ["naïve", "x"])


class TestRecordAssembly(unittest.TestCase):
    def test_quoted_newline_is_one_record(self):
        path = write("multi.csv", 'a,b\n"line1\nline2",3\nx,y\n')
        reader = ChunkedCsvReader(path, chunk_size=10)
        self.assertEqual(
            reader.read_all(),
            [["line1\nline2", "3"], ["x", "y"]],
        )
        self.assertEqual(reader.stats["multiline_records"], 1)

    def test_doubled_quotes_do_not_open_a_record(self):
        path = write("esc.csv", 'a,b\n"say ""hi""",2\nc,d\n')
        reader = ChunkedCsvReader(path, chunk_size=10)
        self.assertEqual(reader.read_all(), [['say "hi"', "2"], ["c", "d"]])
        self.assertEqual(reader.stats["multiline_records"], 0)

    def test_no_final_newline(self):
        path = write("tail.csv", "a,b\n1,2\n3,4")
        reader = ChunkedCsvReader(path, chunk_size=10)
        self.assertEqual(reader.read_all(), [["1", "2"], ["3", "4"]])
        self.assertEqual(reader.stats["bytes"], os.path.getsize(path))

    def test_truncated_quote_at_eof_raises(self):
        path = write("bad.csv", "a,b\n\"open,never closed\n")
        with self.assertRaises(CsvProblem):
            ChunkedCsvReader(path, chunk_size=10).read_all()

    def test_empty_file_and_header_only(self):
        empty = write("empty.csv", "")
        self.assertEqual(ChunkedCsvReader(empty).read_all(), [])
        only = write("only.csv", "a,b\n")
        reader = ChunkedCsvReader(only)
        self.assertEqual(reader.read_all(), [])
        self.assertEqual(reader.headers, ["a", "b"])

    def test_blank_line_is_an_empty_record(self):
        path = write("blank.csv", "a,b\n1,2\n\n3,4\n")
        reader = ChunkedCsvReader(path, chunk_size=10)
        rows = reader.read_all()
        self.assertEqual(rows, [["1", "2"], [], ["3", "4"]])
        self.assertEqual(reader.stats["empty_records"], 1)


class TestChunking(unittest.TestCase):
    def test_row_budget(self):
        path = write("rows.csv", "i,v\n" + "".join(f"{i},x\n" for i in range(50)))
        reader = ChunkedCsvReader(path, chunk_size=7)
        chunks = list(reader.chunks())
        self.assertEqual([len(c) for c in chunks], [7] * 7 + [1])
        self.assertEqual(sum(len(c) for c in chunks), 50)
        self.assertEqual(chunks[0].first_row, 0)
        self.assertEqual(chunks[-1].first_row, 49)

    def test_byte_budget_caps_chunk_size_in_bytes(self):
        path = write("big.csv", "i,v\n" + "".join(f"{i},{'y' * 200}\n" for i in range(200)))
        reader = ChunkedCsvReader(path, chunk_size=10_000, chunk_bytes=4096)
        chunks = list(reader.chunks())
        self.assertGreater(len(chunks), 5)
        for chunk in chunks:
            self.assertLessEqual(chunk.byte_end - chunk.byte_start, 4096 + 220)

    def test_chunk_byte_ranges_are_contiguous_and_cover_the_body(self):
        path = write("cover.csv", "a,b\n" + "".join(f"{i},{i*2}\n" for i in range(30)))
        reader = ChunkedCsvReader(path, chunk_size=4)
        chunks = list(reader.chunks())
        header_bytes = len("a,b\n".encode())
        self.assertEqual(chunks[0].byte_start, header_bytes)
        for earlier, later in zip(chunks, chunks[1:]):
            self.assertEqual(earlier.byte_end, later.byte_start)
        self.assertEqual(chunks[-1].byte_end, os.path.getsize(path))

    def test_resume_from_a_chunk_offset(self):
        path = write("resume.csv", "a,b\n" + "".join(f"{i},{i*2}\n" for i in range(100)))
        first = ChunkedCsvReader(path, chunk_size=30)
        chunks = list(first.chunks())
        tail = [row for chunk in chunks[2:] for row in chunk.rows]
        resumed = ChunkedCsvReader(path, chunk_size=30)
        rest = list(resumed.chunks(resume_at=chunks[2].byte_start))
        self.assertEqual([row for chunk in rest for row in chunk.rows], tail)
        self.assertEqual(resumed.headers, ["a", "b"])
        # Byte offsets stay absolute - that is what resume actually guarantees.
        # `first_row` counts rows *within this pass*, because deriving the global
        # index would mean reading the prefix it just skipped.
        self.assertEqual(rest[0].byte_start, chunks[2].byte_start)
        self.assertEqual(rest[0].first_row, 0)
        self.assertEqual(rest[1].first_row, len(rest[0]))
        self.assertEqual(resumed.stats["resumes"], 1)

    def test_ragged_rows_collected_padded_or_raised(self):
        path = write("ragged.csv", "a,b,c\n1,2,3\n4,5\n6,7,8,9\n")
        collect = ChunkedCsvReader(path, chunk_size=10)
        self.assertEqual(collect.read_all(), [["1", "2", "3"], ["4", "5"], ["6", "7", "8", "9"]])
        self.assertEqual(collect.stats["ragged_short"], 1)
        self.assertEqual(collect.stats["ragged_long"], 1)
        pad = ChunkedCsvReader(path, chunk_size=10, on_ragged=RAGGED_PAD)
        self.assertEqual(pad.read_all(), [["1", "2", "3"], ["4", "5", None], ["6", "7", "8"]])
        strict = ChunkedCsvReader(path, chunk_size=10, on_ragged=RAGGED_RAISE)
        with self.assertRaises(CsvProblem):
            strict.read_all()

    def test_dicts_when_asked(self):
        path = write("dict.csv", "id,name\n1,a\n2,b\n")
        reader = ChunkedCsvReader(path, chunk_size=10, as_dicts=True)
        self.assertEqual(reader.read_all(), [{"id": "1", "name": "a"}, {"id": "2", "name": "b"}])

    def test_header_false_keeps_the_first_row(self):
        path = write("nohdr.csv", "1,2\n3,4\n")
        reader = ChunkedCsvReader(path, chunk_size=10, header=False)
        self.assertEqual(reader.read_all(), [["1", "2"], ["3", "4"]])
        self.assertIsNone(reader.headers)

    def test_reusable_reader_replays_from_disk(self):
        path = write("replay.csv", "a\n1\n2\n3\n")
        reader = ChunkedCsvReader(path, chunk_size=2)
        self.assertEqual([len(c) for c in reader.chunks()], [2, 1])
        self.assertEqual([row for chunk in reader.chunks() for row in chunk.rows],
                         [["1"], ["2"], ["3"]])


class TestSniffing(unittest.TestCase):
    def test_semicolon_and_tab_and_pipe(self):
        for delimiter in (";", "\t", "|"):
            with self.subTest(delimiter=delimiter):
                body = delimiter.join(["h1", "h2", "h3"]) + "\n"
                body += delimiter.join(["a", "b", "c"]) + "\n"
                self.assertEqual(sniff_delimiter(body.splitlines(keepends=True)), delimiter)

    def test_commas_inside_quotes_do_not_win(self):
        text = 'addr,notes\n"1 Main St, Apt 2","says, a lot"\n"x,y",z\n'
        self.assertEqual(sniff_delimiter(text.splitlines(keepends=True)), ",")

    def test_reader_uses_the_sniffed_delimiter(self):
        path = write("semi.csv", "id;name\n1;Ada\n2;Grace\n")
        reader = ChunkedCsvReader(path, chunk_size=10)
        self.assertEqual(reader.read_all(), [["1", "Ada"], ["2", "Grace"]])
        self.assertEqual(reader.delimiter, ";")


class TestByteRanges(unittest.TestCase):
    def test_alignment_lands_after_a_terminator(self):
        path = write("align.csv", "aaa,1\nbb,2\ncccc3\nx,4\n")
        with open(path, "rb") as stream:
            size = os.path.getsize(path)
            for offset in (1, 5, 6, 11, size - 1):
                cut = align_forward(stream, offset, "utf-8", size)
                if cut is not None:
                    with self.subTest(offset=offset):
                        self.assertGreaterEqual(cut, offset)
                        with open(path, "rb") as check:
                            check.seek(cut - 1)
                            self.assertIn(check.read(1), (b"\n", b"\r"))

    def test_ranges_partition_the_file(self):
        path = write("part.csv", "i,j\n" + "".join(f"{i},{i}\n" for i in range(500)))
        ranges, notes = plan_ranges(path, workers=5)
        self.assertEqual(len(ranges), 5)
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], os.path.getsize(path))
        for earlier, later in zip(ranges, ranges[1:]):
            self.assertEqual(earlier[1], later[0])
        self.assertEqual(notes["validated"], True)

    def test_parallel_read_equals_sequential_read(self):
        rows = "".join(f'{i},"a,b",{i*3}\n' for i in range(800))
        path = write("same.csv", "i,q,r\n" + rows)
        sequential = ChunkedCsvReader(path, chunk_size=64).read_all()
        parallel = [row for chunk in iter_chunks_parallel(path, workers=6, chunk_size=64)
                    for row in chunk.rows]
        self.assertEqual(parallel, sequential)
        self.assertEqual(len(parallel), 800)

    def test_multiline_records_refuse_the_split(self):
        body = 'a,b\n1,"two\nlines"\n3,4\n'
        path = write("multi2.csv", body)
        self.assertEqual(count_multiline_records(path), 1)
        ranges, notes = plan_ranges(path, workers=4)
        self.assertEqual(len(ranges), 1)
        self.assertIn("span newlines", notes["reason"])

    def test_crlf_file_splits_without_stranding_a_linefeed(self):
        path = write("crlf.csv", "i,j\r\n" + "".join(f"{i},{i}\r\n" for i in range(400)))
        sequential = ChunkedCsvReader(path, chunk_size=32).read_all()
        ranges, _ = plan_ranges(path, workers=7)
        pieces = []
        for start, end in ranges:
            pieces.extend(row for chunk in _range_rows(path, start, end) for row in chunk.rows)
        self.assertEqual(pieces, sequential)

    def test_utf16_ranges_are_unit_aligned(self):
        path = write("u16b.csv", "i,j\n" + "".join(f"{i},{i}\n" for i in range(300)),
                     encoding="utf-16")
        sequential = ChunkedCsvReader(path, chunk_size=32).read_all()
        parallel = [row for chunk in iter_chunks_parallel(path, workers=4, chunk_size=32,
                                                          encoding="utf-16")
                    for row in chunk.rows]
        self.assertEqual(parallel, sequential)

    def test_workers_more_than_rows_yields_no_empty_ranges(self):
        path = write("few.csv", "i\n1\n2\n3\n")
        ranges, notes = plan_ranges(path, workers=8)
        self.assertLessEqual(len(ranges), notes["size"])
        for start, end in ranges:
            self.assertLess(start, end)


def _range_rows(path, start, end):
    from csvchunker import iter_range_chunks

    return iter_range_chunks(path, start, end, chunk_size=32)


if __name__ == "__main__":
    unittest.main()
