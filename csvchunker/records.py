"""Turn lines into records and fields, with the quoting rules that make CSV hard.

A CSV record is not a line: the delimiter and the newline can both appear inside
a quoted field. So record assembly is driven by quote parity - keep appending
lines until the number of *real* quotes is even - and field parsing is a state
machine that knows `""` inside a quoted field means one literal quote, not an
empty field followed by another.
"""


class CsvProblem(Exception):
    """The file cannot be parsed the way it was asked to be parsed."""


def count_quotes(text, quotechar='"'):
    """Number of quotes that open or close a field; doubled quotes are skipped."""
    total = 0
    index = 0
    length = len(text)
    while index < length:
        if text[index] == quotechar:
            if index + 1 < length and text[index + 1] == quotechar:
                index += 2
                continue
            total += 1
        index += 1
    return total


def parse_fields(text, delimiter=",", quotechar='"'):
    """Split one record body (terminator already removed) into fields."""
    if text == "":
        return []
    fields = []
    current = []
    index = 0
    length = len(text)
    quoted = False
    at_field_start = True
    while index < length:
        char = text[index]
        if quoted:
            if char == quotechar:
                if index + 1 < length and text[index + 1] == quotechar:
                    current.append(quotechar)
                    index += 2
                    continue
                quoted = False
                index += 1
                continue
            current.append(char)
            index += 1
            continue
        if char == delimiter:
            fields.append("".join(current))
            current = []
            at_field_start = True
            index += 1
            continue
        if char == quotechar:
            if at_field_start:
                quoted = True
                at_field_start = False
                index += 1
                continue
            if current and current[-1] == quotechar:
                # `a""b` outside a quoted field: a doubled quote is literal.
                current.append(quotechar)
                index += 2
                continue
        current.append(char)
        at_field_start = False
        index += 1
    if quoted:
        raise CsvProblem(f"unterminated quoted field in record: {text!r}")
    fields.append("".join(current))
    return fields


def is_multiline_candidate(text, quotechar='"'):
    """True when this line ends inside a quoted field.

    Used both by the record assembler and by the byte-range planner, which needs
    to know whether a file can be cut at line boundaries at all.
    """
    return count_quotes(_strip_terminator(text), quotechar) % 2 == 1


def _strip_terminator(text):
    if text.endswith("\r\n"):
        return text[:-2]
    if text.endswith("\n") or text.endswith("\r"):
        return text[:-1]
    return text


def iter_records(lines, delimiter=",", quotechar='"'):
    """Consume `(text, nbytes, position)` lines and yield records.

    A record is `(fields, byte_start, byte_end, line_count)`. `line_count` is 2
    or more only for a quoted field that spans a newline, which is the number the
    caller reports to decide whether byte-range splitting is legal.
    """
    body = ""
    pending = []
    start = None
    for text, nbytes, position in lines:
        if start is None:
            start = position
        body += text
        pending.append(nbytes)
        if is_multiline_candidate(body, quotechar):
            continue
        end = start + sum(pending)
        fields = parse_fields(_strip_terminator(body), delimiter, quotechar)
        yield Record(fields, start, end, len(pending))
        body = ""
        pending = []
        start = None
    if body:
        # Final record with no terminator: still a record, but flag it, because a
        # truncated last line is the most common way a download ends badly.
        end = start + sum(pending)
        yield Record(parse_fields(body, delimiter, quotechar), start, end, len(pending),
                     unterminated=True)


class Record:
    __slots__ = ("fields", "byte_start", "byte_end", "line_count", "unterminated")

    def __init__(self, fields, byte_start, byte_end, line_count, unterminated=False):
        self.fields = fields
        self.byte_start = byte_start
        self.byte_end = byte_end
        self.line_count = line_count
        self.unterminated = unterminated

    def __len__(self):
        return len(self.fields)

    def __repr__(self):
        return f"Record({self.fields!r}, bytes={self.byte_start}:{self.byte_end})"
