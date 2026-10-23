"""Decode a byte stream into lines without ever holding the whole file.

The reader is built on block reads plus an incremental decoder, because three
problems only go away together:

* a multi-byte character can be cut in half by the block boundary
* a line can be longer than a block
* byte offsets have to stay exact if the caller wants to resume where it
  stopped

`len(text.encode(encoding))` gives the byte width of a decoded line for every
encoding supported here, so offsets are recovered without keeping the raw bytes.
"""

import codecs

DEFAULT_BLOCK = 1 << 15

_BOMS = (
    (b"\xef\xbb\xbf", "utf-8", 3),
    (b"\xff\xfe", "utf-16", 2),
    (b"\xfe\xff", "utf-16", 2),
)


def detect_encoding(head):
    """Return `(encoding, bom_bytes)` for the first bytes of a file."""
    for bomb, encoding, size in _BOMS:
        if head.startswith(bomb):
            return encoding, size
    return "utf-8", 0


def byte_encoding(encoding, head):
    """Codec to use when turning decoded text back into a byte count.

    `str.encode("utf-16")` prepends a byte-order mark, so re-encoding a line to
    measure it would invent two bytes that are not in the file. The explicit
    endianness does not.
    """
    if encoding == "utf-16":
        return "utf-16-be" if head.startswith(b"\xfe\xff") else "utf-16-le"
    if encoding == "utf-8":
        return "utf-8"
    return encoding


class LineSource:
    """Yield `(text, nbytes, position)` with the line terminator included.

    Universal newlines are handled explicitly (`\\r\\n`, `\\n`, bare `\\r`) rather
    than by asking Python to translate them, because translation would move the
    byte offsets the resume feature depends on.

    The stream must already be positioned at `start`; this class counts bytes, it
    does not seek. Decoding uses the explicit codec (`utf-16-le`, not `utf-16`) so
    a window that starts in the middle of the file, where no byte-order mark is
    waiting to be read, still picks the right endianness.
    """

    def __init__(self, stream, encoding="utf-8", bom_bytes=0, block_bytes=DEFAULT_BLOCK,
                 start=0, end=None, byte_encoding=None):
        self.stream = stream
        self.encoding = encoding
        self.byte_encoding = byte_encoding or encoding
        self.block_bytes = block_bytes
        self.end = end
        self.decoder = codecs.getincrementaldecoder(self.byte_encoding)()
        self.buffer = ""
        # `start` is absolute, so on a resumed window the byte-order mark is
        # already inside it; adding it again would shift every offset after.
        self.position = start + bom_bytes if start == 0 else start
        self.eof = False
        self._first = True

    def _pull(self):
        """Read one block of bytes into the text buffer. True if anything came."""
        if self.eof:
            return False
        count = self.block_bytes
        if self.end is not None:
            count = min(count, self.end - self.stream.tell())
        if count <= 0:
            raw = b""
        else:
            raw = self.stream.read(count)
        if not raw:
            self.eof = True
            self.buffer += self.decoder.decode(b"", final=True)
            return False
        self.buffer += self.decoder.decode(raw)
        if self._first and self.buffer.startswith("\ufeff"):
            # utf-8-sig style BOM delivered as a character: it is still bytes on
            # disk, so the offset has to know about it.
            self.buffer = self.buffer[1:]
        self._first = False
        return True

    def __iter__(self):
        while True:
            line = self.readline()
            if line is None:
                return
            yield line

    def readline(self):
        """Return the next line, or None at end of input."""
        while True:
            cut = _terminator(self.buffer)
            if cut is not None:
                index, length = cut
                # A bare \r at the very end of the buffer may be the first half
                # of \r\n, so it only terminates when more input exists.
                if index + length == len(self.buffer) and not self.eof and length == 1 \
                        and self.buffer[index] == "\r":
                    if not self._pull():
                        continue
                text = self.buffer[: index + length]
                self.buffer = self.buffer[index + length:]
                nbytes = len(text.encode(self.byte_encoding))
                position = self.position
                self.position += nbytes
                return text, nbytes, position
            if not self._pull():
                if not self.buffer:
                    return None
                text = self.buffer
                self.buffer = ""
                nbytes = len(text.encode(self.byte_encoding))
                position = self.position
                self.position += nbytes
                return text, nbytes, position


def _terminator(text):
    """Index and length of the first line terminator in `text`, or None.

    `\\r` counts as a terminator only when it is not followed by `\\n` yet, so a
    CRLF file never yields a phantom empty line.
    """
    lf = text.find("\n")
    cr = text.find("\r")
    if cr != -1 and (lf == -1 or cr < lf):
        if cr + 1 < len(text):
            return (cr, 2) if text[cr + 1] == "\n" else (cr, 1)
        return None
    if lf != -1:
        return (lf, 1)
    return None
