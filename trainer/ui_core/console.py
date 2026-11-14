"""Shared compressed console history, derived from Aozora Trainer (see NOTICE)."""

import zlib


class CompressedLogBuffer:
    def __init__(self, block_size=128, compression_level=6, max_active_bytes=64 * 1024):
        self.block_size = max(16, int(block_size))
        self.compression_level = max(1, min(9, int(compression_level)))
        self.max_active_bytes = max(4096, int(max_active_bytes))
        self.blocks = []
        self.current_lines = []
        self.current_bytes = 0
        self.line_count = 0
        self.uncompressed_bytes = 0
        self.compressed_bytes = 0

    def clear(self):
        self.blocks.clear()
        self.current_lines.clear()
        self.current_bytes = 0
        self.line_count = 0
        self.uncompressed_bytes = 0
        self.compressed_bytes = 0

    def append(self, text, replace_last=False):
        if replace_last:
            self.remove_last_line()
        lines = str(text).rstrip("\n").splitlines()
        if not lines:
            lines = [""]
        for line in lines:
            line_bytes = len(line.encode("utf-8", errors="replace")) + 1
            self.current_lines.append(line)
            self.current_bytes += line_bytes
            self.line_count += 1
            self.uncompressed_bytes += line_bytes
            if (
                len(self.current_lines) >= self.block_size
                or self.current_bytes >= self.max_active_bytes
            ):
                self._seal_current_block()

    def remove_last_line(self):
        if self.line_count <= 0:
            return
        if self.current_lines:
            removed = self.current_lines.pop()
            self.current_bytes = max(
                0,
                self.current_bytes - len(removed.encode("utf-8", errors="replace")) - 1,
            )
            self.line_count -= 1
            self.uncompressed_bytes = max(
                0,
                self.uncompressed_bytes
                - len(removed.encode("utf-8", errors="replace"))
                - 1,
            )
            return
        if not self.blocks:
            return
        lines = self._decode_block(len(self.blocks) - 1)
        old_payload_size = len(self.blocks[-1][1])
        if lines:
            removed = lines.pop()
            self.line_count -= 1
            self.uncompressed_bytes = max(
                0,
                self.uncompressed_bytes
                - len(removed.encode("utf-8", errors="replace"))
                - 1,
            )
        self.blocks.pop()
        self.compressed_bytes = max(0, self.compressed_bytes - old_payload_size)
        if lines:
            self.current_lines = lines
            self.current_bytes = sum(
                len(line.encode("utf-8", errors="replace")) + 1
                for line in self.current_lines
            )
            if (
                len(self.current_lines) >= self.block_size
                or self.current_bytes >= self.max_active_bytes
            ):
                self._seal_current_block()

    def get_lines(self, start_line, count):
        if count <= 0 or self.line_count <= 0:
            return []
        start_line = max(0, min(int(start_line), self.line_count - 1))
        end_line = min(self.line_count, start_line + int(count))
        output = []
        cursor = 0
        for line_count, payload in self.blocks:
            block_start = cursor
            block_end = cursor + line_count
            if block_end > start_line and block_start < end_line:
                lines = (
                    zlib.decompress(payload)
                    .decode("utf-8", errors="replace")
                    .split("\n")
                )
                local_start = max(0, start_line - block_start)
                local_end = min(line_count, end_line - block_start)
                output.extend(lines[local_start:local_end])
            cursor = block_end
            if cursor >= end_line:
                break
        if cursor < end_line and self.current_lines:
            block_start = cursor
            block_end = cursor + len(self.current_lines)
            if block_end > start_line and block_start < end_line:
                local_start = max(0, start_line - block_start)
                local_end = min(len(self.current_lines), end_line - block_start)
                output.extend(self.current_lines[local_start:local_end])
        return output

    def memory_summary(self):
        stored = self.compressed_bytes + self.current_bytes
        ratio = (
            1.0 if self.uncompressed_bytes <= 0 else stored / self.uncompressed_bytes
        )
        return stored, self.uncompressed_bytes, ratio

    def get_all_text(self):
        if self.line_count <= 0:
            return ""
        return "\n".join(self.get_lines(0, self.line_count))

    def _seal_current_block(self):
        if not self.current_lines:
            return
        raw = "\n".join(self.current_lines).encode("utf-8", errors="replace")
        payload = zlib.compress(raw, self.compression_level)
        self.blocks.append((len(self.current_lines), payload))
        self.compressed_bytes += len(payload)
        self.current_lines = []
        self.current_bytes = 0

    def _decode_block(self, index):
        if not (0 <= index < len(self.blocks)):
            return []
        line_count, payload = self.blocks[index]
        lines = zlib.decompress(payload).decode("utf-8", errors="replace").split("\n")
        return lines[:line_count]
