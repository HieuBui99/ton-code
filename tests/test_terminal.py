import errno
import os
import select
import struct
import subprocess
import sys
import time
import unittest

from src.tui.renderer import clean_text


@unittest.skipUnless(os.name == "posix", "PTY integration requires POSIX")
class TerminalTests(unittest.TestCase):
    def test_typing_paste_steering_and_scrollback(self):
        import fcntl
        import pty
        import termios

        master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
        env = dict(os.environ, TERM="xterm-256color", PROMPT_TOOLKIT_NO_CPR="1")
        process = subprocess.Popen(
            [sys.executable, "-m", "tests.fake_terminal"],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env=env,
            start_new_session=True,
        )
        os.close(slave)
        captured = bytearray()

        def read_until(marker, timeout=8):
            deadline = time.monotonic() + timeout
            while marker not in clean_text(captured.decode("utf-8", errors="replace")):
                if time.monotonic() >= deadline:
                    self.fail(f"Timed out waiting for {marker!r}: {captured[-3000:]!r}")
                if select.select([master], [], [], 0.1)[0]:
                    try:
                        chunk = os.read(master, 65536)
                    except OSError as exc:
                        if exc.errno == errno.EIO:
                            self.fail(
                                f"Terminal closed before {marker!r}: {captured[-3000:]!r}"
                            )
                        raise
                    if not chunk:
                        self.fail(f"Terminal EOF before {marker!r}")
                    captured.extend(chunk)

        try:
            read_until("ton>")
            os.write(master, b"first\r")
            read_until("REPLY first")
            os.write(master, b"dra")
            read_until("stream line 15:")
            os.write(master, b"ft\r")
            read_until("REPLY draft")
            # A pasted multiline message is one submission, even with embedded newlines.
            os.write(master, b"\x1b[200~paste one\npaste two\x1b[201~\r")
            read_until("REPLY paste one\npaste two")
            read_until("stream line 34:")
            os.write(master, b"\x03")
            read_until("Turn interrupted.")
            os.write(master, b"recovered\r")
            read_until("REPLY recovered")
            os.write(master, b"/quit\r")
            process.wait(timeout=8)
            self.assertEqual(process.returncode, 0)
            transcript = clean_text(captured.decode("utf-8", errors="replace"))
            self.assertIn("漢字 café", transcript)
            self.assertIn("offline-test", transcript)
            self.assertIn("Context [", transcript)
            self.assertIn("200,000", transcript)
            self.assertEqual(transcript.count("REPLY first"), 1)
            self.assertEqual(transcript.count("REPLY draft"), 1)
            self.assertGreaterEqual(transcript.count("You (steering)"), 2)
            self.assertNotIn(b"\x1b[?1049h", captured)
            self.assertNotIn(b"\x1b[2J", captured)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            os.close(master)
