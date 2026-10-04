import asyncio
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path

from src.tools.exec import LocalExecutor


@unittest.skipUnless(os.name == "posix", "Process group cleanup uses POSIX signals")
class CancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_kills_process_group_and_drains_pipes(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "pids"
            script = """import os, signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
def stop(*args):
    child.terminate()
    child.wait()
signal.signal(signal.SIGTERM, stop)
with open(sys.argv[1], 'w') as marker:
    marker.write(str(os.getpgrp()) + ' ' + str(child.pid))
print('ready', flush=True)
time.sleep(60)
"""
            command = shlex.join([sys.executable, "-c", script, str(marker)])
            task = asyncio.create_task(
                LocalExecutor().run(command, cwd=Path(directory), timeout_s=60)
            )
            try:
                async with asyncio.timeout(5):
                    while not marker.exists() or not marker.read_text():
                        await asyncio.sleep(0.01)
                group, child = map(int, marker.read_text().split())
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(task), timeout=5)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child, 0)
                with self.assertRaises(ProcessLookupError):
                    os.killpg(group, 0)
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
