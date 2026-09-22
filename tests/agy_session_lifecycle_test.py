import asyncio
import unittest

from src.agy_session import AgySession, session_pool


class FakeStdin:
    def __init__(self, closing: bool = False):
        self.closing = closing

    def is_closing(self):
        return self.closing

    def close(self):
        self.closing = True


class FakeProcess:
    def __init__(self, *, stdin_closing: bool = False):
        self.returncode = None
        self.stdin = FakeStdin(stdin_closing)
        self.stdout = asyncio.StreamReader()

    async def wait(self):
        self.returncode = 0
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


class AgySessionLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        session_pool.clear()

    async def test_closing_stdin_is_not_usable(self):
        session = AgySession("closing")
        session.proc = FakeProcess(stdin_closing=True)
        session.ready.set()

        self.assertFalse(session._process_is_usable())

    async def test_eof_invalidates_and_unregisters_session(self):
        session = AgySession("eof")
        process = FakeProcess()
        process.stdout.feed_eof()
        session.proc = process
        session.ready.set()
        session_pool[session.session_id] = session

        chunks = [chunk async for chunk in session._read_until_result(1.0)]

        self.assertTrue(any("dừng đột ngột" in chunk for chunk in chunks))
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")
        self.assertIsNone(session.proc)
        self.assertFalse(session.ready.is_set())
        self.assertNotIn(session.session_id, session_pool)


if __name__ == "__main__":
    unittest.main()
