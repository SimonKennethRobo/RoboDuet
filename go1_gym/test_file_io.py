"""Filesystem failure injection without simulator dependencies."""
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from go1_gym.file_io import _last_warning, atomic_output, optional_output


class OutputTests(unittest.TestCase):
    def setUp(self):
        _last_warning.clear()

    def test_os_errors_at_open_write_and_close_are_skipped(self):
        for stage in ('open', 'write', 'close'):
            class Stream:
                def __enter__(self):
                    if stage == 'open':
                        raise OSError(5, 'Input/output error')
                    return self

                def write(self):
                    if stage == 'write':
                        raise OSError(5, 'Input/output error')

                def __exit__(self, *args):
                    if stage == 'close':
                        raise OSError(5, 'Input/output error')
            with optional_output(stage), Stream() as stream:
                stream.write()

    def test_warning_throttling(self):
        output = io.StringIO()
        with redirect_stderr(output), patch('go1_gym.file_io.time.monotonic', side_effect=[0, 1, 61]):
            for _ in range(3):
                with optional_output('log'):
                    raise OSError(5, 'Input/output error')
        self.assertEqual(output.getvalue().count('[WARNING]'), 2)

    def test_model_errors_propagate(self):
        for error in (RuntimeError('size mismatch'), ValueError('invalid data')):
            with self.assertRaises(type(error)):
                with optional_output('model'):
                    raise error

    def test_pytorch_storage_errors_are_skipped(self):
        for message in ('PytorchStreamWriter failed writing file data/0: file write failed',
                        'unexpected pos 64 vs 0', 'File /nfs/model cannot be opened.'):
            with optional_output('checkpoint'):
                raise RuntimeError(message)

    def test_failed_write_preserves_previous_checkpoint_and_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latest.pt'
            path.write_bytes(b'old checkpoint')

            def fail(target):
                Path(target).write_bytes(b'partial')
                raise OSError(5, 'Input/output error')

            with optional_output('checkpoint'):
                atomic_output(path, fail)
            self.assertEqual(path.read_bytes(), b'old checkpoint')
            self.assertEqual(list(Path(directory).iterdir()), [path])
            atomic_output(path, lambda target: Path(target).write_bytes(b'new checkpoint'))
            self.assertEqual(path.read_bytes(), b'new checkpoint')

    def test_failed_rename_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latest.pt'
            path.write_bytes(b'old')
            with patch('go1_gym.file_io.os.replace', side_effect=OSError(5, 'Input/output error')):
                with optional_output('checkpoint'):
                    atomic_output(path, lambda target: Path(target).write_bytes(b'new'))
            self.assertEqual(path.read_bytes(), b'old')


if __name__ == '__main__':
    unittest.main()
