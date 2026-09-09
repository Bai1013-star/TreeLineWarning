"""Offline safety boundaries; no data or network required."""
import tempfile
import io
from unittest.mock import patch
from pathlib import Path
import unittest
import zipfile
import public_data as p


class DownloadSafety(unittest.TestCase):
    def test_ply_eof(self):
        for value in (b'', b'ply\n', b'ply\nformat ascii 1.0\n'):
            with self.assertRaises(RuntimeError):
                p.read_ply_header(io.BytesIO(value), 'truncated.ply')
        self.assertEqual(p.read_ply_header(io.BytesIO(b'ply\nend_header\n'), 'valid.ply'),
                         'ply\nend_header\n')

    def test_inventory_central_directory_only(self):
        root = Path(tempfile.mkdtemp(prefix='public_data_header_', dir='D:/TreeQCTraining/tmp'))
        path = root/'fixture.zip'
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('truncated.ply', 'ply\n')
        with patch.object(zipfile.ZipFile, 'open', side_effect=AssertionError('Payload read forbidden')):
            self.assertEqual(p.inventory(path)['entry_count'], 1)
        self.assertFalse(path.with_suffix('.ply_headers.json').exists())

    def test_approval_required(self):
        with self.assertRaises(RuntimeError):
            p.approved(None, 'download')

    def test_ranges(self):
        p.validate_response(206, 'bytes 5-9/10', 5, 10)
        for status, value in [(200, ''), (206, 'bytes 4-9/10'),
                              (206, 'bytes 5-10/10'), (206, 'bytes 5-9/11')]:
            with self.assertRaises(RuntimeError):
                p.validate_response(status, value, 5, 10)

    def test_expansion_cap(self):
        p.expansion_cap(10, {'max_total_expanded_bytes': 10})
        with self.assertRaises(RuntimeError):
            p.expansion_cap(11, {'max_total_expanded_bytes': 10})

    def test_traversal(self):
        # Retain tiny fixtures as evidence; no deletion/cleanup.
        root = Path(tempfile.mkdtemp(prefix='public_data_safety_', dir='D:/TreeQCTraining/tmp'))
        for i, name in enumerate(['../escape', '/absolute', 'C:/outside', 'sub/CON.txt']):
            path = root/f'{i}.zip'
            with zipfile.ZipFile(path, 'w') as z:
                z.writestr(name, 'fixture')
            with self.assertRaises(RuntimeError):
                p.inventory(path)


if __name__ == '__main__':
    unittest.main()
