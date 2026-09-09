"""Offline adversarial boundaries for archive resumption; no network/training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import zenodo_archive_recovery as recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dest_patch = patch.object(recovery, 'DEST', self.root)
        self.log_patch = patch.object(recovery, 'LOG', self.root)
        self.dest_patch.start()
        self.log_patch.start()
        self.spec = dict(name='litchfield.zip', size=2, md5='expected',
                         url='https://zenodo.org/api/records/16875688/files/litchfield.zip/content')

    def tearDown(self):
        self.dest_patch.stop()
        self.log_patch.stop()
        self.temp.cleanup()

    def test_final_response_only(self):
        good = 'HTTP/1.1 206 OK\r\nContent-Range: bytes 0-1/2\r\n\r\n'
        self.assertTrue(recovery.final_range_valid(good, 0, 1, 2))
        self.assertFalse(recovery.final_range_valid(good+'HTTP/1.1 200 OK\r\n\r\n', 0, 1, 2))
        self.assertTrue(recovery.final_range_valid('HTTP/1.1 302 Found\r\n\r\n'+good, 0, 1, 2))
        self.assertFalse(recovery.final_range_valid(good, 1, 1, 2))
        duplicate = good.replace('\r\n\r\n', '\r\nContent-Range: bytes 0-1/2\r\n\r\n')
        self.assertFalse(recovery.final_range_valid(duplicate, 0, 1, 2))

    def test_missing_or_mismatched_source_retained(self):
        path = self.root/'source.json'
        with self.assertRaisesRegex(RuntimeError, 'PROVENANCE_MISSING'):
            recovery.bind_source(path, self.spec, existing_data=True)
        self.assertFalse(path.exists())
        path.write_text('{"other": true}')
        with self.assertRaisesRegex(RuntimeError, 'PROVENANCE_MISMATCH'):
            recovery.bind_source(path, self.spec)
        self.assertEqual(json.loads(path.read_text()), {'other': True})

    def test_orphan_range_directory_cannot_reset_source(self):
        folder = self.root/'litchfield.zip.ranges'
        folder.mkdir()
        (folder/'0-1.part').write_bytes(b'ab')
        with self.assertRaisesRegex(RuntimeError, 'PROVENANCE_MISSING'):
            recovery.range_transfer(self.spec)
        self.assertEqual((folder/'0-1.part').read_bytes(), b'ab')

    def test_piece_requires_its_own_provenance(self):
        folder = self.root/'litchfield.zip.ranges'
        folder.mkdir()
        (folder/'0-1.part').write_bytes(b'ab')
        with self.assertRaisesRegex(RuntimeError, 'PROVENANCE_MISSING'):
            recovery.range_piece(self.spec, 0, 1)

    def test_wytham_requires_verified_litchfield(self):
        wytham = dict(self.spec, name='wytham.zip')
        with patch.object(recovery, 'specifications', return_value=[self.spec, wytham]):
            with self.assertRaisesRegex(RuntimeError, 'LITCHFIELD_VERIFICATION_REQUIRED'):
                recovery.require_litchfield(wytham)
            (self.root/'litchfield.zip').write_bytes(b'ab')
            with patch.object(recovery, 'checked', side_effect=RuntimeError('MD5_MISMATCH')):
                with self.assertRaisesRegex(RuntimeError, 'MD5_MISMATCH'):
                    recovery.require_litchfield(wytham)

    def test_interrupted_payload_is_recovered_without_network(self):
        folder = self.root/'litchfield.zip.ranges'
        folder.mkdir()
        payload = folder/'0-1.attempt0.payload.123'
        payload.write_bytes(b'ab')
        header = folder/'0-1.attempt0.headers.123'
        header.write_text('HTTP/1.1 206 OK\nContent-Range: bytes 0-1/2\n\n')
        recovery.save(Path(str(payload)+'.source.json'),
                      dict(identity=self.spec, start=0, end=1, header=str(header)))
        with patch.object(recovery.subprocess, 'Popen', side_effect=AssertionError('network prohibited')):
            with patch.object(recovery.os, 'fsync') as sync:
                piece = recovery.range_piece(self.spec, 0, 1)
        self.assertEqual(piece.read_bytes(), b'ab')
        self.assertTrue(sync.called)
        self.assertFalse(payload.exists())

    def test_completion_on_last_direct_attempt_succeeds(self):
        spec = dict(self.spec, size=12)
        part = self.root/'litchfield.zip.part'
        def launch(*args, **kwargs):
            with part.open('ab') as out:
                out.write(b'x')
            return Mock(poll=Mock(return_value=0))
        with patch.object(recovery.subprocess, 'Popen', side_effect=launch) as process:
            with patch.object(recovery.time, 'sleep'), patch.object(recovery, 'checked', return_value={'ok': True}):
                with patch.object(recovery.shutil, 'disk_usage', return_value=Mock(free=10**12)):
                    self.assertEqual(recovery.transfer(spec), {'ok': True})
        self.assertEqual(process.call_count, 12)
        self.assertFalse(part.exists())
        self.assertEqual((self.root/'litchfield.zip').stat().st_size, 12)


if __name__ == '__main__':
    unittest.main()
