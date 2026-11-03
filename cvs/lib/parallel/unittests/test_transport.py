import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from cvs.lib.parallel.http_transport import HttpTransport
from cvs.lib.parallel.transport import create_transport

FAKE_CRED = 'fake-cred'


class TestCreateTransport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.auth_file = str(Path(self.tmp.name) / 'auth')
        Path(self.auth_file).write_text(f'{FAKE_CRED}\n')

    @patch('cvs.lib.parallel.ssh_transport.SshTransport')
    def test_create_transport_ssh_returns_ssh_transport(self, mock_ssh_transport):
        mock_ssh_transport.return_value = MagicMock()
        result = create_transport(['h1'], transport='ssh', user='u', password=FAKE_CRED)

        mock_ssh_transport.assert_called_once_with(['h1'], user='u', password=FAKE_CRED, pkey='id_rsa')
        self.assertIs(result, mock_ssh_transport.return_value)

    @patch('cvs.lib.parallel.http_transport.ParallelHTTPClient')
    def test_create_transport_http_returns_http_transport(self, mock_http_client):
        instance = MagicMock()
        instance.destroy = AsyncMock()
        mock_http_client.return_value = instance
        result = create_transport(
            ['h1'],
            transport='http',
            user='ignored',
            password=FAKE_CRED,
            agent_port_map={'h1': 9},
            token_file=self.auth_file,
        )
        self.addCleanup(result.destroy)
        self.assertIsInstance(result, HttpTransport)
        mock_http_client.assert_called_once()

    def test_create_transport_http_requires_agent_port_map(self):
        with self.assertRaises(TypeError):
            create_transport(['h1'], transport='http', token_file=self.auth_file)

    def test_create_transport_unknown_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown transport: 'bogus'"):
            create_transport(['h1'], transport='bogus')


if __name__ == '__main__':
    unittest.main()
