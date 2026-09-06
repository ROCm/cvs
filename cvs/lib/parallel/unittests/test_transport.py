import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from cvs.lib.parallel.http_transport import HttpTransport
from cvs.lib.parallel.transport import create_transport


class TestCreateTransport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.token_file = str(Path(self.tmp.name) / 'secret')
        Path(self.token_file).write_text('tok\n')

    @patch('cvs.lib.parallel.ssh_transport.SshTransport')
    def test_create_transport_ssh_returns_ssh_transport(self, mock_ssh_transport):
        mock_ssh_transport.return_value = MagicMock()
        result = create_transport(['h1'], transport='ssh', user='u', password='p')

        mock_ssh_transport.assert_called_once_with(['h1'], user='u', password='p', pkey='id_rsa')
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
            password='ignored',
            agent_port_map={'h1': 9},
            token_file=self.token_file,
        )
        self.addCleanup(result.destroy)
        self.assertIsInstance(result, HttpTransport)
        mock_http_client.assert_called_once()

    def test_create_transport_http_requires_agent_port_map(self):
        with self.assertRaises(TypeError):
            create_transport(['h1'], transport='http', token_file=self.token_file)

    def test_create_transport_unknown_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown transport: 'bogus'"):
            create_transport(['h1'], transport='bogus')


if __name__ == '__main__':
    unittest.main()
