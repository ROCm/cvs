import unittest
from unittest.mock import patch, MagicMock

from cvs.lib.parallel.transport import create_transport


class TestCreateTransport(unittest.TestCase):
    @patch('cvs.lib.parallel.ssh_transport.SshTransport')
    def test_create_transport_ssh_returns_ssh_transport(self, mock_ssh_transport):
        mock_ssh_transport.return_value = MagicMock()
        result = create_transport(['h1'], transport='ssh', user='u', password='p')

        mock_ssh_transport.assert_called_once_with(['h1'], user='u', password='p', pkey='id_rsa')
        self.assertIs(result, mock_ssh_transport.return_value)

    def test_create_transport_http_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            create_transport(['h1'], transport='http')

    def test_create_transport_unknown_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown transport: 'bogus'"):
            create_transport(['h1'], transport='bogus')


if __name__ == '__main__':
    unittest.main()
