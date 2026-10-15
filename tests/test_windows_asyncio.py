import socket
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.services.windows_asyncio import _CleanupSocket, _SocketTransport


@pytest.mark.parametrize("failure", [None, "reset", "other_reset", "other_error"])
@pytest.mark.parametrize("protocol_failure", [False, True])
def test_cleanup_preserves_protocol_close_and_detach(failure, protocol_failure):
    error = None
    if failure in {"reset", "other_reset"}:
        error = ConnectionResetError("fixture reset")
        error.winerror = 10054 if failure == "reset" else 10053
    elif failure:
        error = OSError("fixture socket failure")
    protocol_error = RuntimeError("fixture protocol failure") if protocol_failure else None
    sock = SimpleNamespace(fileno=lambda: 99, shutdown=Mock(side_effect=error), close=Mock())
    protocol = SimpleNamespace(connection_lost=Mock(side_effect=protocol_error))
    server = SimpleNamespace(_detach=Mock())
    transport = object.__new__(_SocketTransport)
    transport._sock = sock
    transport._protocol = protocol
    transport._server = server
    transport._called_connection_lost = False
    unexpected = failure in {"other_reset", "other_error"}
    expected = error if unexpected else protocol_error
    try:
        if expected:
            with pytest.raises(type(expected), match=str(expected)):
                transport._call_connection_lost(None)
        else:
            transport._call_connection_lost(None)
        protocol.connection_lost.assert_called_once_with(None)
        sock.shutdown.assert_called_once_with(socket.SHUT_RDWR)
        if unexpected:
            assert transport._sock is sock
            sock.close.assert_not_called()
            server._detach.assert_not_called()
            assert not transport._called_connection_lost
        else:
            sock.close.assert_called_once()
            server._detach.assert_called_once()
            assert transport._sock is None and transport._called_connection_lost
            transport._call_connection_lost(None)
            protocol.connection_lost.assert_called_once()
            server._detach.assert_called_once()
    finally:
        transport._sock = None


def test_nonterminal_shutdown_errors_are_not_suppressed():
    error = ConnectionResetError("fixture reset")
    error.winerror = 10054
    proxy = _CleanupSocket(SimpleNamespace(shutdown=Mock(side_effect=error)))
    with pytest.raises(ConnectionResetError):
        proxy.shutdown(socket.SHUT_WR)
