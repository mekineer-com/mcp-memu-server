"""Keep Windows TCP cleanup complete after a peer resets its connection."""
import socket
from asyncio.proactor_events import _ProactorSocketTransport


class _CleanupSocket:
    def __init__(self, sock):
        self.sock = sock

    def __getattr__(self, name):
        return getattr(self.sock, name)

    def shutdown(self, how):
        try:
            return self.sock.shutdown(how)
        except ConnectionResetError as exc:
            if how != socket.SHUT_RDWR or getattr(exc, "winerror", None) != 10054:
                raise


class _SocketTransport(_ProactorSocketTransport):
    def _call_connection_lost(self, exc):
        sock = self._sock
        proxy = _CleanupSocket(sock)
        self._sock = proxy
        try:
            return super()._call_connection_lost(exc)
        finally:
            if self._sock is proxy:
                self._sock = sock


def new_event_loop():
    from asyncio import ProactorEventLoop

    class EventLoop(ProactorEventLoop):
        def _make_socket_transport(self, sock, protocol, waiter=None, extra=None, server=None):
            return _SocketTransport(self, sock, protocol, waiter, extra, server)

    return EventLoop()
