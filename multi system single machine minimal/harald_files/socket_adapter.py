from __future__ import annotations

import json
import socket
from typing import Any, Dict


class SocketRPCClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 50555, timeout_sec: float = 10.0):
        self.host = host
        self.port = port
        self.timeout_sec = timeout_sec

    def rpc(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout_sec)
        try:
            s.connect((self.host, self.port))
        except Exception as e:
            return {"ok": False, "error": f"connect-failed: {e}"}

        try:
            s.sendall(json.dumps(payload).encode("utf-8"))
            data = s.recv(1_000_000)
        finally:
            s.close()

        if not data:
            return {"ok": False, "error": "no-response"}

        try:
            return json.loads(data.decode("utf-8"))
        except Exception as e:
            return {"ok": False, "error": f"bad-json: {e}"}
