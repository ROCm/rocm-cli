# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Require the endpoint API key on every route of a vLLM OpenAI server.

rocm-cli passes ``--middleware rocm_cli_vllm_auth.RequireEndpointApiKey`` to
``vllm serve`` whenever it protects an endpoint with an API key. vLLM's own
``VLLM_API_KEY`` check guards only a fixed set of path prefixes (``/v1`` and
a few others), while the server also registers inference-capable routes
outside them (``/invocations``, ``/tokenize``, ...). This middleware closes
that gap by checking the key on every HTTP and WebSocket request, whatever
its path, so a route vLLM adds in a later release is covered too.

Only ``GET``/``HEAD /health`` and CORS preflight requests pass without the
key: the first is the liveness probe rocm-cli advertises, the second carries
no credentials by design and is answered by vLLM's CORS middleware before
any route runs.

The key is read from ``VLLM_API_KEY`` -- the same variable vLLM's own check
reads, set by rocm-cli on the server process -- so both checks always agree.
Construction fails when the variable is empty, so a launch that asked for
protection cannot come up unprotected.

Pure ASGI and standard library only: it must import in whichever vLLM
release the runtime ships, with no assumption about its Starlette version.
This file is written out by the rocm-cli vLLM engine adapter at launch;
edit the copy in the rocm-cli source tree, not the one on disk.
"""

import hashlib
import hmac
import os

KEY_ENV = "VLLM_API_KEY"

# Exact paths (after any ASGI root_path) that answer without the key, and the
# only methods they answer to. Everything else requires the key.
UNAUTHENTICATED_ROUTES = {"/health": frozenset({"GET", "HEAD"})}

_UNAUTHORIZED_BODY = b'{"error":"Unauthorized"}'

# RFC 6455 policy-violation close code. Closing before accepting makes the
# server reject the WebSocket handshake with HTTP 403.
_WS_POLICY_VIOLATION = 1008


def _digest(value):
    return hashlib.sha256(value).digest()


class RequireEndpointApiKey:
    """ASGI middleware rejecting any request that lacks the bearer key."""

    def __init__(self, app):
        key = os.environ.get(KEY_ENV, "")
        if not key:
            raise RuntimeError(
                f"{KEY_ENV} is empty; refusing to start an endpoint that was "
                "asked to require an API key without one"
            )
        self.app = app
        self._expected = _digest(key.encode("utf-8"))

    def _authorized(self, scope):
        authorization = None
        for name, value in scope.get("headers") or ():
            if name.lower() == b"authorization":
                authorization = value
                break
        if authorization is None:
            return False
        scheme, _, token = authorization.decode("latin-1").partition(" ")
        if scheme.lower() != "bearer":
            return False
        # Compare fixed-length digests so the comparison time depends on
        # neither the key's length nor how much of it matched.
        return hmac.compare_digest(_digest(token.encode("utf-8")), self._expected)

    @staticmethod
    def _route_path(scope):
        path = scope.get("path", "")
        root_path = scope.get("root_path", "")
        if root_path and path.startswith(root_path):
            path = path[len(root_path) :]
        return path

    @staticmethod
    def _is_cors_preflight(scope):
        if scope.get("method") != "OPTIONS":
            return False
        names = {name.lower() for name, _ in scope.get("headers") or ()}
        return b"origin" in names and b"access-control-request-method" in names

    def _exempt(self, scope):
        if scope["type"] != "http":
            return False
        if self._is_cors_preflight(scope):
            return True
        methods = UNAUTHENTICATED_ROUTES.get(self._route_path(scope))
        return methods is not None and scope.get("method") in methods

    async def __call__(self, scope, receive, send):
        kind = scope["type"]
        if kind not in ("http", "websocket") or self._exempt(scope):
            await self.app(scope, receive, send)
            return
        if self._authorized(scope):
            await self.app(scope, receive, send)
            return
        if kind == "websocket":
            await send({"type": "websocket.close", "code": _WS_POLICY_VIOLATION})
            return
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(_UNAUTHORIZED_BODY)).encode()),
                    (b"www-authenticate", b"Bearer"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": _UNAUTHORIZED_BODY})
