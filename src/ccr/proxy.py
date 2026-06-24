"""Lightweight proxy: Anthropic Messages API -> OpenAI Chat Completions API."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from typing import Any

from aiohttp import web, ClientSession, ClientTimeout
from aiohttp.client_exceptions import ClientConnectionResetError as _ClientConnReset

logger = logging.getLogger(__name__)

from .converter import (
    anthropic_to_openai_request,
    openai_to_anthropic_response,
    StreamConverter,
)
from .debug_log import DebugTarget, RequestRecorder, make_recorder


def _is_disconnect(exc: BaseException) -> bool:
    return isinstance(exc, (
        ConnectionResetError,
        BrokenPipeError,
        ConnectionAbortedError,
        _ClientConnReset,
    )) or "Cannot write to closing transport" in str(exc)


class ProxyServer:
    def __init__(self, api_url: str, api_key: str = "dummy", model: str = "", port: int = 0, max_output_tokens: int | None = None, debug: DebugTarget | None = None, debug_keep: int = 100, profile_name: str = ""):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.port = port
        self.max_output_tokens = max_output_tokens
        self.debug = debug
        self.debug_keep = debug_keep
        self.profile_name = profile_name
        self._runner: web.AppRunner | None = None
        self._actual_port: int = 0

    @property
    def actual_port(self) -> int:
        return self._actual_port

    async def start(self) -> int:
        app = web.Application(client_max_size=32 * 1024 * 1024)
        app.router.add_post("/v1/messages", self._handle_messages)
        app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()

        if self.port == 0:
            self.port = _find_free_port()

        site = web.TCPSite(self._runner, "127.0.0.1", self.port)
        await site.start()
        self._actual_port = self.port
        return self.port

    async def stop(self):
        if self._runner:
            await self._runner.cleanup()

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def _handle_messages(self, request: web.Request) -> web.StreamResponse:
        try:
            return await self._handle_messages_inner(request)
        except asyncio.CancelledError:
            logger.debug("Request cancelled by client")
            raise
        except BaseException as exc:
            if _is_disconnect(exc):
                logger.debug("Client disconnected")
                return web.Response(status=499)
            raise

    async def _handle_messages_inner(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        is_stream = body.get("stream", False)

        rec = make_recorder(
            self.debug, self.profile_name, "messages", self.debug_keep, "anthropic"
        )
        if rec is not None:
            rec.log_request_body(body)

        openai_req = anthropic_to_openai_request(body, model_override=self.model, max_output_tokens=self.max_output_tokens)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        timeout = ClientTimeout(total=600, sock_read=300)

        async with ClientSession(timeout=timeout) as session:
            async with session.post(
                self.api_url,
                json=openai_req,
                headers=headers,
            ) as upstream:
                if upstream.status != 200:
                    error_body = await upstream.text()
                    if rec is not None:
                        rec.log_upstream_chunk(error_body.encode("utf-8", errors="replace"))
                        rec.close("upstream_error", status=upstream.status)
                    return web.Response(
                        status=upstream.status,
                        text=error_body,
                        content_type="application/json",
                    )

                if is_stream:
                    return await self._stream_response(request, upstream, rec)
                else:
                    resp_data = await upstream.json()
                    if rec is not None:
                        rec.log_upstream_chunk(json.dumps(resp_data, ensure_ascii=False).encode())
                        rec.close("non_stream_ok")
                    anthropic_resp = openai_to_anthropic_response(resp_data, model=self.model)
                    return web.json_response(anthropic_resp)

    async def _stream_response(
        self, request: web.Request, upstream: Any, rec: RequestRecorder | None = None
    ) -> web.StreamResponse:
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        try:
            await response.prepare(request)
        except BaseException as exc:
            if _is_disconnect(exc):
                logger.debug("Client disconnected before stream started")
            raise

        converter = StreamConverter(model=self.model)
        client_disconnected = False

        async def _safe_write(data: bytes) -> bool:
            nonlocal client_disconnected
            if client_disconnected:
                return False
            try:
                await response.write(data)
                return True
            except BaseException as exc:
                if _is_disconnect(exc):
                    client_disconnected = True
                    logger.debug("Client disconnected, stopping stream")
                    return False
                raise

        try:
            for event in converter.start_events():
                if not await _safe_write(event.encode()):
                    return response

            buffer = ""
            first_byte_logged = False
            async for chunk_bytes in upstream.content.iter_any():
                if client_disconnected:
                    break

                if rec is not None:
                    rec.log_upstream_chunk(chunk_bytes)
                    if not first_byte_logged:
                        first_byte_logged = True
                        rec.event("first_byte")

                buffer += chunk_bytes.decode("utf-8", errors="replace")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()

                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            if rec is not None:
                                rec.event("done")
                            for event in converter.finish_events("stop"):
                                if not await _safe_write(event.encode()):
                                    return response
                            continue

                        try:
                            chunk = json.loads(data_str)
                            if rec is not None and isinstance(chunk, dict):
                                choices = chunk.get("choices")
                                if isinstance(choices, list) and choices:
                                    fr = choices[0].get("finish_reason") if isinstance(choices[0], dict) else None
                                    if fr:
                                        rec.event("finish_reason", value=fr)
                                if chunk.get("usage"):
                                    rec.event("usage", usage=chunk["usage"])
                            for event in converter.feed_chunk(chunk):
                                if not await _safe_write(event.encode()):
                                    return response
                        except json.JSONDecodeError:
                            continue

            if not client_disconnected and buffer.strip():
                if rec is not None:
                    rec.event("flush_trailing", remaining_len=len(buffer.strip()))
                line = buffer.strip()
                if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        for event in converter.feed_chunk(chunk):
                            if not await _safe_write(event.encode()):
                                return response
                    except json.JSONDecodeError:
                        pass

            if not client_disconnected:
                try:
                    await response.write_eof()
                except BaseException as exc:
                    if _is_disconnect(exc):
                        logger.debug("Client disconnected during stream close")
                    else:
                        raise
            if rec is not None:
                rec.close("ok", client_disconnected=client_disconnected)
        except asyncio.CancelledError:
            logger.debug("Stream cancelled by client")
            if rec is not None:
                rec.close("cancelled")
        except BaseException as exc:
            if _is_disconnect(exc):
                logger.debug("Client disconnected during stream")
            else:
                logger.error("Stream error: %s", exc)
            if rec is not None:
                rec.close("client_disconnected" if _is_disconnect(exc) else "error",
                          error=type(exc).__name__)

        return response


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def run_proxy_until_done(
    api_url: str,
    api_key: str,
    model: str,
    port: int = 0,
    max_output_tokens: int | None = None,
    debug: DebugTarget | None = None,
    debug_keep: int = 100,
    profile_name: str = "",
) -> tuple[ProxyServer, int]:
    """Start proxy and return (server, port). Caller is responsible for stopping."""
    server = ProxyServer(
        api_url=api_url, api_key=api_key, model=model, port=port,
        max_output_tokens=max_output_tokens, debug=debug, debug_keep=debug_keep,
        profile_name=profile_name,
    )
    actual_port = await server.start()
    return server, actual_port
