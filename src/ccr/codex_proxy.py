"""Codex proxy: OpenAI Responses API -> OpenAI Chat Completions API.

Serves /v1/responses for the Codex client, translating to Chat Completions
against a Chat-only upstream (e.g. SGLang), and translating the response back.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from typing import Any

from aiohttp import web, ClientSession, ClientTimeout
from aiohttp.client_exceptions import ClientConnectionResetError as _ClientConnReset

from .codex_converter import (
    CodexToolContext,
    build_codex_tool_context_from_request,
    chat_completion_to_response,
    chat_error_to_response_error,
    responses_to_chat_completions,
)
from .codex_stream import ChatToResponsesState, extract_chat_sse_error, sse_event
from .debug_log import DebugTarget, RequestRecorder, make_recorder

logger = logging.getLogger(__name__)


def _is_disconnect(exc: BaseException) -> bool:
    return isinstance(exc, (
        ConnectionResetError,
        BrokenPipeError,
        ConnectionAbortedError,
        _ClientConnReset,
    )) or "Cannot write to closing transport" in str(exc)


class CodexProxyServer:
    """Proxy that speaks Responses API to the client and Chat Completions upstream."""

    def __init__(
        self,
        api_url: str,
        api_key: str = "dummy",
        model: str = "",
        port: int = 0,
        debug: DebugTarget | None = None,
        debug_keep: int = 100,
        profile_name: str = "",
    ):
        # api_url is the full chat completions URL (e.g. http://host/v1/chat/completions)
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.port = port
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
        app.router.add_post("/v1/responses", self._handle_responses)
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

    async def _handle_responses(self, request: web.Request) -> web.StreamResponse:
        try:
            return await self._handle_responses_inner(request)
        except asyncio.CancelledError:
            logger.debug("Request cancelled by client")
            raise
        except BaseException as exc:
            if _is_disconnect(exc):
                logger.debug("Client disconnected")
                return web.Response(status=499)
            raise

    async def _handle_responses_inner(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        is_stream = body.get("stream", False)

        rec = make_recorder(
            self.debug, self.profile_name, "responses", self.debug_keep, "codex"
        )
        if rec is not None:
            rec.log_request_body(body)

        tool_context = build_codex_tool_context_from_request(body)
        chat_req = responses_to_chat_completions(body)
        if self.model:
            chat_req["model"] = self.model

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        timeout = ClientTimeout(total=600, sock_read=300)

        async with ClientSession(timeout=timeout) as session:
            async with session.post(self.api_url, json=chat_req, headers=headers) as upstream:
                if upstream.status != 200:
                    error_body = await upstream.text()
                    if rec is not None:
                        rec.log_upstream_chunk(error_body.encode("utf-8", errors="replace"))
                        rec.event("upstream_error", status=upstream.status)
                    try:
                        error_json = json.loads(error_body)
                        if rec is not None:
                            rec.close("upstream_error", status=upstream.status)
                        return web.json_response(
                            chat_error_to_response_error(error_json),
                            status=upstream.status,
                        )
                    except (json.JSONDecodeError, ValueError):
                        if rec is not None:
                            rec.close("upstream_error", status=upstream.status)
                        return web.json_response(
                            chat_error_to_response_error(error_body),
                            status=upstream.status,
                        )

                if is_stream:
                    return await self._stream_response(request, upstream, tool_context, rec)
                else:
                    resp_data = await upstream.json()
                    if rec is not None:
                        rec.log_upstream_chunk(json.dumps(resp_data, ensure_ascii=False).encode())
                        rec.close("non_stream_ok")
                    responses_resp = chat_completion_to_response(resp_data, tool_context)
                    return web.json_response(responses_resp)

    async def _stream_response(
        self, request: web.Request, upstream: Any, tool_context: CodexToolContext,
        rec: RequestRecorder | None = None,
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

        state = ChatToResponsesState(tool_context=tool_context)
        client_disconnected = False
        stream_failed = False

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

        buffer = ""
        first_byte_logged = False
        try:
            async for chunk_bytes in upstream.content.iter_any():
                if client_disconnected:
                    break

                if rec is not None:
                    rec.log_upstream_chunk(chunk_bytes)
                    if not first_byte_logged:
                        first_byte_logged = True
                        rec.event("first_byte")

                buffer += chunk_bytes.decode("utf-8", errors="replace")

                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    event = _parse_sse_block(block)
                    if event is None:
                        continue
                    event_name, data_str = event

                    if data_str.strip() == "[DONE]":
                        if rec is not None:
                            rec.event("done")
                        for ev in state.finalize():
                            if not await _safe_write(ev):
                                return response
                        continue

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if event_name == "error" or (isinstance(chunk, dict) and chunk.get("error") is not None):
                        message, error_type = extract_chat_sse_error(chunk)
                        if rec is not None:
                            rec.event("upstream_sse_error", message=message, error_type=error_type)
                        await _safe_write(state.failed_event(message, error_type))
                        stream_failed = True
                        break

                    if rec is not None:
                        choices = chunk.get("choices") if isinstance(chunk, dict) else None
                        if isinstance(choices, list) and choices:
                            fr = choices[0].get("finish_reason") if isinstance(choices[0], dict) else None
                            if fr:
                                rec.event("finish_reason", value=fr)
                            usage = chunk.get("usage")
                            if usage:
                                rec.event("usage", usage=usage)

                    for ev in state.handle_chat_chunk(chunk):
                        if not await _safe_write(ev):
                            return response

                if stream_failed:
                    break

            # flush trailing block
            if not client_disconnected and not stream_failed and buffer.strip():
                if rec is not None:
                    rec.event("flush_trailing", remaining_len=len(buffer.strip()))
                event = _parse_sse_block(buffer.strip())
                if event is not None:
                    _, data_str = event
                    if data_str.strip() != "[DONE]":
                        try:
                            chunk = json.loads(data_str)
                            if isinstance(chunk, dict) and chunk.get("error") is None:
                                for ev in state.handle_chat_chunk(chunk):
                                    if not await _safe_write(ev):
                                        return response
                        except json.JSONDecodeError:
                            pass

            if not client_disconnected and not stream_failed:
                if state.completed or state.finish_reason is not None:
                    if rec is not None:
                        rec.event("finalize", reason="completed_or_finish")
                    for ev in state.finalize():
                        if not await _safe_write(ev):
                            return response
                elif state.has_substantive_output():
                    state.finish_reason = "length"
                    if rec is not None:
                        rec.event("finalize", reason="length_fallback")
                    for ev in state.finalize():
                        if not await _safe_write(ev):
                            return response
                else:
                    if rec is not None:
                        rec.event("stream_truncated", reason="no_finish_no_output")
                    await _safe_write(
                        state.failed_event(
                            "Upstream Chat Completions stream ended before sending finish_reason",
                            "stream_truncated",
                        )
                    )

            if not client_disconnected:
                try:
                    await response.write_eof()
                except BaseException as exc:
                    if _is_disconnect(exc):
                        logger.debug("Client disconnected during stream close")
                    else:
                        raise
        except asyncio.CancelledError:
            logger.debug("Stream cancelled by client")
            if rec is not None:
                rec.close("cancelled")
            return response
        except BaseException as exc:
            if _is_disconnect(exc):
                logger.debug("Client disconnected during stream")
            else:
                logger.error("Stream error: %s", exc)
            if rec is not None:
                rec.close("client_disconnected" if _is_disconnect(exc) else "error",
                          error=type(exc).__name__)
            return response

        if rec is not None:
            rec.close("ok", client_disconnected=client_disconnected,
                      finish_reason=state.finish_reason)
        return response


def _parse_sse_block(block: str) -> tuple[str | None, str] | None:
    """Parse one SSE block into (event_name, data). Returns None if no data."""
    event_name: str | None = None
    data_parts: list[str] = []
    for line in block.split("\n"):
        line = line.rstrip("\r")
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_parts.append(line[5:].lstrip())
    if not data_parts:
        return None
    return event_name, "\n".join(data_parts)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def run_codex_proxy_until_done(
    api_url: str,
    api_key: str,
    model: str,
    port: int = 0,
    debug: DebugTarget | None = None,
    debug_keep: int = 100,
    profile_name: str = "",
) -> tuple[CodexProxyServer, int]:
    server = CodexProxyServer(
        api_url=api_url, api_key=api_key, model=model, port=port,
        debug=debug, debug_keep=debug_keep, profile_name=profile_name,
    )
    actual_port = await server.start()
    return server, actual_port
