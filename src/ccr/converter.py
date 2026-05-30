"""Anthropic Messages API <-> OpenAI Chat Completions API format conversion."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any


# ── Anthropic request → OpenAI request ──────────────────────────────

def anthropic_to_openai_request(body: dict[str, Any], model_override: str = "", max_output_tokens: int | None = None) -> dict[str, Any]:
    messages = _convert_messages(body.get("system"), body.get("messages", []))
    tools = _convert_tools(body.get("tools"))

    req: dict[str, Any] = {
        "model": model_override or body.get("model", ""),
        "messages": messages,
        "stream": body.get("stream", False),
    }

    if body.get("stream"):
        req["stream_options"] = {"include_usage": True}

    max_tokens = body.get("max_tokens")
    if max_tokens and max_output_tokens:
        max_tokens = min(max_tokens, max_output_tokens)
    if max_tokens:
        req["max_tokens"] = max_tokens
    if body.get("temperature") is not None:
        req["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        req["top_p"] = body["top_p"]
    if tools:
        req["tools"] = tools

    return req


def _convert_messages(
    system: str | list | None,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    if system:
        sys_text = system if isinstance(system, str) else _blocks_to_text(system)
        out.append({"role": "system", "content": sys_text})

    for msg in messages:
        role = msg["role"]
        content = msg.get("content")

        if role == "assistant":
            converted = _convert_assistant_message(content)
            out.append(converted)
        elif role == "user":
            out.append(_convert_user_message(content))
        else:
            out.append({"role": role, "content": _to_text(content)})

    return out


def _convert_assistant_message(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {"role": "assistant", "content": content}

    text_parts = []
    tool_calls = []

    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
        elif block.get("type") == "text":
            text_parts.append(block["text"])
        elif block.get("type") == "thinking":
            pass
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block["id"],
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": json.dumps(block["input"]) if isinstance(block["input"], dict) else str(block["input"]),
                },
            })

    msg: dict[str, Any] = {"role": "assistant"}
    if text_parts:
        msg["content"] = "\n".join(text_parts)
    if tool_calls:
        msg["tool_calls"] = tool_calls
        if "content" not in msg:
            msg["content"] = None

    return msg


def _convert_user_message(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {"role": "user", "content": content}

    # Check for tool_result blocks → OpenAI tool messages
    tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
    if tool_results:
        # Return multiple messages for tool results
        msgs = []
        for tr in tool_results:
            tr_content = tr.get("content", "")
            if isinstance(tr_content, list):
                tr_content = _blocks_to_text(tr_content)
            msgs.append({
                "role": "tool",
                "tool_call_id": tr["tool_use_id"],
                "content": str(tr_content),
            })
        # If there's only tool results, return as-is
        # The caller handles the flattening
        if len(content) == len(tool_results):
            return msgs  # type: ignore  # special case: returns list
        # Mix of tool results and other content — append user text too
        text_parts = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]
        if text_parts:
            msgs.append({"role": "user", "content": _blocks_to_text(text_parts)})
        return msgs  # type: ignore

    return {"role": "user", "content": _blocks_to_text(content)}


def _convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        }
        for t in tools
    ]


def _blocks_to_text(blocks: list | str) -> str:
    if isinstance(blocks, str):
        return blocks
    parts = []
    for b in blocks:
        if isinstance(b, str):
            parts.append(b)
        elif isinstance(b, dict) and b.get("type") == "text":
            parts.append(b["text"])
    return "\n".join(parts)


def _to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _blocks_to_text(content)
    return str(content)


# ── _convert_messages post-processing: flatten lists ────────────────

_orig_convert_messages = _convert_messages

def _convert_messages_flat(
    system: str | list | None,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw = _orig_convert_messages(system, messages)
    flat: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat

# Patch to use the flattening version
_convert_messages = _convert_messages_flat  # type: ignore


# ── OpenAI non-streaming response → Anthropic response ─────────────

def openai_to_anthropic_response(resp: dict[str, Any], model: str = "") -> dict[str, Any]:
    choice = resp["choices"][0]
    msg = choice["message"]
    content_blocks = []

    if msg.get("reasoning_content"):
        content_blocks.append({
            "type": "thinking",
            "thinking": msg["reasoning_content"],
            "signature": str(int(time.time() * 1000)),
        })

    if msg.get("content"):
        content_blocks.append({"type": "text", "text": msg["content"]})

    if msg.get("tool_calls"):
        for tc in msg["tool_calls"]:
            try:
                args = json.loads(tc["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                args = tc["function"]["arguments"]
            content_blocks.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["function"]["name"],
                "input": args,
            })

    stop_reason = _map_finish_reason(choice.get("finish_reason"))
    usage = resp.get("usage", {})
    details = usage.get("prompt_tokens_details") or {}

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model or resp.get("model", ""),
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0) - details.get("cached_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "cache_read_input_tokens": details.get("cached_tokens", 0),
        },
    }


# ── OpenAI SSE stream → Anthropic SSE events ───────────────────────

class StreamConverter:
    """Converts a stream of OpenAI SSE chunks into Anthropic SSE events."""

    def __init__(self, model: str = ""):
        self.model = model
        self._started = False
        self._block_started = False
        self._tool_call_buffers: dict[int, dict[str, Any]] = {}
        self._current_tool_index: int | None = None
        self._text_emitted = False
        self._thinking_started = False
        self._thinking_ended = False
        self._next_block_index = 0
        self._finished = False
        self._final_usage: dict[str, Any] | None = None
        self._finish_reason: str | None = None

    def _close_thinking_events(self) -> list[str]:
        """Emit signature_delta + content_block_stop for the thinking block."""
        events = [
            _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": self._next_block_index,
                "delta": {"type": "signature_delta", "signature": str(int(time.time() * 1000))},
            }),
            _sse("content_block_stop", {
                "type": "content_block_stop",
                "index": self._next_block_index,
            }),
        ]
        self._thinking_ended = True
        self._next_block_index += 1
        return events

    def start_events(self) -> list[str]:
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        self._msg_id = msg_id
        self._started = True
        return [
            _sse("message_start", {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0},
                },
            }),
        ]

    def feed_chunk(self, chunk: dict[str, Any]) -> list[str]:
        events: list[str] = []

        if not self._started:
            events.extend(self.start_events())

        choices = chunk.get("choices", [])
        if not choices:
            # Capture usage from the final streaming chunk (sent when stream_options.include_usage is true)
            chunk_usage = chunk.get("usage")
            if chunk_usage:
                self._final_usage = chunk_usage
            # If finish was pending, emit closing events now that we have usage data
            if self._finish_reason:
                events.extend(self.finish_events(self._finish_reason))
            return events

        delta = choices[0].get("delta", {})
        finish_reason = choices[0].get("finish_reason")

        # Reasoning/thinking content
        if delta.get("reasoning_content"):
            if not self._thinking_started:
                self._thinking_started = True
                self._block_started = True
                events.append(_sse("content_block_start", {
                    "type": "content_block_start",
                    "index": self._next_block_index,
                    "content_block": {"type": "thinking", "thinking": ""},
                }))
            events.append(_sse("content_block_delta", {
                "type": "content_block_delta",
                "index": self._next_block_index,
                "delta": {"type": "thinking_delta", "thinking": delta["reasoning_content"]},
            }))

        # Text content
        if delta.get("content"):
            # Close thinking block if transitioning from reasoning to content
            if self._thinking_started and not self._thinking_ended:
                events.extend(self._close_thinking_events())

            if not self._text_emitted:
                self._text_emitted = True
                events.append(_sse("content_block_start", {
                    "type": "content_block_start",
                    "index": self._next_block_index,
                    "content_block": {"type": "text", "text": ""},
                }))
            events.append(_sse("content_block_delta", {
                "type": "content_block_delta",
                "index": self._next_block_index,
                "delta": {"type": "text_delta", "text": delta["content"]},
            }))

        # Tool calls
        if delta.get("tool_calls"):
            for tc in delta["tool_calls"]:
                idx = tc.get("index", 0)
                if idx not in self._tool_call_buffers:
                    # Close thinking block if still open
                    if self._thinking_started and not self._thinking_ended:
                        events.extend(self._close_thinking_events())
                    # Close text block if open
                    if self._text_emitted and self._current_tool_index is None:
                        events.append(_sse("content_block_stop", {"type": "content_block_stop", "index": self._next_block_index}))
                        self._next_block_index += 1

                    self._tool_call_buffers[idx] = {
                        "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": "",
                    }
                    self._current_tool_index = idx
                    block_index = self._next_block_index + len(self._tool_call_buffers) - 1
                    self._tool_call_buffers[idx]["_block_index"] = block_index
                    events.append(_sse("content_block_start", {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": self._tool_call_buffers[idx]["id"],
                            "name": self._tool_call_buffers[idx]["name"],
                            "input": {},
                        },
                    }))

                if tc.get("function", {}).get("arguments"):
                    self._tool_call_buffers[idx]["arguments"] += tc["function"]["arguments"]
                    events.append(_sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": self._tool_call_buffers[idx]["_block_index"],
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": tc["function"]["arguments"],
                        },
                    }))

        # Finish — don't emit message_delta yet; the usage-only chunk
        # (from stream_options.include_usage) may arrive after finish_reason.
        if finish_reason:
            self._finish_reason = finish_reason

        return events

    def finish_events(self, finish_reason: str | None = None, usage: dict | None = None) -> list[str]:
        if self._finished:
            return []
        self._finished = True
        events: list[str] = []

        # Close open thinking block
        if self._thinking_started and not self._thinking_ended:
            events.extend(self._close_thinking_events())

        # Close open text block
        if self._text_emitted and self._current_tool_index is None:
            events.append(_sse("content_block_stop", {"type": "content_block_stop", "index": self._next_block_index}))
        for idx, buf in self._tool_call_buffers.items():
            events.append(_sse("content_block_stop", {"type": "content_block_stop", "index": buf["_block_index"]}))

        stop_reason = _map_finish_reason(finish_reason)
        final = self._final_usage or usage
        output_tokens = 0
        input_tokens = 0
        cache_read = 0
        if final:
            output_tokens = final.get("completion_tokens", 0)
            details = final.get("prompt_tokens_details") or {}
            input_tokens = final.get("prompt_tokens", 0) - details.get("cached_tokens", 0)
            cache_read = details.get("cached_tokens", 0)

        events.append(_sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens, "cache_read_input_tokens": cache_read},
        }))
        events.append(_sse("message_stop", {"type": "message_stop"}))
        return events


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _map_finish_reason(reason: str | None) -> str:
    return {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }.get(reason or "stop", "end_turn")
