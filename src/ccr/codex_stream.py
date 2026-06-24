"""Streaming Chat Completions SSE -> Responses SSE conversion.

Ported from cc-switch's streaming_codex_chat.rs (ChatToResponsesState).
A state machine that consumes Chat Completions stream chunks and emits
Responses API SSE events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .codex_converter import (
    CodexToolContext,
    attach_optional_reasoning_content_field,
    canonicalize_tool_arguments_str,
    chat_usage_to_responses_usage,
    custom_tool_input_from_chat_arguments,
    extract_reasoning_field_text,
    response_id_from_chat_id,
    response_status_from_finish_reason,
    response_tool_call_item_from_chat_name,
    response_tool_call_item_id_from_chat_name,
    split_leading_think_block,
    strip_leading_think_open_tag,
    THINK_OPEN_TAG,
)


def sse_event(event: str, data: Any) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


@dataclass
class TextItemState:
    output_index: int | None = None
    item_id: str = ""
    text: str = ""
    added: bool = False
    done: bool = False


@dataclass
class ReasoningItemState:
    output_index: int | None = None
    item_id: str = ""
    text: str = ""
    added: bool = False
    done: bool = False


@dataclass
class ToolCallState:
    output_index: int | None = None
    item_id: str = ""
    call_id: str = ""
    name: str = ""
    arguments: str = ""
    reasoning_content: str = ""
    added: bool = False
    done: bool = False


class InlineThinkMode:
    DETECTING = "detecting"
    REASONING = "reasoning"
    TEXT = "text"


@dataclass
class InlineThinkState:
    mode: str = InlineThinkMode.DETECTING
    buffer: str = ""


class ThinkPrefixDecision:
    NEED_MORE = "need_more"
    REASONING = "reasoning"
    TEXT = "text"


def leading_think_prefix_decision(buffer: str) -> str:
    trimmed = buffer.lstrip()
    if not trimmed:
        return ThinkPrefixDecision.NEED_MORE
    if trimmed.startswith(THINK_OPEN_TAG):
        return ThinkPrefixDecision.REASONING
    if THINK_OPEN_TAG.startswith(trimmed):
        return ThinkPrefixDecision.NEED_MORE
    return ThinkPrefixDecision.TEXT


def chat_delta_reasoning_text(delta: dict) -> str | None:
    return extract_reasoning_field_text(delta)


class ChatToResponsesState:
    """State machine: feed Chat Completions chunks, produce Responses SSE events."""

    def __init__(self, tool_context: CodexToolContext | None = None):
        self.tool_context = tool_context or CodexToolContext()
        self.response_started = False
        self.completed = False
        self.response_id = "resp_ccswitch"
        self.model = ""
        self.created_at = 0
        self.next_output_index = 0
        self.text = TextItemState()
        self.reasoning = ReasoningItemState()
        self.inline_think = InlineThinkState()
        self.tools: dict[int, ToolCallState] = {}
        self.output_items: list[tuple[int, dict]] = []
        self.latest_usage: dict | None = None
        self.finish_reason: str | None = None

    # ---- public ----

    def handle_chat_chunk(self, chunk: dict) -> list[bytes]:
        events: list[bytes] = []

        cid = chunk.get("id")
        if isinstance(cid, str):
            self.response_id = response_id_from_chat_id(cid)
        model = chunk.get("model")
        if isinstance(model, str) and model:
            self.model = model
        created = chunk.get("created")
        if isinstance(created, int):
            self.created_at = created

        events.extend(self.ensure_response_started())

        usage = chunk.get("usage")
        if isinstance(usage, dict) and usage:
            self.latest_usage = chat_usage_to_responses_usage(usage)

        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return events
        choice = choices[0]
        if not isinstance(choice, dict):
            return events

        delta = choice.get("delta")
        if isinstance(delta, dict):
            reasoning = chat_delta_reasoning_text(delta)
            if reasoning:
                events.extend(self.push_reasoning_delta(reasoning))
                self.append_reasoning_to_active_tools(reasoning)

            content = delta.get("content")
            if isinstance(content, str) and content:
                events.extend(self.push_content_delta(content))

            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list):
                events.extend(self.flush_inline_think_at_boundary())
                reasoning_for_tool_call = self.current_reasoning_text()
                events.extend(self.finalize_reasoning())
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        events.extend(
                            self.push_tool_call_delta(tool_call, reasoning_for_tool_call)
                        )

        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str):
            self.finish_reason = finish_reason

        return events

    def finalize(self) -> list[bytes]:
        if self.completed:
            return []
        events: list[bytes] = []
        events.extend(self.ensure_response_started())
        events.extend(self.flush_inline_think_at_boundary())
        events.extend(self.finalize_reasoning())
        events.extend(self.finalize_text())
        events.extend(self.finalize_tools())

        status = response_status_from_finish_reason(self.finish_reason)
        response = self.base_response(status, self.completed_output_items())
        if status == "incomplete":
            response["incomplete_details"] = {"reason": "max_output_tokens"}

        events.append(sse_event("response.completed", {"type": "response.completed", "response": response}))
        self.completed = True
        return events

    def failed_event(self, message: str, error_type: str | None) -> bytes:
        self.completed = True
        error: dict = {"message": message}
        if error_type:
            error["type"] = error_type
        response = self.base_response("failed", self.completed_output_items())
        response["error"] = error
        return sse_event("response.failed", {"type": "response.failed", "response": response})

    def has_substantive_output(self) -> bool:
        return bool(
            self.text.text.strip()
            or self.reasoning.text.strip()
            or self.inline_think.buffer.strip()
            or self.output_items
            or any(
                s.added
                or s.call_id.strip()
                or s.name.strip()
                or s.arguments.strip()
                or s.reasoning_content.strip()
                for s in self.tools.values()
            )
        )

    # ---- internals ----

    def _next_output_index(self) -> int:
        idx = self.next_output_index
        self.next_output_index += 1
        return idx

    def base_response(self, status: str, output: list[dict]) -> dict:
        usage = self.latest_usage
        if usage is None:
            usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "output_tokens_details": {"reasoning_tokens": 0},
            }
        return {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.model,
            "output": output,
            "usage": usage,
        }

    def ensure_response_started(self) -> list[bytes]:
        if self.response_started:
            return []
        self.response_started = True
        response = self.base_response("in_progress", [])
        return [
            sse_event("response.created", {"type": "response.created", "response": response}),
            sse_event(
                "response.in_progress",
                {"type": "response.in_progress", "response": self.base_response("in_progress", [])},
            ),
        ]

    def push_reasoning_delta(self, delta: str) -> list[bytes]:
        events: list[bytes] = []
        if not self.reasoning.added:
            output_index = self._next_output_index()
            item_id = f"rs_{self.response_id}"
            self.reasoning.output_index = output_index
            self.reasoning.item_id = item_id
            self.reasoning.added = True
            events.append(
                sse_event(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": {"id": item_id, "type": "reasoning", "status": "in_progress", "summary": []},
                    },
                )
            )
            events.append(
                sse_event(
                    "response.reasoning_summary_part.added",
                    {
                        "type": "response.reasoning_summary_part.added",
                        "item_id": self.reasoning.item_id,
                        "output_index": output_index,
                        "summary_index": 0,
                        "part": {"type": "summary_text", "text": ""},
                    },
                )
            )
        self.reasoning.text += delta
        output_index = self.reasoning.output_index or 0
        events.append(
            sse_event(
                "response.reasoning_summary_text.delta",
                {
                    "type": "response.reasoning_summary_text.delta",
                    "item_id": self.reasoning.item_id,
                    "output_index": output_index,
                    "summary_index": 0,
                    "delta": delta,
                },
            )
        )
        return events

    def push_text_delta(self, delta: str) -> list[bytes]:
        events: list[bytes] = []
        if not self.text.added:
            output_index = self._next_output_index()
            item_id = f"{self.response_id}_msg"
            self.text.output_index = output_index
            self.text.item_id = item_id
            self.text.added = True
            events.append(
                sse_event(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": {
                            "id": item_id,
                            "type": "message",
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        },
                    },
                )
            )
            events.append(
                sse_event(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "item_id": self.text.item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    },
                )
            )
        self.text.text += delta
        output_index = self.text.output_index or 0
        events.append(
            sse_event(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": self.text.item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "delta": delta,
                },
            )
        )
        return events

    def push_content_delta(self, delta: str) -> list[bytes]:
        mode = self.inline_think.mode
        if mode == InlineThinkMode.TEXT:
            events = self.finalize_reasoning()
            events.extend(self.push_text_delta(delta))
            return events
        if mode == InlineThinkMode.DETECTING:
            self.inline_think.buffer += delta
            decision = leading_think_prefix_decision(self.inline_think.buffer)
            if decision == ThinkPrefixDecision.NEED_MORE:
                return []
            if decision == ThinkPrefixDecision.REASONING:
                self.inline_think.mode = InlineThinkMode.REASONING
                return self.drain_complete_inline_think()
            # TEXT
            self.inline_think.mode = InlineThinkMode.TEXT
            text = self.inline_think.buffer
            self.inline_think.buffer = ""
            events = self.finalize_reasoning()
            events.extend(self.push_text_delta(text))
            return events
        # REASONING
        self.inline_think.buffer += delta
        return self.drain_complete_inline_think()

    def drain_complete_inline_think(self) -> list[bytes]:
        split = split_leading_think_block(self.inline_think.buffer)
        if split is None:
            return []
        reasoning, answer = split
        self.inline_think.mode = InlineThinkMode.TEXT
        self.inline_think.buffer = ""
        events: list[bytes] = []
        if reasoning:
            events.extend(self.push_reasoning_delta(reasoning))
            events.extend(self.finalize_reasoning())
        if answer:
            events.extend(self.push_text_delta(answer))
        return events

    def flush_inline_think_at_boundary(self) -> list[bytes]:
        mode = self.inline_think.mode
        if mode == InlineThinkMode.TEXT:
            return []
        if mode == InlineThinkMode.DETECTING:
            self.inline_think.mode = InlineThinkMode.TEXT
            text = self.inline_think.buffer
            self.inline_think.buffer = ""
            if not text:
                return []
            events = self.finalize_reasoning()
            events.extend(self.push_text_delta(text))
            return events
        # REASONING
        buffered = self.inline_think.buffer
        self.inline_think.buffer = ""
        self.inline_think.mode = InlineThinkMode.TEXT
        split = split_leading_think_block(buffered)
        events: list[bytes] = []
        if split is not None:
            reasoning, answer = split
            if reasoning:
                events.extend(self.push_reasoning_delta(reasoning))
                events.extend(self.finalize_reasoning())
            if answer:
                events.extend(self.push_text_delta(answer))
            return events
        reasoning = strip_leading_think_open_tag(buffered) or buffered
        if not reasoning:
            return []
        events.extend(self.push_reasoning_delta(reasoning))
        events.extend(self.finalize_reasoning())
        return events

    def current_reasoning_text(self) -> str | None:
        t = self.reasoning.text.strip()
        return t or None

    def append_reasoning_to_active_tools(self, delta: str) -> None:
        if not delta.strip():
            return
        for state in self.tools.values():
            if state.done:
                continue
            if not state.reasoning_content:
                state.reasoning_content = delta.lstrip()
            else:
                state.reasoning_content += delta

    def push_tool_call_delta(self, tool_call: dict, reasoning: str | None) -> list[bytes]:
        chat_index = tool_call.get("index")
        if not isinstance(chat_index, int):
            chat_index = 0

        id_delta = tool_call.get("id")
        if not isinstance(id_delta, str):
            id_delta = None
        fn = tool_call.get("function") or {}
        if not isinstance(fn, dict):
            fn = {}
        name_delta = fn.get("name")
        if not isinstance(name_delta, str):
            name_delta = None
        args_delta = fn.get("arguments")
        if not isinstance(args_delta, str):
            args_delta = ""

        state = self.tools.setdefault(chat_index, ToolCallState())
        if id_delta:
            state.call_id = id_delta
        if name_delta:
            if name_delta:
                state.name = name_delta
        if args_delta:
            state.arguments += args_delta
        if not state.reasoning_content and reasoning and reasoning.strip():
            state.reasoning_content = reasoning.strip()

        should_add = False
        if not state.added and state.call_id and state.name:
            should_add = True

        current_name = state.name
        is_custom_tool = self.tool_context.is_custom_tool_chat_name(current_name)
        events: list[bytes] = []

        if should_add:
            assigned = self._next_output_index()
            state.added = True
            if not state.call_id:
                state.call_id = f"call_{chat_index}"
            state.output_index = assigned
            state.item_id = response_tool_call_item_id_from_chat_name(
                state.call_id, state.name, self.tool_context
            )
            item = response_tool_call_item_from_chat_name(
                state.item_id, "in_progress", state.call_id, state.name, "", state.reasoning_content, self.tool_context
            )
            events.append(
                sse_event(
                    "response.output_item.added",
                    {"type": "response.output_item.added", "output_index": assigned, "item": item},
                )
            )
            pending_arguments = state.arguments
            if pending_arguments and not is_custom_tool:
                events.append(
                    sse_event(
                        "response.function_call_arguments.delta",
                        {
                            "type": "response.function_call_arguments.delta",
                            "item_id": state.item_id,
                            "output_index": assigned,
                            "delta": pending_arguments,
                        },
                    )
                )
        elif args_delta and not is_custom_tool:
            output_index = state.output_index
            if output_index is not None and state.added:
                events.append(
                    sse_event(
                        "response.function_call_arguments.delta",
                        {
                            "type": "response.function_call_arguments.delta",
                            "item_id": state.item_id,
                            "output_index": output_index,
                            "delta": args_delta,
                        },
                    )
                )
        return events

    def finalize_reasoning(self) -> list[bytes]:
        if not self.reasoning.added or self.reasoning.done:
            return []
        output_index = self.reasoning.output_index or 0
        item = {
            "id": self.reasoning.item_id,
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": self.reasoning.text}],
        }
        self.output_items.append((output_index, item))
        self.reasoning.done = True
        return [
            sse_event(
                "response.reasoning_summary_text.done",
                {
                    "type": "response.reasoning_summary_text.done",
                    "item_id": self.reasoning.item_id,
                    "output_index": output_index,
                    "summary_index": 0,
                    "text": self.reasoning.text,
                },
            ),
            sse_event(
                "response.reasoning_summary_part.done",
                {
                    "type": "response.reasoning_summary_part.done",
                    "item_id": self.reasoning.item_id,
                    "output_index": output_index,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": self.reasoning.text},
                },
            ),
            sse_event(
                "response.output_item.done",
                {"type": "response.output_item.done", "output_index": output_index, "item": item},
            ),
        ]

    def finalize_text(self) -> list[bytes]:
        if not self.text.added or self.text.done:
            return []
        output_index = self.text.output_index or 0
        item = {
            "id": self.text.item_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": self.text.text, "annotations": []}],
        }
        self.output_items.append((output_index, item))
        self.text.done = True
        return [
            sse_event(
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "item_id": self.text.item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "text": self.text.text,
                },
            ),
            sse_event(
                "response.content_part.done",
                {
                    "type": "response.content_part.done",
                    "item_id": self.text.item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": self.text.text, "annotations": []},
                },
            ),
            sse_event(
                "response.output_item.done",
                {"type": "response.output_item.done", "output_index": output_index, "item": item},
            ),
        ]

    def finalize_tools(self) -> list[bytes]:
        events: list[bytes] = []
        for key in sorted(self.tools.keys()):
            state = self.tools[key]
            if state.done:
                continue
            if not state.name:
                state.done = True
                continue
            if not state.added and not state.done:
                assigned = self._next_output_index()
                state.added = True
                if not state.call_id:
                    state.call_id = f"call_{key}"
                state.output_index = assigned
                state.item_id = response_tool_call_item_id_from_chat_name(
                    state.call_id, state.name, self.tool_context
                )
                item = response_tool_call_item_from_chat_name(
                    state.item_id, "in_progress", state.call_id, state.name, "", state.reasoning_content, self.tool_context
                )
                events.append(
                    sse_event(
                        "response.output_item.added",
                        {"type": "response.output_item.added", "output_index": assigned, "item": item},
                    )
                )

            output_index = state.output_index or 0
            arguments = canonicalize_tool_arguments_str(state.arguments)
            is_custom_tool = self.tool_context.is_custom_tool_chat_name(state.name)
            item = response_tool_call_item_from_chat_name(
                state.item_id, "completed", state.call_id, state.name, arguments, state.reasoning_content, self.tool_context
            )
            state.done = True
            self.output_items.append((output_index, item))

            if is_custom_tool:
                input_str = custom_tool_input_from_chat_arguments(arguments)
                if input_str:
                    events.append(
                        sse_event(
                            "response.custom_tool_call_input.delta",
                            {
                                "type": "response.custom_tool_call_input.delta",
                                "item_id": state.item_id,
                                "output_index": output_index,
                                "delta": input_str,
                            },
                        )
                    )
                events.append(
                    sse_event(
                        "response.custom_tool_call_input.done",
                        {
                            "type": "response.custom_tool_call_input.done",
                            "item_id": state.item_id,
                            "output_index": output_index,
                            "input": input_str,
                        },
                    )
                )
            else:
                events.append(
                    sse_event(
                        "response.function_call_arguments.done",
                        {
                            "type": "response.function_call_arguments.done",
                            "item_id": state.item_id,
                            "output_index": output_index,
                            "arguments": arguments,
                        },
                    )
                )
            events.append(
                sse_event(
                    "response.output_item.done",
                    {"type": "response.output_item.done", "output_index": output_index, "item": item},
                )
            )
        return events

    def completed_output_items(self) -> list[dict]:
        return [item for _, item in sorted(self.output_items, key=lambda x: x[0])]


def extract_chat_sse_error(value: dict) -> tuple[str, str | None]:
    error = value.get("error", value)
    if not isinstance(error, dict):
        error = {}
    message = (
        error.get("message")
        or error.get("detail")
    )
    if message is None:
        message = json.dumps(error, ensure_ascii=False)
    error_type = error.get("type") or error.get("code")
    if not isinstance(error_type, str):
        error_type = None
    return message, error_type
