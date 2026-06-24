"""Codex Responses API <-> OpenAI Chat Completions conversion.

Ported from cc-switch's transform_codex_chat.rs / streaming_codex_chat.rs.
Used when the Codex client (speaking Responses API) talks to a Chat-only
upstream (e.g. SGLang).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# --- tool-context constants (mirrors transform_codex_chat.rs) ---
TOOL_SEARCH_PROXY_NAME = "tool_search"
CUSTOM_TOOL_INPUT_FIELD = "input"
CHAT_TOOL_NAME_MAX_LEN = 64
CUSTOM_TOOL_INPUT_DESCRIPTION = (
    "Raw string input for the original custom tool. Preserve formatting exactly "
    "and follow the original tool definition embedded in the description."
)
CUSTOM_TOOL_PRESERVED_METADATA_HEADING = "Original tool definition:"

EXTRA_CHAT_PASSTHROUGH_FIELDS = (
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "metadata",
    "n",
    "parallel_tool_calls",
    "presence_penalty",
    "response_format",
    "seed",
    "service_tier",
    "stop",
    "stream_options",
    "top_logprobs",
    "user",
)

THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"


# ============================================================
# canonical JSON helpers (mirrors json_canonical.rs)
# ============================================================

def canonical_json_string(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ",".join(canonical_json_string(v) for v in value) + "]"
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: kv[0])
        parts = [f"{json.dumps(k, ensure_ascii=False)}:{canonical_json_string(v)}" for k, v in items]
        return "{" + ",".join(parts) + "}"
    return json.dumps(value, ensure_ascii=False)


def canonicalize_json_string_if_parseable(value: str) -> str:
    trimmed = value.strip()
    if not trimmed:
        return value
    try:
        parsed = json.loads(trimmed)
    except (json.JSONDecodeError, ValueError):
        return value
    return canonical_json_string(parsed)


def canonicalize_tool_arguments_str(value: str) -> str:
    if not value.strip():
        return "{}"
    return canonicalize_json_string_if_parseable(value)


def canonicalize_tool_arguments(value: Any) -> str:
    if value is None:
        return "{}"
    if isinstance(value, str):
        return canonicalize_tool_arguments_str(value)
    return canonical_json_string(value)


def short_sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ============================================================
# reasoning extraction (mirrors codex_chat_common.rs)
# ============================================================

def extract_reasoning_field_text(value: dict) -> str | None:
    """Exhaustively pull reasoning text from upstream message/delta."""
    for key in ("reasoning_content", "reasoning"):
        v = value.get(key)
        if isinstance(v, str) and v:
            return v

    reasoning = value.get("reasoning")
    if isinstance(reasoning, dict):
        for key in ("content", "text", "summary"):
            v = reasoning.get(key)
            if isinstance(v, str) and v:
                return v

    details = value.get("reasoning_details")
    if details is not None:
        text = _extract_reasoning_details_text(details)
        if text:
            return text

    return None


def _extract_reasoning_details_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value or None
    if isinstance(value, list):
        parts = [_extract_reasoning_detail_part_text(p) for p in value]
        text = "\n\n".join(p for p in parts if p)
        return text or None
    if isinstance(value, dict):
        return _extract_reasoning_detail_part_text(value)
    return None


def _extract_reasoning_detail_part_text(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("text", "content", "summary"):
            v = value.get(key)
            if isinstance(v, str) and v:
                return v
        parts = value.get("parts")
        if isinstance(parts, list):
            text = "\n\n".join(p for p in (_extract_reasoning_detail_part_text(x) for x in parts) if p)
            return text or None
    if isinstance(value, str) and value:
        return value
    return None


def extract_reasoning_summary_text(value: dict) -> str | None:
    for key in ("reasoning_content", "content", "text"):
        v = value.get(key)
        if isinstance(v, str) and v:
            return v
    summary = value.get("summary")
    if isinstance(summary, str) and summary:
        return summary
    if isinstance(summary, list):
        parts = []
        for part in summary:
            if isinstance(part, dict):
                for key in ("text", "content"):
                    v = part.get(key)
                    if isinstance(v, str) and v:
                        parts.append(v)
                        break
            elif isinstance(part, str) and part:
                parts.append(part)
        text = "\n\n".join(parts)
        return text or None
    return None


def split_leading_think_block(text: str) -> tuple[str, str] | None:
    """Split a leading <think>...</think> block into (reasoning, answer)."""
    stripped = text.lstrip()
    leading_ws_len = len(text) - len(stripped)
    if not stripped.startswith(THINK_OPEN_TAG):
        return None
    body_start = leading_ws_len + len(THINK_OPEN_TAG)
    close_rel = text.find(THINK_CLOSE_TAG, body_start)
    if close_rel == -1:
        return None
    close_start = close_rel
    answer_start = close_start + len(THINK_CLOSE_TAG)
    reasoning = text[body_start:close_start].strip()
    answer = text[answer_start:].lstrip("\r\n\t ")
    return (reasoning, answer)


def strip_leading_think_open_tag(text: str) -> str | None:
    stripped = text.lstrip()
    leading_ws_len = len(text) - len(stripped)
    after_ws = text[leading_ws_len:]
    if after_ws.startswith(THINK_OPEN_TAG):
        return after_ws[len(THINK_OPEN_TAG):].strip()
    return None


def append_reasoning_content(message: dict, reasoning: str) -> bool:
    reasoning = reasoning.strip()
    if not reasoning:
        return False
    existing = message.get("reasoning_content")
    if isinstance(existing, str) and existing:
        message["reasoning_content"] = existing + "\n\n" + reasoning
    else:
        message["reasoning_content"] = reasoning
    return True


def attach_reasoning_content_field(item: dict, reasoning: str) -> bool:
    reasoning = reasoning.strip()
    if not reasoning:
        return False
    item["reasoning_content"] = reasoning
    return True


def attach_optional_reasoning_content_field(item: dict, reasoning: str | None) -> bool:
    if reasoning is None:
        return False
    return attach_reasoning_content_field(item, reasoning)


# ============================================================
# response-side item builders (mirrors codex_chat_common.rs)
# ============================================================

def response_function_call_item(
    item_id: str, status: str, call_id: str, name: str, arguments: str, reasoning: str | None
) -> dict:
    item: dict = {
        "id": item_id,
        "type": "function_call",
        "status": status,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }
    attach_optional_reasoning_content_field(item, reasoning)
    return item


def response_function_call_item_with_namespace(
    item_id: str,
    status: str,
    call_id: str,
    name: str,
    namespace: str | None,
    arguments: str,
    reasoning: str | None,
) -> dict:
    item = response_function_call_item(item_id, status, call_id, name, arguments, reasoning)
    if namespace:
        item["namespace"] = namespace
    return item


# ============================================================
# tool context (mirrors CodexToolContext)
# ============================================================

class CodexToolKind:
    FUNCTION = "function"
    NAMESPACE = "namespace"
    CUSTOM = "custom"
    TOOL_SEARCH = "tool_search"


class CodexToolSpec:
    def __init__(self, kind: str, name: str, namespace: str | None = None):
        self.kind = kind
        self.name = name
        self.namespace = namespace


class CodexToolContext:
    def __init__(self):
        self.chat_tools: list[dict] = []
        self.seen_chat_names: set[str] = set()
        self.chat_name_to_spec: dict[str, CodexToolSpec] = {}
        self.namespace_name_to_chat_name: dict[tuple[str, str], str] = {}

    def chat_tools_list(self) -> list[dict]:
        return self.chat_tools

    def lookup_chat_name(self, chat_name: str) -> CodexToolSpec | None:
        return self.chat_name_to_spec.get(chat_name)

    def is_custom_tool_chat_name(self, chat_name: str) -> bool:
        spec = self.lookup_chat_name(chat_name)
        return spec is not None and spec.kind == CodexToolKind.CUSTOM

    def chat_name_for_response_function(self, name: str, namespace: str | None) -> str:
        if namespace:
            existing = self.namespace_name_to_chat_name.get((namespace, name))
            if existing:
                return existing
            return flatten_namespace_tool_name(namespace, name)
        return name

    def _add_chat_tool(self, chat_name: str, spec: CodexToolSpec, chat_tool: dict) -> None:
        if not chat_name.strip() or chat_name in self.seen_chat_names:
            return
        self.seen_chat_names.add(chat_name)
        if spec.namespace:
            self.namespace_name_to_chat_name[(spec.namespace, spec.name)] = chat_name
        self.chat_name_to_spec[chat_name] = spec
        self.chat_tools.append(chat_tool)

    def add_function_tool(self, tool: dict, namespace: str | None) -> None:
        original_name = responses_tool_name(tool)
        if original_name is None:
            return
        if namespace:
            chat_name = flatten_namespace_tool_name(namespace, original_name)
        else:
            chat_name = original_name
        chat_tool = responses_function_tool_to_chat_tool(tool, chat_name)
        if chat_tool is None:
            return
        kind = CodexToolKind.NAMESPACE if namespace else CodexToolKind.FUNCTION
        spec = CodexToolSpec(kind, original_name, namespace)
        self._add_chat_tool(chat_name, spec, chat_tool)

    def add_custom_tool(self, tool: dict) -> None:
        name = responses_tool_name(tool)
        if name is None:
            return
        description = json.dumps(responses_custom_tool_description(tool), ensure_ascii=False)
        chat_tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        CUSTOM_TOOL_INPUT_FIELD: {
                            "type": "string",
                            "description": CUSTOM_TOOL_INPUT_DESCRIPTION,
                        }
                    },
                    "required": [CUSTOM_TOOL_INPUT_FIELD],
                },
            },
        }
        spec = CodexToolSpec(CodexToolKind.CUSTOM, name)
        self._add_chat_tool(name, spec, chat_tool)

    def add_tool_search_tool(self) -> None:
        chat_tool = {
            "type": "function",
            "function": {
                "name": TOOL_SEARCH_PROXY_NAME,
                "description": "Search and load Codex tools, plugins, connectors, and MCP namespaces for the current task.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query for tools or connectors to load."},
                        "limit": {"type": "integer", "description": "Maximum number of tool groups to return."},
                    },
                    "required": ["query"],
                },
            },
        }
        spec = CodexToolSpec(CodexToolKind.TOOL_SEARCH, TOOL_SEARCH_PROXY_NAME)
        self._add_chat_tool(TOOL_SEARCH_PROXY_NAME, spec, chat_tool)

    def add_namespace_tool(self, namespace_tool: dict) -> None:
        namespace = namespace_tool.get("name")
        if not isinstance(namespace, str):
            return
        children = namespace_tool.get("tools") or namespace_tool.get("children")
        if not isinstance(children, list):
            return
        for child in children:
            if isinstance(child, dict) and child.get("type") == "function":
                self.add_function_tool(child, namespace)

    def add_response_tool(self, tool: Any) -> None:
        if isinstance(tool, str):
            self.add_custom_tool({"type": "custom", "name": tool})
            return
        if not isinstance(tool, dict):
            return
        ttype = tool.get("type")
        if ttype == "function":
            self.add_function_tool(tool, None)
        elif ttype == "custom":
            self.add_custom_tool(tool)
        elif ttype == "tool_search":
            self.add_tool_search_tool()
        elif ttype == "namespace":
            self.add_namespace_tool(tool)


def build_codex_tool_context_from_request(body: dict) -> CodexToolContext:
    ctx = CodexToolContext()
    tools = body.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            ctx.add_response_tool(tool)
    collect_tool_search_output_tools(body.get("input"), ctx)
    return ctx


def collect_tool_search_output_tools(value: Any, ctx: CodexToolContext) -> None:
    if isinstance(value, list):
        for item in value:
            collect_tool_search_output_tools(item, ctx)
    elif isinstance(value, dict):
        if value.get("type") == "tool_search_output":
            tools = value.get("tools")
            if isinstance(tools, list):
                for tool in tools:
                    ctx.add_response_tool(tool)
        for v in value.values():
            collect_tool_search_output_tools(v, ctx)


def flatten_namespace_tool_name(namespace: str, name: str) -> str:
    full = f"{namespace}__{name}"
    if len(full) <= CHAT_TOOL_NAME_MAX_LEN:
        return full
    suffix = "__" + short_sha256_hex(full.encode())
    prefix_len = CHAT_TOOL_NAME_MAX_LEN - len(suffix)
    prefix = full[:prefix_len]
    return prefix + suffix


def responses_tool_name(tool: dict) -> str | None:
    fn = tool.get("function")
    if isinstance(fn, dict):
        name = fn.get("name")
    else:
        name = tool.get("name")
    if isinstance(name, str):
        name = name.strip()
        if name:
            return name
    return None


def responses_custom_tool_description(tool: dict) -> str:
    return (
        CUSTOM_TOOL_PRESERVED_METADATA_HEADING
        + "\n```json\n"
        + canonical_json_string(tool)
        + "\n```"
    )


def responses_function_tool_to_chat_tool(tool: dict, chat_name: str) -> dict | None:
    if tool.get("type") != "function":
        return None
    fn = tool.get("function")
    if isinstance(fn, dict):
        chat_tool = {"type": "function", "function": dict(fn)}
        chat_tool["function"]["name"] = chat_name
        if "strict" in tool and "strict" not in chat_tool["function"]:
            chat_tool["function"]["strict"] = tool["strict"]
        return chat_tool
    function: dict = {
        "name": chat_name,
        "description": tool.get("description"),
        "parameters": tool.get("parameters", {}),
    }
    if "strict" in tool:
        function["strict"] = tool["strict"]
    return {"type": "function", "function": function}


# ============================================================
# Responses -> Chat request (mirrors responses_to_chat_completions)
# ============================================================

def instruction_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                t = item.get("text")
                if isinstance(t, str) and t.strip():
                    parts.append(t)
            elif isinstance(item, str) and item.strip():
                parts.append(item)
        return "\n\n".join(parts)
    return ""


def responses_role_to_chat_role(role: str) -> str:
    return {"developer": "system", "system": "system"}.get(role, role)


def responses_content_to_chat_content(role: str, content: Any) -> Any:
    if content is None or isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content

    chat_parts: list[dict] = []
    has_non_text = False

    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type", "")
        if ptype in ("input_text", "output_text", "text"):
            text = part.get("text")
            if isinstance(text, str) and text:
                chat_parts.append({"type": "text", "text": text})
        elif ptype == "refusal":
            text = part.get("refusal")
            if isinstance(text, str) and text:
                chat_parts.append({"type": "text", "text": text})
        elif ptype == "input_image":
            image_url = part.get("image_url")
            if image_url is not None:
                if isinstance(image_url, dict):
                    iu = image_url
                else:
                    iu = {"url": str(image_url)}
                chat_parts.append({"type": "image_url", "image_url": iu})
                has_non_text = True
        elif ptype == "input_file":
            file = responses_input_file_to_chat_file(part)
            if file is not None:
                chat_parts.append({"type": "file", "file": file})
                has_non_text = True
        elif ptype == "input_audio":
            input_audio = part.get("input_audio")
            if input_audio is not None:
                chat_parts.append({"type": "input_audio", "input_audio": input_audio})
                has_non_text = True

    if not has_non_text:
        return "\n".join(p["text"] for p in chat_parts if p.get("type") == "text")
    return chat_parts


def responses_input_file_to_chat_file(part: dict) -> dict | None:
    if "file_id" not in part and "file_data" not in part:
        return None
    file: dict = {}
    for key in ("file_id", "file_data", "filename"):
        if key in part:
            file[key] = part[key]
    return file


def responses_function_call_to_chat_tool_call(item: dict, ctx: CodexToolContext) -> dict:
    call_id = item.get("call_id") or item.get("id") or ""
    if not isinstance(call_id, str):
        call_id = ""
    name = item.get("name") or ""
    if not isinstance(name, str):
        name = ""
    namespace = item.get("namespace")
    if not isinstance(namespace, str):
        namespace = None
    chat_name = ctx.chat_name_for_response_function(name, namespace)
    arguments = canonicalize_tool_arguments(item.get("arguments"))
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": chat_name, "arguments": arguments},
    }


def responses_custom_tool_call_to_chat_tool_call(item: dict) -> dict:
    call_id = item.get("call_id") or item.get("id") or ""
    if not isinstance(call_id, str):
        call_id = ""
    name = item.get("name") or ""
    if not isinstance(name, str):
        name = ""
    input_val = item.get("input", "")
    arguments = canonical_json_string({CUSTOM_TOOL_INPUT_FIELD: input_val})
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def responses_tool_search_call_to_chat_tool_call(item: dict) -> dict:
    call_id = item.get("call_id") or item.get("id") or ""
    if not isinstance(call_id, str):
        call_id = ""
    arguments = item.get("arguments")
    if arguments is None:
        arguments = "{}"
    else:
        arguments = canonical_json_string(arguments)
    return {"id": call_id, "type": "function", "function": {"name": TOOL_SEARCH_PROXY_NAME, "arguments": arguments}}


def responses_tool_choice_to_chat(tool_choice: Any, ctx: CodexToolContext) -> Any:
    if isinstance(tool_choice, dict):
        ttype = tool_choice.get("type")
        if ttype == "function":
            name = tool_choice.get("name") or ""
            namespace = tool_choice.get("namespace")
            if not isinstance(namespace, str):
                namespace = None
            chat_name = ctx.chat_name_for_response_function(name, namespace)
            return {"type": "function", "function": {"name": chat_name}}
        if ttype == "tool_search":
            return {"type": "function", "function": {"name": TOOL_SEARCH_PROXY_NAME}}
        if ttype == "custom":
            name = tool_choice.get("name") or ""
            return {"type": "function", "function": {"name": name}}
    return tool_choice


def responses_item_reasoning_text(item: dict) -> str | None:
    # reasoning items carry their text in summary[].text or encrypted_content
    return extract_reasoning_summary_text(item)


def responses_message_reasoning_text(item: dict) -> str | None:
    return extract_reasoning_summary_text(item)


def append_pending_reasoning(pending: list[str], reasoning: str | None) -> None:
    if reasoning and reasoning.strip():
        if not pending or pending[-1] != reasoning:
            pending.append(reasoning)


def attach_pending_reasoning_to_assistant(message: dict, pending: list[str]) -> None:
    if not pending:
        return
    reasoning = "\n\n".join(pending)
    pending.clear()
    append_reasoning_content(message, reasoning)


def attach_reasoning_to_last_assistant(messages: list[dict], last_assistant_index: int | None, reasoning: str) -> bool:
    if last_assistant_index is None:
        return False
    if last_assistant_index < 0 or last_assistant_index >= len(messages):
        return False
    msg = messages[last_assistant_index]
    if msg.get("role") != "assistant":
        return False
    append_reasoning_content(msg, reasoning)
    return True


def ensure_tool_call_reasoning_content(message: dict) -> None:
    if message.get("role") != "assistant":
        return
    if not message.get("tool_calls"):
        return
    rc = message.get("reasoning_content")
    if not isinstance(rc, str) or not rc.strip():
        message["reasoning_content"] = " "


def backfill_tool_call_reasoning_placeholders(messages: list[dict]) -> None:
    for msg in messages:
        ensure_tool_call_reasoning_content(msg)


def update_last_assistant_index(messages: list[dict], message: dict, last_idx: list[int | None]) -> None:
    if message.get("role") == "assistant":
        last_idx[0] = len(messages)


def flush_pending_tool_calls(
    messages: list[dict],
    pending_tool_calls: list[dict],
    pending_reasoning: list[str],
    last_assistant_index: list[int | None],
) -> None:
    if not pending_tool_calls:
        return
    message: dict = {
        "role": "assistant",
        "content": None,
        "tool_calls": pending_tool_calls[:],
    }
    attach_pending_reasoning_to_assistant(message, pending_reasoning)
    pending_tool_calls.clear()
    last_assistant_index[0] = len(messages)
    messages.append(message)


def responses_message_item_to_chat_message(item: dict, pending_reasoning: list[str]) -> dict:
    role = item.get("role") or "user"
    if not isinstance(role, str):
        role = "user"
    chat_role = responses_role_to_chat_role(role)
    content = item.get("content")
    chat_content = responses_content_to_chat_content(chat_role, content) if content is not None else None
    message: dict = {"role": chat_role, "content": chat_content}
    if chat_role == "assistant":
        rtext = responses_message_reasoning_text(item)
        append_pending_reasoning(pending_reasoning, rtext)
        attach_pending_reasoning_to_assistant(message, pending_reasoning)
    elif pending_reasoning:
        pending_reasoning.clear()
    return message


def append_responses_item_as_chat_message(
    item: Any,
    messages: list[dict],
    pending_tool_calls: list[dict],
    pending_reasoning: list[str],
    last_assistant_index: list[int | None],
    ctx: CodexToolContext,
) -> None:
    if not isinstance(item, dict):
        return
    item_type = item.get("type")

    if item_type == "function_call":
        rtext = responses_item_reasoning_text(item)
        append_pending_reasoning(pending_reasoning, rtext)
        pending_tool_calls.append(responses_function_call_to_chat_tool_call(item, ctx))
        return
    if item_type == "custom_tool_call":
        rtext = responses_item_reasoning_text(item)
        append_pending_reasoning(pending_reasoning, rtext)
        pending_tool_calls.append(responses_custom_tool_call_to_chat_tool_call(item))
        return
    if item_type == "tool_search_call":
        rtext = responses_item_reasoning_text(item)
        append_pending_reasoning(pending_reasoning, rtext)
        pending_tool_calls.append(responses_tool_search_call_to_chat_tool_call(item))
        return
    if item_type in ("function_call_output", "custom_tool_call_output", "tool_search_output"):
        flush_pending_tool_calls(messages, pending_tool_calls, pending_reasoning, last_assistant_index)
        call_id = item.get("call_id") or ""
        if not isinstance(call_id, str):
            call_id = ""
        if item_type == "function_call_output":
            output_val = item.get("output")
            if isinstance(output_val, str):
                output = canonicalize_json_string_if_parseable(output_val)
            elif output_val is None:
                output = ""
            else:
                output = canonical_json_string(output_val)
        else:
            output = canonical_json_string(item)
        messages.append({"role": "tool", "tool_call_id": call_id, "content": output})
        return
    if item_type == "reasoning":
        reasoning = responses_item_reasoning_text(item) or ""
        attached = False
        if not pending_tool_calls:
            attached = attach_reasoning_to_last_assistant(messages, last_assistant_index[0], reasoning)
        if not attached:
            append_pending_reasoning(pending_reasoning, reasoning)
        return
    if item_type in ("input_text", "input_image", "input_file", "input_audio"):
        flush_pending_tool_calls(messages, pending_tool_calls, pending_reasoning, last_assistant_index)
        role = item.get("role") or "user"
        if not isinstance(role, str):
            role = "user"
        role = responses_role_to_chat_role(role)
        message: dict = {
            "role": role,
            "content": responses_content_to_chat_content(role, [item]),
        }
        if role == "assistant":
            attach_pending_reasoning_to_assistant(message, pending_reasoning)
            update_last_assistant_index(messages, message, last_assistant_index)
            messages.append(message)
            return
        elif pending_reasoning:
            pending_reasoning.clear()
        update_last_assistant_index(messages, message, last_assistant_index)
        messages.append(message)
        return

    # "message" or None or unknown
    flush_pending_tool_calls(messages, pending_tool_calls, pending_reasoning, last_assistant_index)
    if item.get("role") is not None or item.get("content") is not None:
        message = responses_message_item_to_chat_message(item, pending_reasoning)
        update_last_assistant_index(messages, message, last_assistant_index)
        messages.append(message)


def append_responses_input_as_chat_messages(input_data: Any, messages: list[dict], ctx: CodexToolContext) -> None:
    pending_tool_calls: list[dict] = []
    pending_reasoning: list[str] = []
    last_assistant_index: list[int | None] = [None]

    if isinstance(input_data, str):
        messages.append({"role": "user", "content": input_data})
    elif isinstance(input_data, list):
        for item in input_data:
            append_responses_item_as_chat_message(
                item, messages, pending_tool_calls, pending_reasoning, last_assistant_index, ctx
            )
    elif isinstance(input_data, dict):
        append_responses_item_as_chat_message(
            input_data, messages, pending_tool_calls, pending_reasoning, last_assistant_index, ctx
        )

    flush_pending_tool_calls(messages, pending_tool_calls, pending_reasoning, last_assistant_index)
    backfill_tool_call_reasoning_placeholders(messages)


def collapse_system_messages_to_head(messages: list[dict]) -> list[dict]:
    """Collapse consecutive leading system messages into one."""
    result: list[dict] = []
    system_parts: list[str] = []
    seen_non_system = False
    for msg in messages:
        if msg.get("role") == "system" and not seen_non_system:
            content = msg.get("content")
            if isinstance(content, str):
                if content.strip():
                    system_parts.append(content)
            elif content is not None:
                result.append(msg)
                seen_non_system = True
                continue
        else:
            seen_non_system = True
            result.append(msg)
    if system_parts:
        result.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    return result


def is_openai_o_series(model: str) -> bool:
    return len(model) > 1 and model[0] == "o" and model[1:2].isdigit()


def responses_to_chat_completions(body: dict) -> dict:
    """Convert an OpenAI Responses request into an OpenAI Chat Completions request."""
    result: dict = {}
    ctx = build_codex_tool_context_from_request(body)

    model = body.get("model")
    if model is not None:
        result["model"] = model

    messages: list[dict] = []
    instructions = body.get("instructions")
    instr_text = instruction_text(instructions)
    if instr_text:
        messages.append({"role": "system", "content": instr_text})

    input_data = body.get("input")
    if input_data is not None:
        append_responses_input_as_chat_messages(input_data, messages, ctx)
    messages = collapse_system_messages_to_head(messages)
    result["messages"] = messages

    model_str = model if isinstance(model, str) else ""
    if "max_output_tokens" in body:
        if is_openai_o_series(model_str):
            result["max_completion_tokens"] = body["max_output_tokens"]
        else:
            result["max_tokens"] = body["max_output_tokens"]
    if "max_tokens" in body:
        result["max_tokens"] = body["max_tokens"]
    if "max_completion_tokens" in body:
        result["max_completion_tokens"] = body["max_completion_tokens"]

    for key in ("temperature", "top_p", "stream"):
        if key in body:
            result[key] = body[key]

    # reasoning.effort: SGLang/Chat-only upstreams don't consume it; drop it.
    # (cc-switch applies per-platform thinking params here, not needed for SGLang.)

    tools = ctx.chat_tools_list()
    if tools:
        result["tools"] = tools

    if "tool_choice" in body:
        result["tool_choice"] = responses_tool_choice_to_chat(body["tool_choice"], ctx)

    for key in EXTRA_CHAT_PASSTHROUGH_FIELDS:
        if key in body:
            result[key] = body[key]

    has_tools = bool(result.get("tools"))
    if not has_tools:
        result.pop("tool_choice", None)
        result.pop("parallel_tool_calls", None)

    # Streaming: ensure upstream returns usage in the final chunk.
    if result.get("stream"):
        so = result.get("stream_options")
        if isinstance(so, dict):
            so["include_usage"] = True
        else:
            result["stream_options"] = {"include_usage": True}

    return result


# ============================================================
# Chat -> Responses response (non-streaming)
# mirrors chat_completion_to_response_with_context
# ============================================================

def chat_reasoning_text(message: dict) -> str | None:
    reasoning = extract_reasoning_field_text(message)
    if reasoning:
        return reasoning
    content = message.get("content")
    if isinstance(content, str):
        split = split_leading_think_block(content)
        if split and split[0]:
            return split[0]
    return None


def chat_reasoning_to_response_output_item(reasoning: str | None, response_id: str) -> dict | None:
    if not reasoning:
        return None
    return {
        "id": f"rs_{response_id}",
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": reasoning}],
    }


def chat_message_to_response_output_item(message: dict, response_id: str) -> dict | None:
    content_parts: list[dict] = []
    content = message.get("content")
    if isinstance(content, str):
        split = split_leading_think_block(content)
        text = split[1] if split else content
        if text:
            content_parts.append({"type": "output_text", "text": text, "annotations": []})
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type", "")
            if ptype in ("text", "output_text"):
                text = part.get("text")
                if isinstance(text, str) and text:
                    content_parts.append({"type": "output_text", "text": text, "annotations": []})
            elif ptype == "refusal":
                text = part.get("refusal")
                if isinstance(text, str) and text:
                    content_parts.append({"type": "refusal", "refusal": text})

    refusal = message.get("refusal")
    if isinstance(refusal, str) and refusal:
        content_parts.append({"type": "refusal", "refusal": refusal})

    if not content_parts:
        return None
    return {
        "id": f"{response_id}_msg",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": content_parts,
    }


def chat_tool_calls_to_response_output_items(
    message: dict, reasoning: str | None, ctx: CodexToolContext
) -> list[dict]:
    output: list[dict] = []
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for index, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            name = fn.get("name") or ""
            if not isinstance(name, str) or not name:
                continue
            output.append(chat_tool_call_to_response_item(tc, index, reasoning, ctx))
    elif isinstance(message.get("function_call"), dict):
        item = chat_legacy_function_call_to_response_item(message["function_call"], reasoning, ctx)
        if item is not None:
            output.append(item)
    return output


def chat_tool_call_to_response_item(
    tool_call: dict, index: int, reasoning: str | None, ctx: CodexToolContext
) -> dict:
    call_id = tool_call.get("id")
    if not isinstance(call_id, str) or not call_id:
        call_id = f"call_{index}"
    fn = tool_call.get("function") or {}
    name = fn.get("name") or ""
    arguments = canonicalize_tool_arguments(fn.get("arguments"))
    item_id = response_tool_call_item_id_from_chat_name(call_id, name, ctx)
    return response_tool_call_item_from_chat_name(item_id, "completed", call_id, name, arguments, reasoning, ctx)


def chat_legacy_function_call_to_response_item(
    function_call: dict, reasoning: str | None, ctx: CodexToolContext
) -> dict | None:
    call_id = function_call.get("id")
    if not isinstance(call_id, str) or not call_id:
        call_id = "call_0"
    name = function_call.get("name") or ""
    if not isinstance(name, str) or not name:
        return None
    arguments = canonicalize_tool_arguments(function_call.get("arguments"))
    item_id = response_tool_call_item_id_from_chat_name(call_id, name, ctx)
    return response_tool_call_item_from_chat_name(item_id, "completed", call_id, name, arguments, reasoning, ctx)


def response_tool_call_item_id_from_chat_name(call_id: str, chat_name: str, ctx: CodexToolContext) -> str:
    if ctx.is_custom_tool_chat_name(chat_name):
        return f"ctc_{call_id}"
    return f"fc_{call_id}"


def response_tool_call_item_from_chat_name(
    item_id: str,
    status: str,
    call_id: str,
    chat_name: str,
    arguments: str,
    reasoning: str | None,
    ctx: CodexToolContext,
) -> dict:
    spec = ctx.lookup_chat_name(chat_name)
    if spec is not None and spec.kind == CodexToolKind.TOOL_SEARCH:
        return response_tool_search_call_item(call_id, status, arguments, reasoning)
    if spec is not None and spec.kind == CodexToolKind.CUSTOM:
        return response_custom_tool_call_item(item_id, status, call_id, spec.name, arguments, reasoning)
    if spec is not None:
        return response_function_call_item_with_namespace(
            item_id, status, call_id, spec.name, spec.namespace, arguments, reasoning
        )
    return response_function_call_item(item_id, status, call_id, chat_name, arguments, reasoning)


def parse_tool_arguments_object(arguments: str) -> dict:
    """Parse tool arguments into an object, mirroring cc-switch.

    Empty -> {}, valid JSON object -> that object, otherwise wrap the raw
    string under {"query": ...} so callers always get a dict.
    """
    if not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, ValueError):
        return {"query": arguments}
    if isinstance(parsed, dict):
        return parsed
    return {"query": arguments}


def response_tool_search_call_item(call_id: str, status: str, arguments: str, reasoning: str | None) -> dict:
    # `execution: "client"` is REQUIRED: without it Codex treats the
    # tool_search_call as server-side and never executes the search locally,
    # so the turn stalls silently (the "hang" on "remind me every 7 min").
    # `arguments` must be an object, not a JSON string, for the same reason.
    # Mirrors cc-switch transform_codex_chat.rs:1534.
    item: dict = {
        "id": f"tsc_{call_id}",
        "type": "tool_search_call",
        "status": status,
        "call_id": call_id,
        "execution": "client",
        "arguments": parse_tool_arguments_object(arguments),
    }
    attach_optional_reasoning_content_field(item, reasoning)
    return item


def response_custom_tool_call_item(
    item_id: str, status: str, call_id: str, name: str, arguments: str, reasoning: str | None
) -> dict:
    input_str = custom_tool_input_from_chat_arguments(arguments)
    item: dict = {
        "id": item_id,
        "type": "custom_tool_call",
        "status": status,
        "call_id": call_id,
        "name": name,
        "input": input_str,
    }
    attach_optional_reasoning_content_field(item, reasoning)
    return item


def custom_tool_input_from_chat_arguments(arguments: str) -> str:
    if not arguments.strip():
        return ""
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, ValueError):
        return arguments
    if isinstance(parsed, dict):
        v = parsed.get(CUSTOM_TOOL_INPUT_FIELD)
        if isinstance(v, str):
            return v
    return arguments


def response_id_from_chat_id(id_: str | None) -> str:
    id_ = id_ or "ccswitch"
    if id_.startswith("resp_"):
        return id_
    return f"resp_{id_}"


def response_status_from_finish_reason(finish_reason: str | None) -> str:
    if finish_reason == "length":
        return "incomplete"
    return "completed"


def chat_usage_to_responses_usage(usage: Any) -> dict:
    if not isinstance(usage, dict):
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
        }
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens

    result: dict = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }

    ptd = usage.get("prompt_tokens_details")
    if isinstance(ptd, dict):
        cached = ptd.get("cached_tokens")
    else:
        itd = usage.get("input_tokens_details")
        cached = itd.get("cached_tokens") if isinstance(itd, dict) else None
    if cached is not None:
        result["input_tokens_details"] = {"cached_tokens": cached}

    ctd = usage.get("completion_tokens_details")
    if isinstance(ctd, dict):
        details = dict(ctd)
        if "reasoning_tokens" not in details:
            details["reasoning_tokens"] = 0
        result["output_tokens_details"] = details
    else:
        result["output_tokens_details"] = {"reasoning_tokens": 0}

    if "cache_read_input_tokens" in usage:
        result["cache_read_input_tokens"] = usage["cache_read_input_tokens"]
    if "cache_creation_input_tokens" in usage:
        result["cache_creation_input_tokens"] = usage["cache_creation_input_tokens"]

    return result


def chat_completion_to_response(body: dict, ctx: CodexToolContext | None = None) -> dict:
    """Convert a non-streaming Chat Completions response into a Responses response."""
    if ctx is None:
        ctx = CodexToolContext()
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("No choices in chat response")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("No message in chat choice")

    response_id = response_id_from_chat_id(body.get("id") if isinstance(body.get("id"), str) else None)
    model = body.get("model") or ""
    created_at = body.get("created") or 0
    finish_reason = choice.get("finish_reason")

    reasoning = chat_reasoning_text(message)
    output: list[dict] = []
    reasoning_item = chat_reasoning_to_response_output_item(reasoning, response_id)
    if reasoning_item is not None:
        output.append(reasoning_item)
    message_item = chat_message_to_response_output_item(message, response_id)
    if message_item is not None:
        output.append(message_item)
    output.extend(chat_tool_calls_to_response_output_items(message, reasoning, ctx))

    response: dict = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": response_status_from_finish_reason(finish_reason),
        "model": model,
        "output": output,
        "usage": chat_usage_to_responses_usage(body.get("usage")),
    }
    if finish_reason == "length":
        response["incomplete_details"] = {"reason": "max_output_tokens"}
    return response


def chat_error_to_response_error(body: Any) -> dict:
    if body is None:
        return {
            "error": {
                "message": "Upstream returned an empty error response",
                "type": "upstream_error",
                "code": None,
                "param": None,
            }
        }
    if isinstance(body, str):
        return {"error": {"message": body, "type": "upstream_error", "code": None, "param": None}}
    source = body.get("error", body) if isinstance(body, dict) else body
    if not isinstance(source, dict):
        source = {}
    message = (
        source.get("message")
        or source.get("detail")
        or source.get("status_msg")
        or (source.get("base_resp", {}) or {}).get("status_msg")
    )
    if message is None:
        message = json.dumps(source, ensure_ascii=False)
    error_type = source.get("type") or source.get("code")
    return {
        "error": {
            "message": message,
            "type": error_type if isinstance(error_type, str) else "upstream_error",
            "code": source.get("code"),
            "param": source.get("param"),
        }
    }
