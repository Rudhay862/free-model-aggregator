#!/usr/bin/env python3
"""free-model-aggregator-01: a quality-ordered free-model aggregator.

One long-running server. It reads the trial brief at $MAKEFUN_PROMPT_FILE
(providers directory, quality table, workload manifest), registers its
catalog with the env at $MAKEFUN_ENV_URL, and serves the OpenAI Chat
Completions API and the Anthropic Messages API on /v1/chat/completions and
/v1/messages, plus GET /v1/models.

Routing rule (deterministic, no clock, no randomness):
  * for each request, take the manifest's requirements for that request
    (requests arrive one at a time, in manifest order);
  * candidates are the free models that meet every requirement, ordered by
    rating descending, then provider id, then model id;
  * one candidate per provider (its best eligible model);
  * a JSON-mode request (OpenAI response_format) never goes to an
    anthropic-wire provider, whose wire cannot carry JSON mode;
  * try candidates in order; a refusal (402/429/5xx, or a transport error)
    moves to the next candidate; the first success answers the request.
    A provider is tried at most once per request.
  * a provider that refuses three times in a row (no success in between)
    is treated as spent for the rest of the trial and is not tried again:
    its budget is gone or it is in outage, and retrying it on every
    later request only wastes provider calls (the tie-break). Three, not
    two: a scheduled failure pair (two consecutive refusals with budget
    left) must not retire a provider — the practice trials showed a
    two-strike rule wrongly retiring cloudflare after its scheduled
    failures and dropping later requests to worse models.
  * the provider's answer text is passed through whole, and its usage
    counts are reported exactly as the provider reported them; the stop
    reason is mapped to the client's wire (stop / end_turn, family 0).

`python3 main.py --list` prints the catalog as one JSON document and exits
0 without starting the server. With no brief it prints {"providers":[]}.

Standard library only. Python 3.13.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CATALOG_SCHEMA = "free-model-aggregator-01/catalog/v1"
PROTOCOL_VERSION = 1
FIXED_CREATED = 1767225600
CREATED_AT = "2026-01-01T00:00:00Z"
ANSWER_PREFIX = "ANSWER ECHO"
CHUNK_CHARS = 64
PROVIDER_TIMEOUT = 5.0
CONTROL_TIMEOUT = 5.0
POLL_INTERVAL = 0.05

# Module state, filled by main() before the server starts.
G_PROVIDERS = {}   # provider id -> {wire, auth, key, key_env, path, base_url}
G_MODELS = {}      # (provider id, model id) -> row dict
G_MANIFEST = []    # [{id, api, stream, weight, requires}]
G_CATALOG = {"providers": []}
G_COUNTER = 0
G_LOCK = threading.Lock()
G_FAIL_STREAK = {}  # provider id -> consecutive refusals since its last success
G_DEAD = set()      # provider ids treated as spent for the rest of the trial


def log(line):
    sys.stderr.write("aggregator: %s\n" % line)
    sys.stderr.flush()


# ---------------------------------------------------------------- brief

def find_brief_docs(text):
    """Return the free-model-aggregator-01 JSON documents embedded in text."""
    docs = {}
    dec = json.JSONDecoder()
    start = 0
    while True:
        i = text.find('{"schema"', start)
        if i < 0:
            break
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            start = i + 1
            continue
        if isinstance(obj, dict) and isinstance(obj.get("schema"), str):
            docs[obj["schema"]] = obj
        start = end
    return docs


def load_brief(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return None
    docs = find_brief_docs(text)
    providers = docs.get("free-model-aggregator-01/providers/v1")
    ratings = docs.get("free-model-aggregator-01/ratings/v1")
    workload = docs.get("free-model-aggregator-01/workload/v1")
    if not providers or not ratings or not workload:
        return None
    return providers, ratings, workload


def build_catalog(providers_doc, ratings_doc):
    by_provider = {}
    for r in ratings_doc.get("ratings", []):
        by_provider.setdefault(r["provider"], []).append({
            "id": r["model"],
            "free": bool(r["free"]),
            "context_length": r["context_length"],
            "tools": bool(r["tools"]),
            "vision": bool(r["vision"]),
            "json": bool(r["json"]),
            "rating_milli": r["rating_milli"],
        })
    catalog_providers = []
    for p in providers_doc.get("providers", []):
        catalog_providers.append({
            "id": p["id"],
            "base_url": p["path"],
            "key_env": p.get("key_env", ""),
            "wire": p["wire"],
            "models": by_provider.get(p["id"], []),
        })
    return {"schema": CATALOG_SCHEMA, "providers": catalog_providers}


def free_model_list():
    """Free models, deduplicated by id, in catalog order (for GET /v1/models)."""
    seen = set()
    out = []
    for provider in G_CATALOG["providers"]:
        for m in provider["models"]:
            if not m["free"] or m["id"] in seen:
                continue
            seen.add(m["id"])
            out.append(m)
    return out


# ---------------------------------------------------------------- http bits

def http_call(method, url, body_obj=None, headers=None, timeout=CONTROL_TIMEOUT):
    """One HTTP call. Returns (status, raw_bytes); status 0 on transport error."""
    data = None
    hdrs = dict(headers or {})
    if body_obj is not None:
        data = json.dumps(body_obj, separators=(",", ":")).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        try:
            return err.code, err.read()
        except Exception:
            return err.code, b""
    except Exception:
        return 0, b""


def parse_sse_frames(text):
    frames = []
    for chunk in text.replace("\r\n", "\n").split("\n\n"):
        if not chunk.strip():
            continue
        event = None
        data_lines = []
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].removeprefix(" "))
        if data_lines:
            frames.append({"event": event, "data": "\n".join(data_lines)})
    return frames


# ---------------------------------------------------------------- request model

def blocks_text(content):
    """The text a wire's parser sees in one message content (string or parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
        return "\n".join(parts)
    return ""


def extract_openai(body):
    norm = {
        "system": "", "messages": [], "tools": [], "tool_choice": None,
        "max_tokens": None, "temperature": None, "top_p": None,
        "stops": [], "json_mode": False, "json_type": None,
        "stream": body.get("stream") is True,
    }
    system_parts = []
    for msg in body.get("messages", []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            text = blocks_text(msg.get("content"))
            if text != "":
                system_parts.append(text)
            continue
        if role not in ("user", "assistant"):
            continue
        norm["messages"].append({"role": role, "parts": openai_parts(msg.get("content"))})
    norm["system"] = "\n".join(system_parts)
    for tool in body.get("tools", []) or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            norm["tools"].append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            })
    choice = body.get("tool_choice")
    if choice == "auto" or choice == "none" or choice == "required":
        norm["tool_choice"] = choice
    elif isinstance(choice, dict) and choice.get("type") == "function":
        fn = choice.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            norm["tool_choice"] = ("tool", fn["name"])
    stop = body.get("stop")
    if isinstance(stop, str):
        norm["stops"] = [stop]
    elif isinstance(stop, list):
        norm["stops"] = [s for s in stop if isinstance(s, str)]
    fmt = body.get("response_format")
    if isinstance(fmt, dict) and fmt.get("type") in ("json_object", "json_schema"):
        norm["json_mode"] = True
        norm["json_type"] = fmt["type"]
    for key in ("max_tokens", "max_completion_tokens"):
        if isinstance(body.get(key), int):
            norm["max_tokens"] = body[key]
            break
    if isinstance(body.get("temperature"), (int, float)):
        norm["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        norm["top_p"] = body["top_p"]
    return norm


def openai_parts(content):
    if isinstance(content, str):
        return [{"text": content}]
    parts = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"text": block["text"]})
            elif block.get("type") == "image_url":
                url = (block.get("image_url") or {}).get("url")
                if isinstance(url, str):
                    parts.append(image_part(url))
    return parts


def extract_anthropic(body):
    norm = {
        "system": "", "messages": [], "tools": [], "tool_choice": None,
        "max_tokens": None, "temperature": None, "top_p": None,
        "stops": [], "json_mode": False, "json_type": None,
        "stream": body.get("stream") is True,
    }
    norm["system"] = blocks_text(body.get("system"))
    for msg in body.get("messages", []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        norm["messages"].append({"role": role, "parts": anthropic_parts(msg.get("content"))})
    for tool in body.get("tools", []) or []:
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            continue
        norm["tools"].append({
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {}),
        })
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        ctype = choice.get("type")
        if ctype in ("auto", "any", "none"):
            norm["tool_choice"] = ctype
        elif ctype == "tool" and isinstance(choice.get("name"), str):
            norm["tool_choice"] = ("tool", choice["name"])
    stops = body.get("stop_sequences")
    if isinstance(stops, list):
        norm["stops"] = [s for s in stops if isinstance(s, str)]
    if isinstance(body.get("max_tokens"), int):
        norm["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        norm["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        norm["top_p"] = body["top_p"]
    return norm


def anthropic_parts(content):
    if isinstance(content, str):
        return [{"text": content}]
    parts = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"text": block["text"]})
            elif block.get("type") == "image":
                source = block.get("source") or {}
                if source.get("type") == "base64" and isinstance(source.get("data"), str):
                    parts.append({
                        "image_mime": source.get("media_type", "image/png"),
                        "image_b64": source["data"],
                    })
                elif source.get("type") == "url" and isinstance(source.get("url"), str):
                    parts.append({"image_url": source["url"]})
    return parts


def image_part(url):
    """An image part from an OpenAI image_url value (data URL or plain URL)."""
    if isinstance(url, str) and url.startswith("data:"):
        head, _, data = url[5:].partition(",")
        mime = head.split(";")[0] or "image/png"
        return {"image_mime": mime, "image_b64": data}
    return {"image_url": url}


# ---------------------------------------------------------------- provider requests

def openai_content(parts):
    if len(parts) == 1 and "text" in parts[0]:
        return parts[0]["text"]
    out = []
    for p in parts:
        if "text" in p:
            out.append({"type": "text", "text": p["text"]})
        elif "image_mime" in p:
            out.append({"type": "image_url", "image_url": {
                "url": "data:%s;base64,%s" % (p["image_mime"], p["image_b64"]),
            }})
        elif "image_url" in p:
            out.append({"type": "image_url", "image_url": {"url": p["image_url"]}})
    return out


def anthropic_content(parts):
    if len(parts) == 1 and "text" in parts[0]:
        return parts[0]["text"]
    out = []
    for p in parts:
        if "text" in p:
            out.append({"type": "text", "text": p["text"]})
        elif "image_mime" in p:
            out.append({"type": "image", "source": {
                "type": "base64", "media_type": p["image_mime"], "data": p["image_b64"],
            }})
        elif "image_url" in p:
            out.append({"type": "image", "source": {"type": "url", "url": p["image_url"]}})
    return out


def gemini_parts(parts):
    out = []
    for p in parts:
        if "text" in p:
            out.append({"text": p["text"]})
        elif "image_mime" in p:
            out.append({"inlineData": {"mimeType": p["image_mime"], "data": p["image_b64"]}})
        elif "image_url" in p:
            out.append({"fileData": {"fileUri": p["image_url"]}})
    return out


def tool_choice_openai(choice):
    if choice == "auto":
        return "auto"
    if choice == "any":
        return "required"
    if choice == "none":
        return "none"
    if isinstance(choice, tuple) and choice[0] == "tool":
        return {"type": "function", "function": {"name": choice[1]}}
    return None


def tool_choice_anthropic(choice):
    if choice in ("auto", "any", "none"):
        return {"type": choice}
    if isinstance(choice, tuple) and choice[0] == "tool":
        return {"type": "tool", "name": choice[1]}
    return None


def build_openai_request(norm, model, stream):
    body = {"model": model, "messages": []}
    if norm["system"] != "":
        body["messages"].append({"role": "system", "content": norm["system"]})
    for msg in norm["messages"]:
        body["messages"].append({"role": msg["role"], "content": openai_content(msg["parts"])})
    if norm["max_tokens"] is not None:
        body["max_tokens"] = norm["max_tokens"]
    if norm["temperature"] is not None:
        body["temperature"] = norm["temperature"]
    if norm["top_p"] is not None:
        body["top_p"] = norm["top_p"]
    if norm["tools"]:
        body["tools"] = [{
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        } for t in norm["tools"]]
    choice = tool_choice_openai(norm["tool_choice"])
    if choice is not None:
        body["tool_choice"] = choice
    if norm["stops"]:
        body["stop"] = list(norm["stops"])
    if norm["json_mode"]:
        body["response_format"] = {"type": norm["json_type"]}
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    return "/v1/chat/completions", body


def build_anthropic_request(norm, model, stream):
    body = {"model": model, "messages": []}
    if norm["system"] != "":
        body["system"] = norm["system"]
    for msg in norm["messages"]:
        body["messages"].append({"role": msg["role"], "content": anthropic_content(msg["parts"])})
    if norm["max_tokens"] is not None:
        body["max_tokens"] = norm["max_tokens"]
    if norm["temperature"] is not None:
        body["temperature"] = norm["temperature"]
    if norm["top_p"] is not None:
        body["top_p"] = norm["top_p"]
    if norm["tools"]:
        body["tools"] = [{
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        } for t in norm["tools"]]
    choice = tool_choice_anthropic(norm["tool_choice"])
    if choice is not None:
        body["tool_choice"] = choice
    if norm["stops"]:
        body["stop_sequences"] = list(norm["stops"])
    if stream:
        body["stream"] = True
    return "/v1/messages", body


def build_gemini_request(norm, model, stream):
    action = "streamGenerateContent" if stream else "generateContent"
    path = "/v1beta/models/%s:%s" % (model, action)
    body = {"contents": []}
    if norm["system"] != "":
        body["systemInstruction"] = {"parts": [{"text": norm["system"]}]}
    for msg in norm["messages"]:
        role = "user" if msg["role"] == "user" else "model"
        body["contents"].append({"role": role, "parts": gemini_parts(msg["parts"])})
    if norm["tools"]:
        body["tools"] = [{
            "functionDeclarations": [{
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            } for t in norm["tools"]],
        }]
    config = {}
    if norm["max_tokens"] is not None:
        config["maxOutputTokens"] = norm["max_tokens"]
    if norm["temperature"] is not None:
        config["temperature"] = norm["temperature"]
    if norm["top_p"] is not None:
        config["topP"] = norm["top_p"]
    if norm["stops"]:
        config["stopSequences"] = list(norm["stops"])
    if norm["json_mode"]:
        config["responseMimeType"] = "application/json"
    choice = norm["tool_choice"]
    if choice == "any" or (isinstance(choice, tuple) and choice[0] == "tool"):
        config["toolConfig"] = {"functionCallingConfig": {"mode": "ANY"}}
    elif choice == "none":
        config["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}
    elif choice == "auto":
        config["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
    if config:
        body["generationConfig"] = config
    return path, body


def provider_headers(wire, auth, key):
    headers = {}
    if auth == "bearer":
        headers["Authorization"] = "Bearer " + key
    elif auth == "x-api-key":
        headers["x-api-key"] = key
    elif auth == "x-goog-api-key":
        headers["x-goog-api-key"] = key
    if wire == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    return headers


# ---------------------------------------------------------------- provider answers

def text_blocks(blocks):
    return "".join(
        b.get("text", "")
        for b in blocks
        if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
    )


def parse_openai_answer(raw, stream):
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not stream:
        try:
            value = json.loads(text)
        except ValueError:
            return None
        if not isinstance(value, dict):
            return None
        choices = value.get("choices")
        if not isinstance(choices, list) or not choices:
            return None
        choice = choices[0]
        if not isinstance(choice, dict):
            return None
        message = choice.get("message")
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        if isinstance(content, str):
            answer = content
        elif isinstance(content, list):
            answer = text_blocks(content)
        else:
            return None
        usage = value.get("usage")
        if not isinstance(usage, dict):
            return None
        uin = usage.get("prompt_tokens")
        uout = usage.get("completion_tokens")
        if not isinstance(uin, int) or not isinstance(uout, int):
            return None
        if not answer.startswith(ANSWER_PREFIX):
            return None
        return answer, uin, uout
    answer = ""
    uin = None
    uout = None
    for frame in parse_sse_frames(text):
        if frame["data"].strip() == "[DONE]":
            continue
        try:
            chunk = json.loads(frame["data"])
        except ValueError:
            return None
        if not isinstance(chunk, dict):
            return None
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            if isinstance(usage.get("prompt_tokens"), int):
                uin = usage["prompt_tokens"]
            if isinstance(usage.get("completion_tokens"), int):
                uout = usage["completion_tokens"]
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            return None
        delta = choice.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            answer += delta["content"]
    if uin is None or uout is None:
        return None
    if not answer.startswith(ANSWER_PREFIX):
        return None
    return answer, uin, uout


def parse_anthropic_answer(raw, stream):
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not stream:
        try:
            value = json.loads(text)
        except ValueError:
            return None
        if not isinstance(value, dict) or value.get("type") != "message":
            return None
        content = value.get("content")
        if not isinstance(content, list):
            return None
        answer = text_blocks(content)
        usage = value.get("usage")
        if not isinstance(usage, dict):
            return None
        uin = usage.get("input_tokens")
        uout = usage.get("output_tokens")
        if not isinstance(uin, int) or not isinstance(uout, int):
            return None
        if not answer.startswith(ANSWER_PREFIX):
            return None
        return answer, uin, uout
    answer = ""
    uin = None
    uout = None
    for frame in parse_sse_frames(text):
        try:
            data = json.loads(frame["data"])
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        event = frame["event"]
        if event == "message_start":
            usage = (data.get("message") or {}).get("usage") or {}
            if isinstance(usage.get("input_tokens"), int):
                uin = usage["input_tokens"]
        elif event == "content_block_delta":
            delta = data.get("delta") or {}
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                answer += delta["text"]
        elif event == "message_delta":
            usage = data.get("usage") or {}
            if isinstance(usage.get("output_tokens"), int):
                uout = usage["output_tokens"]
    if uin is None or uout is None:
        return None
    if not answer.startswith(ANSWER_PREFIX):
        return None
    return answer, uin, uout


def gemini_answer_parts(value):
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    first = candidates[0]
    if not isinstance(first, dict):
        return None
    content = first.get("content") or {}
    parts = content.get("parts")
    if not isinstance(parts, list):
        return None
    return "".join(
        p.get("text", "")
        for p in parts
        if isinstance(p, dict) and isinstance(p.get("text"), str)
    )


def parse_gemini_answer(raw, stream):
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not stream:
        try:
            value = json.loads(text)
        except ValueError:
            return None
        if not isinstance(value, dict):
            return None
        answer = gemini_answer_parts(value)
        if answer is None:
            return None
        usage = value.get("usageMetadata") or {}
        uin = usage.get("promptTokenCount")
        uout = usage.get("candidatesTokenCount")
        if not isinstance(uin, int) or not isinstance(uout, int):
            return None
        if not answer.startswith(ANSWER_PREFIX):
            return None
        return answer, uin, uout
    answer = ""
    uin = None
    uout = None
    for frame in parse_sse_frames(text):
        try:
            value = json.loads(frame["data"])
        except ValueError:
            return None
        if not isinstance(value, dict):
            return None
        piece = gemini_answer_parts(value)
        if piece is None:
            return None
        answer += piece
        usage = value.get("usageMetadata") or {}
        if isinstance(usage.get("promptTokenCount"), int):
            uin = usage["promptTokenCount"]
        if isinstance(usage.get("candidatesTokenCount"), int):
            uout = usage["candidatesTokenCount"]
    if uin is None or uout is None:
        return None
    if not answer.startswith(ANSWER_PREFIX):
        return None
    return answer, uin, uout


def parse_provider_answer(wire, raw, stream):
    if wire == "openai":
        return parse_openai_answer(raw, stream)
    if wire == "anthropic":
        return parse_anthropic_answer(raw, stream)
    return parse_gemini_answer(raw, stream)


# ---------------------------------------------------------------- client rendering

def split_pieces(text):
    return [text[i:i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)] or [""]


def render_openai_json(answer, uin, uout):
    return json.dumps({
        "id": "chatcmpl-agg",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": "auto",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
            "logprobs": None,
        }],
        "usage": {
            "prompt_tokens": uin,
            "completion_tokens": uout,
            "total_tokens": uin + uout,
        },
    }, separators=(",", ":")).encode("utf-8")


def render_openai_stream(answer, uin, uout):
    frames = []

    def chunk(delta, finish):
        return {
            "id": "chatcmpl-agg",
            "object": "chat.completion.chunk",
            "created": FIXED_CREATED,
            "model": "auto",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    frames.append("data: %s\n\n" % json.dumps(
        chunk({"role": "assistant", "content": ""}, None), separators=(",", ":")))
    for piece in split_pieces(answer):
        frames.append("data: %s\n\n" % json.dumps(
            chunk({"content": piece}, None), separators=(",", ":")))
    frames.append("data: %s\n\n" % json.dumps(
        chunk({}, "stop"), separators=(",", ":")))
    frames.append("data: %s\n\n" % json.dumps({
        "id": "chatcmpl-agg",
        "object": "chat.completion.chunk",
        "created": FIXED_CREATED,
        "model": "auto",
        "choices": [],
        "usage": {
            "prompt_tokens": uin,
            "completion_tokens": uout,
            "total_tokens": uin + uout,
        },
    }, separators=(",", ":")))
    frames.append("data: [DONE]\n\n")
    return "".join(frames).encode("utf-8")


def render_anthropic_json(answer, uin, uout):
    return json.dumps({
        "id": "msg_agg",
        "type": "message",
        "role": "assistant",
        "model": "auto",
        "content": [{"type": "text", "text": answer}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": uin, "output_tokens": uout},
    }, separators=(",", ":")).encode("utf-8")


def render_anthropic_stream(answer, uin, uout):
    frames = []

    def frame(event, data):
        frames.append("event: %s\ndata: %s\n\n" % (
            event, json.dumps(data, separators=(",", ":"))))

    frame("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_agg", "type": "message", "role": "assistant",
            "model": "auto", "content": [], "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": uin, "output_tokens": 0},
        },
    })
    frame("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""},
    })
    for piece in split_pieces(answer):
        frame("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece},
        })
    frame("content_block_stop", {"type": "content_block_stop", "index": 0})
    frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": uout},
    })
    frame("message_stop", {"type": "message_stop"})
    return "".join(frames).encode("utf-8")


# ---------------------------------------------------------------- routing

def candidates_for(requires, json_mode):
    rows = []
    for (pid, mid), row in G_MODELS.items():
        if not row["free"]:
            continue
        if requires.get("tools") and not row["tools"]:
            continue
        if requires.get("vision") and not row["vision"]:
            continue
        if requires.get("json") and not row["json"]:
            continue
        min_context = requires.get("min_context")
        if min_context is not None and row["context_length"] < min_context:
            continue
        rows.append((pid, mid, row))
    rows.sort(key=lambda r: (-r[2]["rating_milli"], r[0], r[1]))
    out = []
    seen = set()
    for pid, mid, row in rows:
        if pid in seen:
            continue
        if json_mode and G_PROVIDERS[pid]["wire"] == "anthropic":
            continue
        seen.add(pid)
        out.append((pid, mid, row))
    return out


def serve_request(api, body):
    """Answer one client request. Returns (status, content_type, payload)."""
    global G_COUNTER
    with G_LOCK:
        n = G_COUNTER
        G_COUNTER += 1
    if n < len(G_MANIFEST):
        manifest = G_MANIFEST[n]
        request_id = manifest["id"]
        requires = manifest.get("requires", {})
    else:
        request_id = "r%02d" % (n + 1)
        requires = {}
    norm = extract_openai(body) if api == "openai" else extract_anthropic(body)
    stream = norm["stream"]
    for pid, mid, row in candidates_for(requires, norm["json_mode"]):
        if pid in G_DEAD:
            continue
        provider = G_PROVIDERS[pid]
        wire = provider["wire"]
        if wire == "openai":
            path, pbody = build_openai_request(norm, mid, stream)
        elif wire == "anthropic":
            path, pbody = build_anthropic_request(norm, mid, stream)
        else:
            path, pbody = build_gemini_request(norm, mid, stream)
        url = provider["base_url"].rstrip("/") + path
        headers = provider_headers(wire, provider["auth"], provider["key"])
        status, raw = http_call("POST", url, pbody, headers, timeout=PROVIDER_TIMEOUT)
        if status < 200 or status >= 300:
            log("%s %s/%s refused %s" % (request_id, pid, mid, status))
            with G_LOCK:
                streak = G_FAIL_STREAK.get(pid, 0) + 1
                G_FAIL_STREAK[pid] = streak
                if streak >= 3:
                    G_DEAD.add(pid)
            continue
        parsed = parse_provider_answer(wire, raw, stream)
        if parsed is None:
            log("%s %s/%s unparseable answer" % (request_id, pid, mid))
            continue
        answer, uin, uout = parsed
        with G_LOCK:
            G_FAIL_STREAK[pid] = 0
        log("%s served by %s/%s (%d)" % (request_id, pid, mid, row["rating_milli"]))
        if api == "openai":
            if stream:
                return 200, "text/event-stream; charset=utf-8", render_openai_stream(answer, uin, uout)
            return 200, "application/json", render_openai_json(answer, uin, uout)
        if stream:
            return 200, "text/event-stream; charset=utf-8", render_anthropic_stream(answer, uin, uout)
        return 200, "application/json", render_anthropic_json(answer, uin, uout)
    log("%s no provider could serve it" % request_id)
    if api == "openai":
        payload = json.dumps({"error": {
            "message": "all providers refused the request",
            "type": "api_error", "param": None, "code": "all_providers_failed",
        }}, separators=(",", ":")).encode("utf-8")
    else:
        payload = json.dumps({"type": "error", "error": {
            "type": "api_error", "message": "all providers refused the request",
        }}, separators=(",", ":")).encode("utf-8")
    return 503, "application/json", payload


# ---------------------------------------------------------------- http server

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, status, content_type, payload):
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        return self.rfile.read(length) if length > 0 else b""

    def _models_payload(self):
        auth = self.headers.get("Authorization") or ""
        if auth.startswith("Bearer "):
            wire = "openai"
        elif self.headers.get("x-api-key"):
            wire = "anthropic"
        else:
            wire = "openai"
        models = free_model_list()
        if wire == "openai":
            return {
                "object": "list",
                "data": [{
                    "id": m["id"], "object": "model", "created": FIXED_CREATED,
                    "owned_by": "aggregator", "free": True,
                    "context_length": m["context_length"], "tools": m["tools"],
                    "vision": m["vision"], "json": m["json"],
                } for m in models],
            }
        return {
            "data": [{
                "type": "model", "id": m["id"], "display_name": m["id"],
                "created_at": CREATED_AT, "free": True,
                "context_length": m["context_length"], "tools": m["tools"],
                "vision": m["vision"], "json": m["json"],
            } for m in models],
            "has_more": False,
        }

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/v1/models":
            payload = json.dumps(self._models_payload(), separators=(",", ":"))
            self._send(200, "application/json", payload)
        else:
            self._send(404, "application/json", '{"error":"not_found"}')

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/v1/chat/completions":
            api = "openai"
        elif path == "/v1/messages":
            api = "anthropic"
        else:
            self._send(404, "application/json", '{"error":"not_found"}')
            return
        raw = self._read_body()
        try:
            body = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._send(400, "application/json", '{"error":"bad_json"}')
            return
        if not isinstance(body, dict):
            self._send(400, "application/json", '{"error":"bad_json"}')
            return
        try:
            status, content_type, payload = serve_request(api, body)
        except Exception as err:  # never leave the client hanging
            log("handler error: %s" % err)
            status, content_type, payload = 500, "application/json", b'{"error":"internal"}'
        self._send(status, content_type, payload)


# ---------------------------------------------------------------- main

def main(argv):
    global G_PROVIDERS, G_MODELS, G_MANIFEST, G_CATALOG
    list_mode = "--list" in argv
    brief_path = os.environ.get("MAKEFUN_PROMPT_FILE") or ""
    env_url = (os.environ.get("MAKEFUN_ENV_URL") or "").rstrip("/")
    brief = load_brief(brief_path) if brief_path else None
    if brief is not None:
        providers_doc, ratings_doc, workload_doc = brief
        G_CATALOG = build_catalog(providers_doc, ratings_doc)
        for p in providers_doc.get("providers", []):
            G_PROVIDERS[p["id"]] = {
                "wire": p["wire"], "auth": p["auth"], "key": p.get("key", ""),
                "key_env": p.get("key_env", ""), "path": p["path"],
                "base_url": env_url + p["path"] if env_url else p["path"],
            }
        for r in ratings_doc.get("ratings", []):
            G_MODELS[(r["provider"], r["model"])] = {
                "free": bool(r["free"]), "context_length": r["context_length"],
                "tools": bool(r["tools"]), "vision": bool(r["vision"]),
                "json": bool(r["json"]), "rating_milli": r["rating_milli"],
            }
        G_MANIFEST = workload_doc.get("manifest", [])
    catalog_text = json.dumps(G_CATALOG, separators=(",", ":"))
    if list_mode:
        sys.stdout.write(catalog_text + "\n")
        return 0
    if brief is None or not env_url:
        sys.stderr.write(
            "free-model-aggregator-01: needs MAKEFUN_PROMPT_FILE and MAKEFUN_ENV_URL\n")
        return 1
    # Absolute provider base URLs from the env's instance route when available.
    for _ in range(50):
        status, raw = http_call("GET", env_url + "/instance", timeout=CONTROL_TIMEOUT)
        if status == 200:
            try:
                instance = json.loads(raw.decode("utf-8"))
                for p in instance.get("providers", []):
                    if p["id"] in G_PROVIDERS:
                        G_PROVIDERS[p["id"]]["base_url"] = p["base_url"]
                        G_PROVIDERS[p["id"]]["key"] = p.get("key", G_PROVIDERS[p["id"]]["key"])
                break
            except (ValueError, UnicodeDecodeError, KeyError, TypeError):
                break
        time.sleep(0.2)
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    registered = False
    for _ in range(50):
        status, raw = http_call("POST", env_url + "/register", {
            "protocol_version": PROTOCOL_VERSION,
            "port": port,
            "catalog": G_CATALOG,
        }, timeout=CONTROL_TIMEOUT)
        if status == 200:
            registered = True
            break
        time.sleep(0.2)
    if not registered:
        sys.stderr.write("free-model-aggregator-01: could not register with the env\n")
        server.shutdown()
        server.server_close()
        return 1
    log("registered on port %d, %d requests" % (port, len(G_MANIFEST)))
    while True:
        status, raw = http_call("GET", env_url + "/work", timeout=CONTROL_TIMEOUT)
        if status == 200:
            try:
                work = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                work = {}
            if isinstance(work, dict) and work.get("done") is True:
                break
        time.sleep(POLL_INTERVAL)
    server.shutdown()
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
