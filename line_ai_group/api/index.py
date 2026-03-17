import hashlib
import hmac
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Callable
from uuid import uuid4

import anthropic
import openai
from flask import Flask, Response, request

from lib.loop_guard import is_duplicate_event

app = Flask(__name__)

MENTION_RE = re.compile(r"<@[^>]+>")
SESSION_RE = re.compile(
    r"\[\[AI_SESSION\|id=(?P<id>[^|]+)\|turn=(?P<turn>\d+)\|"
    r"max=(?P<max>\d+)\|claude_left=(?P<claude_left>\d+)\]\]"
)
DUO_KEYWORDS = ("議論", "会話", "壁打ち", "相談", "2人で", "二人で", "duo")
SLACK_API_URL = "https://slack.com/api"
DEFAULT_MAX_TURNS = 3
DEFAULT_CLAUDE_TURNS = 1
GPT_MAX_TOKENS = 400
CLAUDE_MAX_TOKENS = 220

CLAUDE_CLIENT = anthropic.Anthropic()
OPENAI_CLIENT = openai.OpenAI()
BOT_USER_ID_CACHE: dict[str, str] = {}

app = Flask(__name__)


def _verify_slack_request(signing_secret: str) -> bool:
    timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
    signature = request.headers.get("X-Slack-Signature", "")

    if not timestamp or not signature:
        app.logger.warning("Slack request missing signature headers")
        return False

    try:
        request_time = int(timestamp)
    except ValueError:
        app.logger.warning("Slack request had invalid timestamp: %s", timestamp)
        return False

    if abs(time.time() - request_time) > 60 * 5:
        app.logger.warning("Slack request timestamp too old: %s", timestamp)
        return False

    raw_body = request.get_data(cache=True)
    base_string = b"v0:" + timestamp.encode("utf-8") + b":" + raw_body
    expected = "v0=" + hmac.new(
        signing_secret.encode("utf-8"),
        base_string,
        hashlib.sha256,
    ).hexdigest()
    matched = hmac.compare_digest(expected, signature)
    if not matched:
        app.logger.warning(
            "Slack signature mismatch for path=%s timestamp=%s",
            request.path,
            timestamp,
        )
    return matched


def _slack_api_request(bot_token: str, method: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{SLACK_API_URL}/{method}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {bot_token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    if not body.get("ok"):
        raise RuntimeError(f"Slack API error: {body.get('error', 'unknown_error')}")
    return body


def _post_slack_message(bot_token: str, channel: str, text: str, thread_ts: str) -> None:
    _slack_api_request(
        bot_token,
        "chat.postMessage",
        {
            "channel": channel,
            "text": text,
            "thread_ts": thread_ts,
        },
    )


def _post_error_message(bot_token: str, channel: str, thread_ts: str, message: str) -> None:
    try:
        _post_slack_message(bot_token, channel, message, thread_ts)
    except Exception:
        app.logger.exception("Failed to post Slack error message")


def _lookup_bot_user_id(bot_token: str) -> str:
    if bot_token in BOT_USER_ID_CACHE:
        return BOT_USER_ID_CACHE[bot_token]
    body = _slack_api_request(bot_token, "auth.test", {})
    user_id = body["user_id"]
    BOT_USER_ID_CACHE[bot_token] = user_id
    return user_id


def _extract_prompt(text: str) -> str:
    text = SESSION_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    return text.strip()


def _wants_duo(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(keyword in prompt for keyword in DUO_KEYWORDS) or "duo" in lowered


def _build_session_marker(session_id: str, turn: int, max_turns: int, claude_left: int) -> str:
    return (
        f"[[AI_SESSION|id={session_id}|turn={turn}|max={max_turns}|"
        f"claude_left={claude_left}]]"
    )


def _parse_session_marker(text: str) -> dict[str, int | str] | None:
    match = SESSION_RE.search(text)
    if not match:
        return None
    return {
        "id": match.group("id"),
        "turn": int(match.group("turn")),
        "max": int(match.group("max")),
        "claude_left": int(match.group("claude_left")),
    }


def _build_single_prompt(bot_name: str, user_prompt: str) -> str:
    return (
        f"あなたは{bot_name}です。日本語で簡潔に答えてください。"
        "長くても120語程度に抑え、必要なら箇条書きで整理してください。\n\n"
        f"ユーザーの入力:\n{user_prompt}"
    )


def _build_duo_prompt(bot_name: str, user_prompt: str, turn: int, max_turns: int) -> str:
    return (
        f"あなたは{bot_name}です。Slack上で別のAIと短く議論しています。"
        "日本語で、120語以内、1つの反論か補足と1つの新しい論点だけを返してください。"
        "前置きやメタ説明は不要です。\n\n"
        f"現在ターン: {turn}/{max_turns}\n"
        f"相手またはユーザーの直前メッセージ:\n{user_prompt}"
    )


def _reply_with_gpt(prompt: str, duo_mode: bool, turn: int, max_turns: int) -> str:
    response = OPENAI_CLIENT.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    _build_duo_prompt("GPT", prompt, turn, max_turns)
                    if duo_mode
                    else _build_single_prompt("GPT", prompt)
                ),
            }
        ],
        max_tokens=GPT_MAX_TOKENS,
    )
    return (response.choices[0].message.content or "").strip()


def _reply_with_claude(prompt: str, duo_mode: bool, turn: int, max_turns: int) -> str:
    response = CLAUDE_CLIENT.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=CLAUDE_MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": (
                    _build_duo_prompt("Claude", prompt, turn, max_turns)
                    if duo_mode
                    else _build_single_prompt("Claude", prompt)
                ),
            }
        ],
    )
    return response.content[0].text.strip()


def _maybe_build_handoff(
    current_bot: str,
    current_bot_token: str,
    counterpart_bot_token: str,
    response_text: str,
    session: dict[str, int | str],
) -> str:
    next_turn = int(session["turn"]) + 1
    max_turns = int(session["max"])
    claude_left = int(session["claude_left"])
    if current_bot == "claude":
        claude_left -= 1

    next_bot = "claude" if current_bot == "gpt" else "gpt"
    if next_turn > max_turns:
        return response_text + "\n\n会話はここで終了します。"
    if next_bot == "claude" and claude_left <= 0:
        return response_text + "\n\nClaudeの発話上限に達したので会話を終了します。"

    next_user_id = _lookup_bot_user_id(counterpart_bot_token)
    marker = _build_session_marker(str(session["id"]), next_turn, max_turns, claude_left)
    baton = (
        f"\n\n<@{next_user_id}> {marker}\n"
        "前の発言を受けて、短く応答してください。"
    )
    return response_text + baton


def _handle_slack_request(
    bot_name: str,
    signing_secret_env: str,
    bot_token_env: str,
    counterpart_bot_token_env: str,
    responder: Callable[[str, bool, int, int], str],
) -> Response:
    signing_secret = os.environ[signing_secret_env]
    bot_token = os.environ[bot_token_env]
    counterpart_bot_token = os.environ[counterpart_bot_token_env]

    app.logger.info(
        "Slack request received path=%s content_type=%s",
        request.path,
        request.headers.get("Content-Type", ""),
    )

    if not _verify_slack_request(signing_secret):
        return Response("Bad Request", status=400, mimetype="text/plain")

    payload = request.get_json(silent=True) or {}
    app.logger.info(
        "Slack payload type=%s event_type=%s",
        payload.get("type"),
        (payload.get("event") or {}).get("type"),
    )

    if payload.get("type") == "url_verification":
        app.logger.info("Slack url_verification succeeded for path=%s", request.path)
        return Response(payload.get("challenge", ""), mimetype="text/plain")

    if payload.get("type") != "event_callback":
        return Response("OK", mimetype="text/plain")

    event = payload.get("event") or {}
    if payload.get("event_id") and is_duplicate_event(payload["event_id"]):
        return Response("OK", mimetype="text/plain")
    if event.get("type") != "app_mention":
        return Response("OK", mimetype="text/plain")

    text = event.get("text", "")
    session = _parse_session_marker(text)
    is_human_start = not session and not event.get("bot_id") and event.get("subtype") != "bot_message"
    if not session and not is_human_start:
        return Response("OK", mimetype="text/plain")

    prompt = _extract_prompt(text)
    if not prompt:
        prompt = "自己紹介して。"

    duo_mode = bool(session) or _wants_duo(prompt)
    if session:
        turn = int(session["turn"])
        max_turns = int(session["max"])
    else:
        turn = 1
        max_turns = DEFAULT_MAX_TURNS
        session = {
            "id": uuid4().hex[:8],
            "turn": turn,
            "max": max_turns,
            "claude_left": DEFAULT_CLAUDE_TURNS,
        }

    channel = event.get("channel")
    thread_ts = event.get("thread_ts") or event.get("ts")
    if not channel or not thread_ts:
        app.logger.warning("Slack event missing channel or thread_ts")
        return Response("OK", mimetype="text/plain")

    try:
        reply_text = responder(prompt, duo_mode, turn, max_turns)
        if duo_mode:
            reply_text = _maybe_build_handoff(
                bot_name.lower(),
                bot_token,
                counterpart_bot_token,
                reply_text,
                session,
            )
        _post_slack_message(bot_token, channel, reply_text, thread_ts)
        app.logger.info("Slack reply posted path=%s channel=%s", request.path, channel)
    except anthropic.APIStatusError as exc:
        app.logger.exception("Anthropic API request failed")
        _post_error_message(
            bot_token,
            channel,
            thread_ts,
            f"Claude API error: {exc.message or 'request failed'}",
        )
    except openai.APIStatusError as exc:
        app.logger.exception("OpenAI API request failed")
        _post_error_message(
            bot_token,
            channel,
            thread_ts,
            f"OpenAI API error: {exc.message or 'request failed'}",
        )
    except urllib.error.HTTPError:
        app.logger.exception("Slack API request failed")
    except urllib.error.URLError:
        app.logger.exception("Slack API connection failed")
        _post_error_message(
            bot_token,
            channel,
            thread_ts,
            "Slack API connection failed.",
        )
    except Exception:
        app.logger.exception("Slack event handling failed")
        _post_error_message(
            bot_token,
            channel,
            thread_ts,
            "Unexpected server error.",
        )

    return Response("OK", mimetype="text/plain")


@app.post("/slack/events/gpt")
def slack_gpt() -> Response:
    return _handle_slack_request(
        "GPT",
        "SLACK_GPT_SIGNING_SECRET",
        "SLACK_GPT_BOT_TOKEN",
        "SLACK_CLAUDE_BOT_TOKEN",
        _reply_with_gpt,
    )


@app.post("/slack/events/claude")
def slack_claude() -> Response:
    return _handle_slack_request(
        "Claude",
        "SLACK_CLAUDE_SIGNING_SECRET",
        "SLACK_CLAUDE_BOT_TOKEN",
        "SLACK_GPT_BOT_TOKEN",
        _reply_with_claude,
    )
