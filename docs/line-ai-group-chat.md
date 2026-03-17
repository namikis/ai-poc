# LINE AIグループチャット 実装手順書

Claude と OpenAI（GPT）の2体構成で、ユーザーを含む3人のLINEグループチャットを構築する。

## 構成図

```
LINE Group Chat
  ├── ユーザー（あなた）
  ├── Claude Bot（LINE Bot → Anthropic API）
  └── GPT Bot（LINE Bot → OpenAI API）

┌────────────┐    Webhook     ┌──────────────────┐    API     ┌──────────────┐
│  LINE App   │ ─────────────▶│  中継サーバー      │ ─────────▶│ Anthropic API │
│ (グループ)   │ ◀─────────────│  (FastAPI等)      │ ─────────▶│ OpenAI API    │
└────────────┘   Reply API    └──────────────────┘           └──────────────┘
```

## 前提条件

- Python 3.11+
- LINE アカウント
- Anthropic API キー（従量課金 / クレジットカード登録済み）
- OpenAI API キー（Tier 1以上 / $5支払い済み）

---

## Step 1: API キーの取得

### 1-1. Anthropic API

1. [Anthropic Console](https://console.anthropic.com/) にアクセス
2. API Keys → Create Key
3. 推奨モデル: `claude-haiku-4-5-20251001`（コスト重視）または `claude-sonnet-4-6`（品質重視）

### 1-2. OpenAI API

1. [OpenAI Platform](https://platform.openai.com/) にアクセス
2. API Keys → Create new secret key
3. 推奨モデル: `gpt-4o-mini`（コスト重視）または `gpt-4o`（品質重視）

---

## Step 2: LINE Bot の作成（チャネル作成まで）

### 2-1. LINE Developers コンソールでチャネル作成

1. [LINE Developers](https://developers.line.biz/) にログイン
2. プロバイダーを作成（未作成の場合）
3. **Messaging API チャネルを2つ作成**:
   - `Claude Bot` — Claude応答用
   - `GPT Bot` — GPT応答用
4. 各チャネルで以下を取得:
   - **チャネルシークレット**（Basic settings）
   - **チャネルアクセストークン**（Messaging API → Issue）

> Webhook URL の設定は Step 5（デプロイ後）に行う。

---

## Step 3: 中継サーバーの実装

### 3-1. プロジェクト構成（Vercel Serverless Functions）

```
line_ai_group/
├── api/
│   ├── webhook_claude.py   # Claude Bot の Webhook エンドポイント
│   └── webhook_gpt.py      # GPT Bot の Webhook エンドポイント
├── lib/
│   └── loop_guard.py       # 無限ループ防止ユーティリティ
├── requirements.txt
└── vercel.json
```

### 3-2. 依存パッケージ

```txt
# requirements.txt
line-bot-sdk>=3.0.0
anthropic>=0.40.0
openai>=1.50.0
```

### 3-3. 環境変数

Vercel ダッシュボード（Settings → Environment Variables）で設定する:

```
CLAUDE_CHANNEL_SECRET=xxx
CLAUDE_CHANNEL_ACCESS_TOKEN=xxx
GPT_CHANNEL_SECRET=xxx
GPT_CHANNEL_ACCESS_TOKEN=xxx
ANTHROPIC_API_KEY=sk-ant-xxx
OPENAI_API_KEY=sk-xxx
CLAUDE_BOT_USER_ID=Uxxxxxxx
GPT_BOT_USER_ID=Uxxxxxxx
```

### 3-4. Vercel 設定

```json
// vercel.json
{
  "rewrites": [
    { "source": "/webhook/claude", "destination": "/api/webhook_claude" },
    { "source": "/webhook/gpt", "destination": "/api/webhook_gpt" }
  ]
}
```

### 3-5. サーバー実装

**api/webhook_claude.py**

```python
from http.server import BaseHTTPRequestHandler
from linebot.v3 import WebhookParser
from linebot.v3.messaging import (
    ApiClient, MessagingApi, Configuration,
    ReplyMessageRequest, TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent
import anthropic
import json
import os

config = Configuration(access_token=os.environ["CLAUDE_CHANNEL_ACCESS_TOKEN"])
parser = WebhookParser(os.environ["CLAUDE_CHANNEL_SECRET"])
ai_client = anthropic.Anthropic()

BOT_USER_IDS = {
    os.environ.get("CLAUDE_BOT_USER_ID", ""),
    os.environ.get("GPT_BOT_USER_ID", ""),
}


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode()
        signature = self.headers.get("X-Line-Signature", "")

        events = parser.parse(body, signature)

        for event in events:
            if not isinstance(event, MessageEvent):
                continue
            if not isinstance(event.message, TextMessageContent):
                continue
            if event.source.user_id in BOT_USER_IDS:
                continue

            response = ai_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                messages=[{"role": "user", "content": event.message.text}],
            )
            reply_text = response.content[0].text

            with ApiClient(config) as api:
                line_api = MessagingApi(api)
                line_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=reply_text)],
                    )
                )

        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")
```

**api/webhook_gpt.py**

```python
from http.server import BaseHTTPRequestHandler
from linebot.v3 import WebhookParser
from linebot.v3.messaging import (
    ApiClient, MessagingApi, Configuration,
    ReplyMessageRequest, TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent
import openai
import json
import os

config = Configuration(access_token=os.environ["GPT_CHANNEL_ACCESS_TOKEN"])
parser = WebhookParser(os.environ["GPT_CHANNEL_SECRET"])
ai_client = openai.OpenAI()

BOT_USER_IDS = {
    os.environ.get("CLAUDE_BOT_USER_ID", ""),
    os.environ.get("GPT_BOT_USER_ID", ""),
}


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode()
        signature = self.headers.get("X-Line-Signature", "")

        events = parser.parse(body, signature)

        for event in events:
            if not isinstance(event, MessageEvent):
                continue
            if not isinstance(event.message, TextMessageContent):
                continue
            if event.source.user_id in BOT_USER_IDS:
                continue

            response = ai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": event.message.text}],
                max_tokens=1024,
            )
            reply_text = response.choices[0].message.content

            with ApiClient(config) as api:
                line_api = MessagingApi(api)
                line_api.reply_message(
                    ReplyMessageRequest(
                        reply_token=event.reply_token,
                        messages=[TextMessage(text=reply_text)],
                    )
                )

        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")
```

---

## Step 4: 無限ループ防止

Bot 同士が互いのメッセージに反応し続ける問題を防ぐ。

### 方針

LINE グループ内では、Bot が送信したメッセージに対して Webhook は**発火しない**（LINE の仕様）。
ただし、**別の Bot のメッセージには Webhook が発火する**ため、各 Bot の `user_id` を環境変数に登録し、Step 3 のコード内で判定・スキップしている。

加えて、LINE 側の再送や同一イベントの重複処理を避けるため、`lib/loop_guard.py` で `webhook_event_id` を短期間メモリ保持して二重応答も防ぐ。

### Bot の user_id の取得方法

デプロイ後、各 Bot の user_id は LINE Developers コンソールの「Basic settings」→「Your user ID」、
または Bot Profile API で取得できる:

```bash
curl -H "Authorization: Bearer {CHANNEL_ACCESS_TOKEN}" \
  https://api.line.me/v2/bot/info
```

レスポンスの `userId` を環境変数 `CLAUDE_BOT_USER_ID` / `GPT_BOT_USER_ID` に設定する。

---

## Step 5: Vercel にデプロイ

### 5-1. Vercel CLI インストール & デプロイ

```bash
# Vercel CLI インストール
npm install -g vercel

# プロジェクトディレクトリに移動
cd line_ai_group

# デプロイ（初回はプロジェクト設定の対話あり）
vercel
```

### 5-2. 環境変数の設定

```bash
# CLI から設定する場合
vercel env add CLAUDE_CHANNEL_SECRET
vercel env add CLAUDE_CHANNEL_ACCESS_TOKEN
vercel env add GPT_CHANNEL_SECRET
vercel env add GPT_CHANNEL_ACCESS_TOKEN
vercel env add ANTHROPIC_API_KEY
vercel env add OPENAI_API_KEY
vercel env add CLAUDE_BOT_USER_ID
vercel env add GPT_BOT_USER_ID
```

または Vercel ダッシュボード（Settings → Environment Variables）から設定。

ローカル確認用に `line_ai_group/.env.example` を複製して `.env` を作っておくと管理しやすい:

```bash
cd line_ai_group
cp .env.example .env
```

### 5-3. 本番デプロイ

```bash
# 環境変数設定後に本番デプロイ
vercel --prod
```

デプロイ後、URL が確定する（例: `https://line-ai-group.vercel.app`）。

### 注意: Vercel Serverless Functions のタイムアウト

- Hobby プラン: **最大 10 秒**
- Pro プラン ($20/月): **最大 60 秒**
- AI API の応答が遅い場合タイムアウトする可能性あり。Hobby プランで問題が出る場合は Pro を検討

---

## Step 6: Webhook URL の設定

デプロイ完了後、サーバーの URL が確定するので LINE 側に設定する。

各チャネルの Messaging API 設定で:
- Webhook URL:
  - Claude Bot: `https://<デプロイ先URL>/webhook/claude`
  - GPT Bot: `https://<デプロイ先URL>/webhook/gpt`
- Webhook の利用: ON
- 応答メッセージ: OFF（自動応答を無効化）
- あいさつメッセージ: OFF

---

## Step 7: グループに招待

1. LINE アプリで新規グループを作成
2. 2つの Bot を友だち追加し、グループに招待

---

## Step 8: 動作確認チェックリスト

- [ ] Anthropic / OpenAI の API キーが取得済み
- [ ] LINE Developers で2つのチャネルが作成されている
- [ ] 中継サーバーがデプロイされ、HTTPS でアクセス可能
- [ ] 各チャネルの Webhook URL にデプロイ先 URL が設定されている
- [ ] `.env` に全ての環境変数が設定されている
- [ ] グループに2つの Bot が参加している
- [ ] ユーザーのメッセージに Claude Bot が応答する
- [ ] ユーザーのメッセージに GPT Bot が応答する
- [ ] Claude / GPT のどちらかの発言をもう片方が拾って無限往復しない

---

## 推定コスト（月間）

| 項目 | 試算（1日50メッセージ想定） |
|------|--------------------------|
| Claude API (Haiku) | ~$0.50/月 |
| OpenAI API (GPT-4o Mini) | ~$0.10/月 |
| LINE Messaging API | 無料（月200通まで無料、超過 ~¥0） |
| サーバー | $0〜5/月（無料枠利用時） |
| **合計** | **~$1〜6/月** |

---

## 発展: 会話コンテキストの共有

現状の実装では各メッセージが独立した問い合わせになる。
グループチャットらしい会話の流れを実現するには:

1. **会話履歴の保持** — Redis や SQLite でグループごとの直近N件を保存
2. **相手の発言も含める** — Claude に GPT の発言も渡す（逆も同様）
3. **システムプロンプトで役割設定** — 例: 「あなたはClaudeです。GPTという別のAIも同じグループにいます。」
