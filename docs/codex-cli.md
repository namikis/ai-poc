# Codex CLI メモ

## インストール

```bash
# Node.js 22以上が必要
nvm install 22
nvm use 22

# インストール
npm install -g @openai/codex

# バージョン確認
codex --version
```

## 認証

```bash
# ChatGPTアカウントでサインイン（初回のみ）
codex login
```

## 基本的な使い方

### 対話モード（デフォルト）

```bash
# プロンプトなしで起動
codex

# プロンプト付きで起動
codex "このプロジェクトの構造を説明して"
```

### 非対話モード（`exec`）

スクリプトやCI等からプログラム的に呼び出す場合に使用。

```bash
# 基本
codex exec "プロンプト"

# フルオート（承認なし + サンドボックス書き込み可）
codex exec --full-auto "プロンプト"

# 読み取り専用サンドボックス（ファイル変更を防止）
codex exec --full-auto -s read-only "プロンプト"

# stdinからプロンプトを読む
echo "プロンプト" | codex exec -

# JSONL形式で出力
codex exec --json "プロンプト"

# 最後のメッセージをファイルに保存
codex exec -o output.txt "プロンプト"
```

## 主要オプション

| オプション | 説明 |
|-----------|------|
| `-m, --model <MODEL>` | 使用するモデルを指定 |
| `-s, --sandbox <MODE>` | `read-only`, `workspace-write`, `danger-full-access` |
| `-a, --ask-for-approval <POLICY>` | `untrusted`, `on-request`, `never` |
| `--full-auto` | `-a on-request -s workspace-write` のエイリアス |
| `-C, --cd <DIR>` | 作業ディレクトリを指定 |
| `-i, --image <FILE>` | 画像を添付 |
| `-c, --config <key=value>` | 設定をオーバーライド（例: `-c model="o3"`） |
| `--search` | Web検索を有効化 |
| `--oss` | ローカルOSSモデルを使用（LM Studio / Ollama） |
| `-p, --profile <PROFILE>` | `~/.codex/config.toml` のプロファイルを指定 |

## サブコマンド一覧

| コマンド | 説明 |
|---------|------|
| `codex exec` | 非対話実行（エイリアス: `codex e`） |
| `codex review` | コードレビューを非対話実行 |
| `codex login` | 認証管理 |
| `codex logout` | 認証情報を削除 |
| `codex mcp` | 外部MCPサーバーの管理 |
| `codex mcp-server` | Codex自体をMCPサーバーとして起動（stdio） |
| `codex resume` | 前回のセッションを再開 |
| `codex fork` | 前回のセッションをフォーク |
| `codex sandbox` | サンドボックス内でコマンド実行 |
| `codex apply` | 最新のdiffをgit applyで適用（エイリアス: `codex a`） |

## nvm経由で使う場合の注意

デフォルトのNode.jsバージョンが22未満の場合、毎回 `nvm use 22` が必要。

```bash
# 一行で実行
source ~/.nvm/nvm.sh && nvm use 22 && codex exec --full-auto "プロンプト"

# Node 22をデフォルトにする場合
nvm alias default 22
```

## 設定ファイル

`~/.codex/config.toml` で永続的な設定が可能。

```toml
model = "o3"

[sandbox_permissions]
# 必要に応じてカスタマイズ
```

## 環境情報（このマシン）

- Codex CLI: v0.115.0
- Node.js: v22.22.1（nvm経由）
- デフォルトモデル: gpt-5.4
