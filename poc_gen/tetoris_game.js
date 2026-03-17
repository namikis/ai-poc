// テトリミノの定義（各ピースの回転状態）
const TETROMINOS = {
    I: {
        shape: [
            [[0,0,0,0], [1,1,1,1], [0,0,0,0], [0,0,0,0]],
            [[0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0]],
            [[0,0,0,0], [0,0,0,0], [1,1,1,1], [0,0,0,0]],
            [[0,1,0,0], [0,1,0,0], [0,1,0,0], [0,1,0,0]]
        ],
        color: '#00f0f0'
    },
    O: {
        shape: [
            [[1,1], [1,1]],
            [[1,1], [1,1]],
            [[1,1], [1,1]],
            [[1,1], [1,1]]
        ],
        color: '#f0f000'
    },
    T: {
        shape: [
            [[0,1,0], [1,1,1], [0,0,0]],
            [[0,1,0], [0,1,1], [0,1,0]],
            [[0,0,0], [1,1,1], [0,1,0]],
            [[0,1,0], [1,1,0], [0,1,0]]
        ],
        color: '#a000f0'
    },
    S: {
        shape: [
            [[0,1,1], [1,1,0], [0,0,0]],
            [[0,1,0], [0,1,1], [0,0,1]],
            [[0,0,0], [0,1,1], [1,1,0]],
            [[1,0,0], [1,1,0], [0,1,0]]
        ],
        color: '#00f000'
    },
    Z: {
        shape: [
            [[1,1,0], [0,1,1], [0,0,0]],
            [[0,0,1], [0,1,1], [0,1,0]],
            [[0,0,0], [1,1,0], [0,1,1]],
            [[0,1,0], [1,1,0], [1,0,0]]
        ],
        color: '#f00000'
    },
    J: {
        shape: [
            [[1,0,0], [1,1,1], [0,0,0]],
            [[0,1,1], [0,1,0], [0,1,0]],
            [[0,0,0], [1,1,1], [0,0,1]],
            [[0,1,0], [0,1,0], [1,1,0]]
        ],
        color: '#0000f0'
    },
    L: {
        shape: [
            [[0,0,1], [1,1,1], [0,0,0]],
            [[0,1,0], [0,1,0], [0,1,1]],
            [[0,0,0], [1,1,1], [1,0,0]],
            [[1,1,0], [0,1,0], [0,1,0]]
        ],
        color: '#f0a000'
    }
};

// ゲーム定数
const BOARD_WIDTH = 10;
const BOARD_HEIGHT = 20;
const BLOCK_SIZE = 20;

// ゲーム状態
const gameState = {
    board: [],
    currentPiece: null,
    currentPosition: { x: 0, y: 0 },
    currentRotation: 0,
    nextPiece: null,
    score: 0,
    level: 1,
    lines: 0,
    isGameOver: false,
    isPaused: false,
    dropInterval: 1000,
    lastDropTime: 0,
    pieceTypes: ['I', 'O', 'T', 'S', 'Z', 'J', 'L']
};

// Canvas要素
let canvas, ctx, nextCanvas, nextCtx;

// 初期化
function init() {
    canvas = document.getElementById('gameCanvas');
    ctx = canvas.getContext('2d');
    nextCanvas = document.getElementById('nextCanvas');
    nextCtx = nextCanvas.getContext('2d');

    // ボードの初期化
    gameState.board = Array(BOARD_HEIGHT).fill(null).map(() => Array(BOARD_WIDTH).fill(0));

    // 最初のピースを生成
    gameState.nextPiece = getRandomPiece();
    spawnNewPiece();

    // キーボードイベントの設定
    document.addEventListener('keydown', handleKeyPress);

    // ゲームループ開始
    requestAnimationFrame(gameLoop);
}

// ランダムなピースを取得
function getRandomPiece() {
    const type = gameState.pieceTypes[Math.floor(Math.random() * gameState.pieceTypes.length)];
    return {
        type: type,
        shape: TETROMINOS[type].shape,
        color: TETROMINOS[type].color
    };
}

// 新しいピースをスポーン
function spawnNewPiece() {
    gameState.currentPiece = gameState.nextPiece;
    gameState.nextPiece = getRandomPiece();
    gameState.currentRotation = 0;
    gameState.currentPosition = {
        x: Math.floor(BOARD_WIDTH / 2) - Math.floor(gameState.currentPiece.shape[0][0].length / 2),
        y: 0
    };

    // スポーン位置で衝突する場合はゲームオーバー
    if (checkCollision(gameState.currentPiece, gameState.currentPosition, gameState.currentRotation)) {
        gameOver();
    }
}

// 衝突判定
function checkCollision(piece, position, rotation) {
    const shape = piece.shape[rotation];
    for (let y = 0; y < shape.length; y++) {
        for (let x = 0; x < shape[y].length; x++) {
            if (shape[y][x]) {
                const boardX = position.x + x;
                const boardY = position.y + y;

                // 壁との衝突
                if (boardX < 0 || boardX >= BOARD_WIDTH || boardY >= BOARD_HEIGHT) {
                    return true;
                }

                // ボード上のピースとの衝突（y < 0は許容）
                if (boardY >= 0 && gameState.board[boardY][boardX]) {
                    return true;
                }
            }
        }
    }
    return false;
}

// ピースを移動
function movePiece(dx, dy) {
    const newPosition = {
        x: gameState.currentPosition.x + dx,
        y: gameState.currentPosition.y + dy
    };

    if (!checkCollision(gameState.currentPiece, newPosition, gameState.currentRotation)) {
        gameState.currentPosition = newPosition;
        return true;
    }
    return false;
}

// ピースを回転
function rotatePiece(direction) {
    const numRotations = gameState.currentPiece.shape.length;
    let newRotation = gameState.currentRotation + direction;

    if (newRotation < 0) newRotation = numRotations - 1;
    if (newRotation >= numRotations) newRotation = 0;

    // 回転できるか確認
    if (!checkCollision(gameState.currentPiece, gameState.currentPosition, newRotation)) {
        gameState.currentRotation = newRotation;
        return true;
    }

    // ウォールキック（左右に少しずらして回転を試みる）
    for (let offset of [1, -1, 2, -2]) {
        const newPosition = {
            x: gameState.currentPosition.x + offset,
            y: gameState.currentPosition.y
        };
        if (!checkCollision(gameState.currentPiece, newPosition, newRotation)) {
            gameState.currentPosition = newPosition;
            gameState.currentRotation = newRotation;
            return true;
        }
    }

    return false;
}

// ピースをボードに固定
function lockPiece() {
    const shape = gameState.currentPiece.shape[gameState.currentRotation];
    for (let y = 0; y < shape.length; y++) {
        for (let x = 0; x < shape[y].length; x++) {
            if (shape[y][x]) {
                const boardY = gameState.currentPosition.y + y;
                const boardX = gameState.currentPosition.x + x;
                if (boardY >= 0) {
                    gameState.board[boardY][boardX] = gameState.currentPiece.color;
                }
            }
        }
    }

    // ラインをチェックして消去
    checkAndClearLines();

    // 新しいピースをスポーン
    spawnNewPiece();
}

// ラインのチェックと消去
function checkAndClearLines() {
    let linesCleared = 0;

    for (let y = BOARD_HEIGHT - 1; y >= 0; y--) {
        if (gameState.board[y].every(cell => cell !== 0)) {
            // ラインを削除
            gameState.board.splice(y, 1);
            // 上に空のラインを追加
            gameState.board.unshift(Array(BOARD_WIDTH).fill(0));
            linesCleared++;
            y++; // 同じ行を再チェック
        }
    }

    if (linesCleared > 0) {
        // スコア計算
        const points = [0, 100, 300, 500, 800];
        gameState.score += points[linesCleared] * gameState.level;
        gameState.lines += linesCleared;

        // レベルアップ（10ライン毎）
        const newLevel = Math.floor(gameState.lines / 10) + 1;
        if (newLevel > gameState.level) {
            gameState.level = newLevel;
            gameState.dropInterval = Math.max(100, 1000 - (gameState.level - 1) * 50);
        }

        updateUI();
    }
}

// UI更新
function updateUI() {
    document.getElementById('score').textContent = gameState.score;
    document.getElementById('level').textContent = gameState.level;
    document.getElementById('lines').textContent = gameState.lines;
}

// ゲームオーバー
function gameOver() {
    gameState.isGameOver = true;
    const overlay = document.getElementById('gameOverlay');
    overlay.classList.remove('hidden');
}

// ゲームリセット
function resetGame() {
    gameState.board = Array(BOARD_HEIGHT).fill(null).map(() => Array(BOARD_WIDTH).fill(0));
    gameState.score = 0;
    gameState.level = 1;
    gameState.lines = 0;
    gameState.isGameOver = false;
    gameState.isPaused = false;
    gameState.dropInterval = 1000;
    gameState.lastDropTime = 0;

    gameState.nextPiece = getRandomPiece();
    spawnNewPiece();

    updateUI();

    const overlay = document.getElementById('gameOverlay');
    overlay.classList.add('hidden');
}

// キーボード入力処理
function handleKeyPress(e) {
    if (gameState.isGameOver) {
        if (e.key === 'r' || e.key === 'R') {
            resetGame();
        }
        return;
    }

    if (e.key === 'p' || e.key === 'P') {
        gameState.isPaused = !gameState.isPaused;
        const overlay = document.getElementById('gameOverlay');
        const title = document.getElementById('overlayTitle');
        const message = document.getElementById('overlayMessage');

        if (gameState.isPaused) {
            overlay.classList.remove('hidden');
            title.textContent = 'PAUSED';
            message.textContent = 'Press P to continue';
        } else {
            overlay.classList.add('hidden');
        }
        return;
    }

    if (gameState.isPaused) return;

    switch(e.key) {
        case 'ArrowLeft':
            movePiece(-1, 0);
            break;
        case 'ArrowRight':
            movePiece(1, 0);
            break;
        case 'ArrowDown':
            if (movePiece(0, 1)) {
                gameState.score += 1; // ソフトドロップボーナス
                updateUI();
            }
            break;
        case 'ArrowUp':
        case 'z':
        case 'Z':
            rotatePiece(-1);
            break;
        case ' ':
        case 'x':
        case 'X':
            rotatePiece(1);
            break;
        case 'r':
        case 'R':
            resetGame();
            break;
    }

    e.preventDefault();
}

// メインゲームループ
function gameLoop(timestamp) {
    if (!gameState.isGameOver && !gameState.isPaused) {
        // 自動落下
        if (timestamp - gameState.lastDropTime > gameState.dropInterval) {
            if (!movePiece(0, 1)) {
                lockPiece();
            }
            gameState.lastDropTime = timestamp;
        }
    }

    // 描画
    render();

    requestAnimationFrame(gameLoop);
}

// 描画
function render() {
    // メインキャンバスをクリア
    ctx.fillStyle = '#0a0a15';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // グリッドを描画
    ctx.strokeStyle = '#1a1a2e';
    ctx.lineWidth = 1;
    for (let y = 0; y <= BOARD_HEIGHT; y++) {
        ctx.beginPath();
        ctx.moveTo(0, y * BLOCK_SIZE);
        ctx.lineTo(BOARD_WIDTH * BLOCK_SIZE, y * BLOCK_SIZE);
        ctx.stroke();
    }
    for (let x = 0; x <= BOARD_WIDTH; x++) {
        ctx.beginPath();
        ctx.moveTo(x * BLOCK_SIZE, 0);
        ctx.lineTo(x * BLOCK_SIZE, BOARD_HEIGHT * BLOCK_SIZE);
        ctx.stroke();
    }

    // ボード上の固定されたピースを描画
    for (let y = 0; y < BOARD_HEIGHT; y++) {
        for (let x = 0; x < BOARD_WIDTH; x++) {
            if (gameState.board[y][x]) {
                drawBlock(ctx, x, y, gameState.board[y][x]);
            }
        }
    }

    // 現在のピースを描画
    if (gameState.currentPiece && !gameState.isGameOver) {
        const shape = gameState.currentPiece.shape[gameState.currentRotation];
        for (let y = 0; y < shape.length; y++) {
            for (let x = 0; x < shape[y].length; x++) {
                if (shape[y][x]) {
                    const boardX = gameState.currentPosition.x + x;
                    const boardY = gameState.currentPosition.y + y;
                    if (boardY >= 0) {
                        drawBlock(ctx, boardX, boardY, gameState.currentPiece.color);
                    }
                }
            }
        }
    }

    // 次のピースを描画
    renderNextPiece();
}

// ブロックを描画
function drawBlock(context, x, y, color) {
    context.fillStyle = color;
    context.fillRect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);

    // ボーダー
    context.strokeStyle = '#000';
    context.lineWidth = 2;
    context.strokeRect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
}

// 次のピースを描画
function renderNextPiece() {
    // キャンバスをクリア
    nextCtx.fillStyle = '#0a0a15';
    nextCtx.fillRect(0, 0, nextCanvas.width, nextCanvas.height);

    if (gameState.nextPiece) {
        const shape = gameState.nextPiece.shape[0];
        const blockSize = 16;

        // 中央に配置
        const offsetX = (nextCanvas.width - shape[0].length * blockSize) / 2;
        const offsetY = (nextCanvas.height - shape.length * blockSize) / 2;

        for (let y = 0; y < shape.length; y++) {
            for (let x = 0; x < shape[y].length; x++) {
                if (shape[y][x]) {
                    nextCtx.fillStyle = gameState.nextPiece.color;
                    nextCtx.fillRect(
                        offsetX + x * blockSize,
                        offsetY + y * blockSize,
                        blockSize,
                        blockSize
                    );

                    // ボーダー
                    nextCtx.strokeStyle = '#000';
                    nextCtx.lineWidth = 1;
                    nextCtx.strokeRect(
                        offsetX + x * blockSize,
                        offsetY + y * blockSize,
                        blockSize,
                        blockSize
                    );
                }
            }
        }
    }
}

// ページ読み込み時に初期化
window.addEventListener('load', init);
