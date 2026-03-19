# KKR Facial Affect Reader - macOS環境構築ガイド

このドキュメントは、macOS環境でKKR Facial Affect Readerをセットアップして実行するためのガイドです。

## 目次

1. [システム要件](#システム要件)
2. [環境構築手順](#環境構築手順)
3. [起動方法](#起動方法)
4. [使用方法](#使用方法)
5. [トラブルシューティング](#トラブルシューティング)

---

## システム要件

### 必須環境

- **OS**: macOS 11.0 (Big Sur) 以降
- **Python**: 3.10以降
- **RAM**: 8GB以上推奨
- **ウェブカメラ**: リアルタイム解析を行う場合

### 必要なファイル

以下のファイルが既に配置されていることを確認してください：

```
KKR-MultipleOption/
├── OpenFace-3.0/              # OpenFace 3.0リポジトリ
│   └── weights/               # モデルウェイト
│       ├── mobilenet0.25_Final.pth
│       └── stage2_epoch_7_loss_1.1606_acc_0.5589.pth
├── OpenFace3_Sample_Models/   # Valence/Arousalモデル
│   ├── valence.onnx
│   └── arousal.onnx
└── KKR/                       # アプリケーションコード
    ├── KKR.py
    ├── OpenFace3Wrapper.py
    ├── OpenFace3Adapter.py
    └── requirements.txt
```

---

## 環境構築手順

### ステップ1: Python環境の確認

```bash
# Pythonバージョンを確認（3.10以降が必要）
python3 --version
```

**出力例**: `Python 3.10.7`

### ステップ2: プロジェクトディレクトリに移動

```bash
cd /Users/xxxxx/KKR-MultipleOption
```

### ステップ3: 仮想環境の作成

```bash
# 仮想環境を作成
python3 -m venv venv_kkr

# 仮想環境をアクティベート
source venv_kkr/bin/activate

# Pythonバージョンを再確認
python --version
```

### ステップ4: 依存パッケージのインストール

#### 4-1. pipのアップグレード

```bash
pip install --upgrade pip
```

#### 4-2. PyTorchのインストール

**Apple Silicon (M1/M2/M3/M4) Mac の場合:**

```bash
pip install torch torchvision torchaudio
```

**Intel Mac の場合:**

```bash
pip install torch torchvision torchaudio
```

> **注**: macOS版PyTorchは自動的にCPU版がインストールされます。

#### 4-3. その他の依存パッケージのインストール

```bash
cd KKR
pip install -r requirements.txt
cd ..
```

**インストールされる主要パッケージ:**
- `numpy` - 数値計算
- `opencv-python` - 画像処理
- `pillow` - 画像操作
- `matplotlib` - グラフ描画
- `onnxruntime` - ONNXモデル推論
- `tensorflow` - Kerasモデルサポート
- `keras` - レガシーモデル対応
- `efficientnet_pytorch` - OpenFace 3.0依存
- `timm` - モデルローダー

### ステップ5: インストールの確認

```bash
# 仮想環境をアクティベート（まだの場合）
source venv_kkr/bin/activate

# PyTorchが正しくインストールされているか確認
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# OpenCVの確認
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# ONNXRuntimeの確認
python -c "import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}')"
```

---

## 起動方法

```bash
# 仮想環境をアクティベート
source venv_kkr/bin/activate

# KKRディレクトリに移動
cd KKR

# アプリケーションを起動（OpenFace 3.0バックエンドを指定）
python KKR.py --backend openface3
```

### コマンドラインオプション

```bash
# ヘルプを表示
python KKR.py --help

# オプション一覧
--backend openface3     # OpenFace 3.0バックエンドを使用（macOS推奨）
--fpslim 30             # FPS制限（デフォルト: 30）
--maxf 10000            # 保持する最大フレーム数（デフォルト: 10000）
```

---

## 使用方法

### 初回起動時の設定

1. **KKRアプリケーションが起動します**
   - GUIウィンドウが表示されます
   - 右側に制御パネルがあります

2. **カメラ権限の許可**
   - 初回起動時、macOSがカメラへのアクセス許可を求めます
   - 「OK」をクリックして許可してください
   - **許可しない場合**: システム環境設定 > セキュリティとプライバシー > カメラ から手動で許可

3. **モデルの読み込み**
   
   **「Load Model」ボタンをクリック**
   
   a. **Valenceモデルを選択:**
   ```
   OpenFace3_Sample_Models/valence.onnx
   ```
   
   b. **Arousalモデルを選択:**
   ```
   OpenFace3_Sample_Models/arousal.onnx
   ```

### リアルタイム解析（ウェブカメラ）

1. モデルを読み込んだ後、**「Start」ボタン**をクリック
2. カメラ映像が表示され、顔検出が開始されます
3. 数秒後、Valence/Arousalの推定値がグラフに表示されます
4. 停止するには**「Stop」ボタン**をクリック

### ビデオファイル解析

1. **「Load Video」ボタン**をクリック
2. 解析したいビデオファイル（MP4など）を選択
3. 自動的に解析が開始されます
4. ビデオ終了後、または**「Stop」ボタン**で停止

### 結果の表示

- **左上**: カメラ/ビデオ映像（顔検出ボックス付き）
- **右上**: Valence-Arousal 2次元プレーン
- **下部**: 時系列グラフ（青=Valence、赤=Arousal）
- **「Show Graph」ボタン**: 詳細なグラフを別ウィンドウで表示

### 出力の意味

- **Valence（感情価）**: 1〜5
  - 1 = 非常にネガティブ
  - 3 = ニュートラル
  - 5 = 非常にポジティブ

- **Arousal（覚醒度）**: 1〜5
  - 1 = 非常に落ち着いている
  - 3 = 中程度
  - 5 = 非常に興奮している

---

## トラブルシューティング

### 問題1: カメラが検出されない

**症状**: 「Initialization failed」エラー

**解決方法**:

1. **カメラ権限を確認**
   ```
   システム環境設定 > セキュリティとプライバシー > カメラ
   ```
   Pythonまたはターミナルにカメラアクセスが許可されているか確認

2. **他のアプリケーションがカメラを使用していないか確認**
   - Zoom、Skype、FaceTimeなどを終了

### 問題2: モジュールが見つからない

**症状**: `ModuleNotFoundError: No module named 'torch'`

**解決方法**:

```bash
# 仮想環境が正しくアクティベートされているか確認
which python
# 出力: /Users/xxxxx/KKR-MultipleOption/venv_kkr/bin/python

# 仮想環境をアクティベート
source venv_kkr/bin/activate

# 必要なパッケージを再インストール
pip install torch torchvision
cd KKR && pip install -r requirements.txt
```

### 問題3: OpenFace 3.0モデルが見つからない

**症状**: `FileNotFoundError: Cannot locate OpenFace-3.0 folder`

**解決方法**:

1. **OpenFace-3.0フォルダの存在を確認**
   ```bash
   ls -la OpenFace-3.0/
   ls -la OpenFace-3.0/weights/
   ```

2. **環境変数を設定**
   ```bash
   export OPENFACE3_PATH=/Users/xxxxx/KKR-MultipleOption/OpenFace-3.0
   ```

3. **weights フォルダに必要なファイルがあるか確認**
   - `mobilenet0.25_Final.pth`
   - `stage2_epoch_7_loss_1.1606_acc_0.5589.pth`

### 問題4: ONNXモデルのエラー

**症状**: `Can't load the model file`

**解決方法**:

1. **モデルファイルの存在を確認**
   ```bash
   ls -la OpenFace3_Sample_Models/
   # valence.onnx と arousal.onnx が存在するか確認
   ```

2. **正しいパスを選択**
   - プロジェクトルートから `OpenFace3_Sample_Models/` フォルダを選択
   - ファイル選択ダイアログで正確なファイル名を確認

### 問題5: 顔が検出されない

**症状**: カメラは動作するが、顔が検出されない

**解決方法**:

1. **照明を確認**
   - 明るい場所で実行
   - 顔に直接光が当たるように

2. **カメラの向きと距離**
   - カメラから50cm〜1m程度の距離
   - 顔が画面の中央に来るように

3. **検出感度の確認**
   - コンソール出力で検出状況を確認
   - `[OpenFace3Wrapper] Face detection error` が表示される場合は検出失敗

### 問題6: パフォーマンスが遅い

**症状**: FPSが低い、レスポンスが遅い

**解決方法**:

1. **FPS制限を調整**
   ```bash
   python KKR.py --backend openface3 --fpslim 15
   ```

2. **他のアプリケーションを終了**
   - メモリを多く使用するアプリを終了

3. **解像度を下げる**
   - OpenFace3Wrapper.pyの `output_width` と `output_height` を調整

### 問題7: TkinterのGUI表示エラー

**症状**: `_tkinter.TclError: couldn't connect to display`

**解決方法**:

```bash
# X11が必要な場合はXQuartzをインストール
brew install --cask xquartz

# システムを再起動
```

### 問題8: Apple Silicon (M1/M2/M3) での互換性問題

**症状**: `dyld: Symbol not found` エラー

**解決方法**:

1. **Rosetta 2のインストール（必要な場合）**
   ```bash
   softwareupdate --install-rosetta
   ```

2. **Arm64ネイティブ版Pythonを使用**
   ```bash
   # Python バージョンを確認
   python3 --version
   file $(which python3)
   # arm64 と表示されることを確認
   ```

---

## デバッグ情報の取得

問題が解決しない場合、以下の情報を収集してください：

```bash
# システム情報
sw_vers

# Python環境
python --version
pip list

# OpenFace 3.0の確認
python -c "import sys; sys.path.insert(0, 'OpenFace-3.0'); from model.MLT import MLT; print('OpenFace 3.0 OK')"

# モデルファイルの確認
ls -lh OpenFace-3.0/weights/
ls -lh OpenFace3_Sample_Models/

# KKRの起動（詳細ログ付き）
cd KKR
python KKR.py --backend openface3 2>&1 | tee kkr_debug.log
```

---

## パフォーマンス最適化

### 推奨設定（標準PC）

```bash
python KKR.py --backend openface3 --fpslim 30 --maxf 10000
```

### 低スペックPC向け

```bash
python KKR.py --backend openface3 --fpslim 15 --maxf 5000
```

### 高性能PC向け

```bash
python KKR.py --backend openface3 --fpslim 0 --maxf 20000
```

---

## アンインストール

KKRをシステムから削除する場合：

```bash
# プロジェクトディレクトリに移動
cd /Users/xxxxx/KKR-MultipleOption

# 仮想環境を削除
rm -rf venv_kkr

# （オプション）プロジェクト全体を削除
cd ..
rm -rf KKR-MultipleOption
```

---

## 追加情報

### プロジェクト構成

```
KKR-MultipleOption/
├── venv_kkr/                  # Python仮想環境（自動生成）
├── start_kkr_macos.sh         # macOS起動スクリプト
├── SETUP_MACOS.md             # このドキュメント
├── README.md                  # プロジェクト概要
├── KKR/
│   ├── KKR.py                 # メインアプリケーション
│   ├── OpenFace3Wrapper.py    # OpenFace 3.0ラッパー
│   ├── OpenFace3Adapter.py    # DLL互換アダプター
│   └── requirements.txt       # Python依存パッケージ
├── OpenFace-3.0/              # OpenFace 3.0リポジトリ
│   ├── weights/               # モデルウェイト
│   ├── Pytorch_Retinaface/    # 顔検出モジュール
│   └── model/                 # MLTモデル定義
└── OpenFace3_Sample_Models/   # Valence/Arousalモデル
    ├── valence.onnx
    └── arousal.onnx
```

### 技術スタック

- **顔検出**: RetinaFace (PyTorch)
- **AU推定**: OpenFace 3.0 MLT (Multi-Task Learning)
- **感情推定**: ONNX Runtime (Valence/Arousal)
- **GUI**: Tkinter
- **画像処理**: OpenCV, PIL
- **深層学習**: PyTorch, TensorFlow/Keras

### 参考リンク

- [OpenFace 3.0 GitHub](https://github.com/TadasBaltrusaitis/OpenFace)
- [PyTorch公式サイト](https://pytorch.org/)
- [ONNX Runtime](https://onnxruntime.ai/)
