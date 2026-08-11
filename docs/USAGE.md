# Hướng dẫn sử dụng — Vietnamese Speech Emotion Recognition

> **Đối tượng**: user cuối (không nhất thiết phải là developer).
> **Đọc sau khi đã đọc** [README.md](../README.md) (overview).
> **Developer**: xem [docs/ARCHITECTURE.md](ARCHITECTURE.md).

---

## 1. Nhanh nhất — Dùng bản deploy trên Hugging Face

Không cần cài gì, chỉ cần trình duyệt.

### Bước 1. Truy cập Space

Mở trình duyệt (Chrome / Edge / Firefox / Safari), vào URL:

```
https://huggingface.co/spaces/<username>/vietnamese-speech-emotion
```

> Thay `<username>` bằng username của người deploy (trong README có link
> nếu bạn đang dùng bản public).

### Bước 2. Đợi model load (~30 giây)

Lần đầu truy cập, Space tải model `MERaLiON-SER-v1` (~1.2 GB) từ HF Hub.
Bạn sẽ thấy spinner hoặc banner "Loading model...". Sau khi load xong,
status banner chuyển sang xanh lá.

### Bước 3. Upload audio

Có 3 cách:

**📁 Upload file**:
- Tab "Upload"
- Kéo thả file wav / mp3 / ogg / m4a / flac / webm vào ô upload
- Hoặc click vào ô để chọn file từ máy

**🎙️ Microphone**:
- Tab "Microphone"
- Click nút ghi âm (hình microphone)
- Nói từ 3–30 giây
- Click nút dừng (hình vuông)

**🎵 Sample có sẵn**:
- Tab "Samples"
- Click vào 1 file mẫu trong danh sách

### Bước 4. Bấm "Analyze"

Nút xanh lớn ở giữa. Đợi 1–3 giây.

### Bước 5. Đọc kết quả

Kết quả hiện ở panel bên phải:

```
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  Emotions:  happy                                          │
│  (hoặc: neutral, sad, angry, fearful, disgusted, surprised) │
│                                                            │
│  Confidence: 68%                                           │
│                                                            │
│  Raw label:    happy                                       │
│  Latency:      72 ms                                      │
│  Device:        CPU                                        │
│                                                            │
├────────────────────────────────────────────────────────────┤
│  Bar plot (7-class probability distribution)               │
│  neutral   ████████        0.18                            │
│  happy     ██████████████  0.68                            │
│  sad       ███             0.05                            │
│  angry     █              0.03                             │
│  fearful   ▏              0.02                             │
│  disgusted ▏              0.02                             │
│  surprised ▏              0.02                             │
└────────────────────────────────────────────────────────────┘
```

**Đọc nhanh**:
- **Tên emotion to ở giữa, màu theo**: happy = cam, sad = xanh dương,
  angry = đỏ, neutral = xám, fearful/disgusted/surprised = tím/vàng.
- **Confidence = top emotion score**. Cao (>70%) → model tự tin.
  Thấp (<40%) → model phân vân, nên xem thêm bar plot bên dưới.
- **Bar plot = full 7-vector** model output. Mỗi thanh là xác suất
  model cho rằng audio thuộc class đó.

---

## 2. Tự cài local

### 2.1. Cài Python ≥ 3.10

Kiểm tra:

```bash
python3 --version
```

Nếu chưa có hoặc bản cũ, cài một trong:
- Ubuntu/Debian: `sudo apt install python3.11 python3.11-venv`
- macOS (Homebrew): `brew install python@3.11`
- Windows: tải installer từ https://www.python.org/downloads/

### 2.2. Clone + cài đặt

```bash
# 1. Clone
git clone https://huggingface.co/spaces/<username>/vietnamese-speech-emotion
cd vietnamese-speech-emotion

# 2. Tạo virtualenv
python3 -m venv .venv
source .venv/bin/activate              # Linux/macOS
# hoặc: .venv\Scripts\activate        # Windows

# 3. Cài dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Chạy app
python app.py
```

Output kỳ vọng:

```
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```

Mở http://127.0.0.1:7860 trong trình duyệt. Từ bước 3 trở đi giống
phần 1 (Upload / Mic / Sample → Analyze → đọc kết quả).

### 2.3. Kích hoạt GPU (nếu có card NVIDIA)

Mặc định `requirements.txt` cài torch CPU. Nếu bạn có GPU NVIDIA và
muốn inference nhanh hơn 40 lần:

```bash
# Trong cùng virtualenv
pip install --upgrade torch \
    --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```bash
python -c "import torch; \
    print('CUDA:', torch.cuda.is_available()); \
    print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Kỳ vọng:
```
CUDA: True
Device: NVIDIA GeForce RTX 4050 Laptop GPU
```

Sau đó restart app, status banner sẽ hiển thị `device: cuda` thay vì
`device: cpu`.

### 2.4. Yêu cầu hệ thống tối thiểu

| Tài nguyên | Tối thiểu | Đề xuất |
|---|---|---|
| Python | 3.10 | 3.11 |
| RAM | 4 GB | 8 GB |
| Disk | 4 GB | 8 GB |
| GPU | không cần | NVIDIA 6 GB VRAM |
| Internet | cần (lần đầu load model) | ổn định |

### 2.5. Troubleshooting

**Lỗi `ModelLoadFailedError: needs >= 1.5 GiB free RAM`**:
- Đóng Chrome tabs, Spotify, Docker, etc.
- Hoặc thêm swap partition
- Hoặc dùng bản deploy trên HF Space (16 GB RAM free)

**Lỗi `OSError: [WinError 1314]` trên Windows**:
- Cài Microsoft C++ Build Tools
- Hoặc skip librosa → chỉ dùng soundfile (đơn giản hơn)

**Lỗi `OSError: cannot connect to localhost:7860`**:
- Cổng 7860 đã bị app khác chiếm
- Sửa `app.py`: đổi `server_port=7860` → `server_port=7861`

**Lỗi model load quá chậm (~5 phút)**:
- Internet chậm, model 1.2 GB
- Dùng VPN hoặc download model thủ công qua HF CLI

---

## 3. Dùng như Python library

Ngoài UI, bạn có thể gọi model từ Python script.

### 3.1. Cú pháp cơ bản

```python
from src.inference import get_adapter, predict, warmup
from src.audio import load_audio

# 1. Warmup (lazy load model, ~30s CPU / ~9s GPU)
info = warmup()
print("Loaded:", info["model_id"], "on", info.get("device", "cpu"))

# 2. Load audio
waveform, sample_rate = load_audio("my_recording.wav")
print("Duration:", len(waveform) / sample_rate, "seconds")

# 3. Predict
result = predict(("my_recording.wav", None))
print("Top emotion:", result["label"])
print("Confidence:", result["scores"][result["label"]])
print("Full scores:", result["scores"])
```

### 3.2. Output schema

```python
{
    "label": str,             # top-1 emotion, e.g. "happy"
    "scores": dict[str, float],  # {label: probability} cho 7 classes
    "raw": str,               # trùng với label
    "latency_ms": int,        # thời gian inference
    "device": str,            # "cuda" hoặc "cpu"
    "dtype": str,             # "float16" hoặc "float32"
}
```

### 3.3. Ví dụ: batch inference

```python
from src.audio import load_audio
from src.inference import get_adapter
import os

adapter = get_adapter()  # load 1 lần, dùng nhiều lần

for filename in os.listdir("test_audio/"):
    if not filename.endswith(".wav"):
        continue
    wav, sr = load_audio(f"test_audio/{filename}")
    pred = adapter.predict(wav, sr)
    print(f"{filename}: {pred['label']} ({pred['scores'][pred['label']]:.2f})")
```

Lưu ý: trong script chạy standalone `from src.inference import ...`,
cần chạy từ thư mục project root và `.venv` đã active.

---

## 4. Chạy benchmark

Để tái lập số liệu 40.25% / 40.70% trong README:

### 4.1. Yêu cầu

- Đã cài `requirements.txt`
- Có GPU (khuyến nghị) hoặc ít nhất 4 GiB RAM free
- Internet để tải ViSEC dataset (~367 MB, lần đầu)

### 4.2. Chạy

```bash
# Đảm bảo đang ở project root và đã activate .venv
.venv/bin/python bench/run_meralion.py \
    --per-class 100 \
    --device cuda \
    --dtype float16
```

Output:

```
Loading MERaLiON-SER-v1 on cuda (float16)…
Device: NVIDIA GeForce RTX 4050, dtype: float16
Loading hustep-lab/ViSEC (5,280 clips)…
Subset: 100 clips/class × 4 = 400 clips

[####      ] 50/400  happy=85% 17.2 clips/s ETA 0:21

== Benchmark results ==
Accuracy:  40.25% (161/400)
Macro F1:  40.70%
Per-class F1:
  - happy:   0.3097
  - neutral: 0.4952
  - sad:     0.3259
  - angry:   0.4970

Wrote bench/results/meralion.json
Wrote bench/results/meralion_predictions.csv (400 lines)
```

### 4.3. Các flag hữu ích

| Flag | Ý nghĩa | Mặc định |
|---|---|---|
| `--per-class N` | Số clip per class (subset balance) | 100 |
| `--device {cuda,cpu}` | Thiết bị | auto-detect |
| `--dtype {float16,float32}` | Precision | auto (cuda→fp16, cpu→fp32) |
| `--seed N` | Random seed cho subset balance | 0 |
| `--output PATH` | Output JSON path | `bench/results/meralion.json` |
| `--limit N` | Giới hạn tổng clip (debug) | 400 |

### 4.4. Tự đánh giá lại từ CSV

```bash
python -c "
import json, pandas as pd
df = pd.read_csv('bench/results/meralion_predictions.csv')
y_true = df['true']
y_pred = df['pred']
from sklearn.metrics import accuracy_score, f1_score
print('Accuracy:', accuracy_score(y_true, y_pred))
print('Macro F1:', f1_score(y_true, y_pred, average='macro'))
"
```

---

## 5. Format audio được hỗ trợ

Library `librosa` + `soundfile` trong `src/audio.py` hỗ trợ:

| Format | MIME | Decode via |
|---|---|---|
| WAV | `audio/wav`, `audio/x-wav` | soundfile |
| MP3 | `audio/mpeg` | librosa + audioread |
| OGG | `audio/ogg` | librosa + audioread |
| M4A / AAC | `audio/mp4`, `audio/aac` | librosa + audioread |
| FLAC | `audio/flac` | soundfile |
| WebM | `audio/webm` | librosa + audioread |

**Yêu cầu chung**:
- Sample rate: bất kỳ (sẽ được resample về 16 kHz mono)
- Channels: stereo sẽ được downmix thành mono
- Duration: 1–60 giây (quá ngắn hoặc quá dài có thể kết quả kém)
- File size: ≤ 50 MB (giới hạn Gradio)

**Tip audio quality**:
- Dùng microphone ngoài sẽ tốt hơn mic laptop.
- Tránh tiếng ồn nền (quạt, traffic) → ảnh hưởng confidence.
- Nói đủ to, không quá nhỏ.
- Tiếng Việt rõ ràng, không nói ngọng / nói giọng địa phương quá nặng.

---

## 6. Đánh giá chất lượng output

### 6.1. Khi nào nên tin model?

| Confidence | Ý nghĩa | Hành động gợi ý |
|---|---|---|
| **> 70%** | Model rất tự tin | Tin tưởng kết quả |
| **40–70%** | Model phân vân | Xem bar plot, xem class thứ 2 |
| **< 40%** | Model bối rối | Thử upload audio khác, hoặc kiểm tra audio quality |

### 6.2. Các pattern thường gặp

Model có một số pattern thường gặp trên ViSEC:

- **Happy ↔ Neutral**: hay nhầm. Hai class này prosody khá giống nhau.
- **Sad ↔ Neutral**: cũng hay nhầm. Sad thường bị model dự đoán neutral.
- **Angry**: thường nhận đúng vì prosody rất đặc trưng.
- **Fearful / Disgusted / Surprised**: ít gặp trong ViSEC, khi model
  predict các class này thường là model đang confused.

### 6.3. Verify nhanh

Nếu muốn kiểm tra model có hoạt động không:

1. Upload 1 file audio nói "Tôi rất vui" với giọng vui rõ ràng.
2. Kỳ vọng: top emotion = `happy` với confidence > 50%.
3. Nếu kết quả khác, có thể:
   - Audio quá nhỏ / quá to
   - Có nhiều tiếng ồn
   - Giọng nói đặc trưng vùng miền mà model chưa thấy nhiều

---

## 7. FAQ

### Q1. Tại sao accuracy chỉ 40%? Model tệ quá?

40.25% trên ViSEC 4-class — vẫn là **best-effort** với model có sẵn
không cần fine-tune. Stop criterion 65% chưa đạt vì:

- Dataset quá nhỏ (400 clips balanced, mỗi clip 5-10s)
- Model MERaLiON chỉ fine-tune emotion với Vietnamese là secondary language
- Fine-tune specifically cho Vietnamese cần làm riêng (scope ngoài đồ án)

Xem chi tiết trong [docs/BENCHMARK.md](BENCHMARK.md) và [README.md → Limitations](../README.md#limitations).

### Q2. Có chạy được trên điện thoại không?

**Hiện tại không.** Model cần ≥ 1.5 GB RAM. Điện thoại Android tầm trung
chỉ có 4 GB RAM chia cho nhiều app.

**Tương lai**: sau khi quantization model (FP8 / INT8), có thể chạy
trên iPhone 15 Pro / Pixel 8 Pro trở lên — nhưng cần 6-12 tháng dev.

### Q3. Có hỗ trợ tiếng Anh / tiếng khác không?

MERaLiON-SER-v1 hỗ trợ 7 raw labels emotion áp dụng cho mọi ngôn ngữ.
Tuy nhiên, model được train chủ yếu với dữ liệu tiếng Anh + một số
tiếng Đông Á khác (Trung, Nhật, Hàn, Việt, Indonesia).

**Hiện tại UI và benchmark chỉ đánh giá tiếng Việt**.

### Q4. Tại sao không dùng Whisper Vietnamese + PhoBERT?

Kết hợp 2 model lớn (Whisper-VN ~1.5 GB + PhoBERT ~0.5 GB) sẽ cần
> 4 GB RAM, latency cũng cao hơn. MERaLiON-SER-v1 đã tích hợp sẵn
audio feature extractor và emotion classifier trong 1 model duy nhất
~1.2 GB.

### Q5. License?

- **Code**: MIT License (file [LICENSE](../LICENSE))
- **Model MERaLiON-SER-v1**: Apache-2.0
- **Dataset ViSEC**: CC-BY-4.0

Bạn có thể dùng thương mại, sửa đổi, phân phối lại — với điều kiện
giữ attribution cho MERaLiON và ViSEC.

### Q6. Có cách nào cải thiện accuracy không?

Có, nhưng cần effort:

1. **Fine-tune MERaLiON** trên full ViSEC (5,280 clips) với learning
   rate thấp, ~2 GPU-hours.
2. **Data augmentation**: pitch shift, time stretch, noise injection
   → tăng từ 5,280 clips giả lập lên 50,000+.
3. **Ensemble**: chạy nhiều model (MERaLiON + Wav2Vec2-VN + custom),
   vote majority.
4. **Pseudo-labeling**: dùng MERaLiON predict trên audio lớn hơn,
   lấy confident predictions làm training data.

### Q7. Có API endpoint không? Có webhook không?

Hiện tại chỉ có UI Gradio. Gradio tự động tạo REST API ở
`/predict` endpoint cho developer tích hợp. Webhook chưa có — nếu cần,
sửa `inference.predict()` thêm side-effect (HTTP POST đến URL).

### Q8. Làm sao đóng góp?

- Tạo Pull Request trên GitHub repo (nếu bạn fork)
- Hoặc gửi issue trên HuggingFace Space
- Hoặc email maintainer (xem README → Authors)

Các đóng góp giá trị:
- Bug report (có steps to reproduce)
- Vietnamese audio samples để cải thiện evaluation
- Code patch để fix issue trong issue tracker
- Documentation improvements

---

## 8. Bảng thuật ngữ

| Thuật ngữ | Giải thích |
|---|---|
| **SER** | Speech Emotion Recognition |
| **Whisper** | ASR model của OpenAI, encoder phù hợp cho audio tasks |
| **LoRA** | Low-Rank Adaptation — trainable params nhỏ trên top frozen model |
| **ECAPA-TDNN** | Speaker embedding network (dùng để kết hợp prosody cues) |
| **ViSEC** | Dataset tiếng Việt có nhãn emotion, duy nhất public |
| **FP16 / FP32** | Half precision / full precision floating point (ảnh hưởng speed/memory) |
| **Softmax** | Hàm chuẩn hóa vector thành probability distribution |
| **Top-1** | Class có xác suất cao nhất |
| **Macro-F1** | Trung bình F1 score trên tất cả class, không phân biệt support |

---

## 9. Liên hệ & hỗ trợ

- **Project README**: [README.md](../README.md)
- **Architecture**: [docs/ARCHITECTURE.md](ARCHITECTURE.md)
- **Benchmark chi tiết**: [docs/BENCHMARK.md](BENCHMARK.md)
- **Presentation guide**: [docs/PRESENTATION.md](PRESENTATION.md)
- **HF Space link**: xem README
- **Issue tracker**: trên GitHub hoặc HF Space → Files → Community tab
