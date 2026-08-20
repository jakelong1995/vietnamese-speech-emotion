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
Bạn sẽ thấy spinner "Đang tải model MERaLiON-SER-v1...". Sau khi load
xong, spinner biến mất và caption hiển thị `model_id`, `device`, số
class.

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

### Bước 4. Bấm "Analyze"

Nút primary màu xanh dưới mỗi tab. Đợi 1–3 giây (có spinner "Đang
phân tích...").

### Bước 5. Đọc kết quả

Kết quả hiện ngay dưới nút Analyze, trong cùng tab:

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
streamlit run streamlit_app.py
```

Output kỳ vọng:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Mở http://localhost:8501 trong trình duyệt. Từ bước 3 trở đi giống
phần 1 (Tải lên / Thu âm → Phân tích → đọc kết quả).

### 2.3. Tăng tốc phần cứng

App tự chọn thiết bị, **không cần cấu hình gì**. Thứ tự ưu tiên trong
`src/device.py`: `mps` (Apple Silicon) → `cuda` (NVIDIA) → `cpu`.

Caption dưới tiêu đề và ô màu xanh trong sidebar sẽ hiển thị thiết bị
đang dùng, ví dụ `Apple GPU (Metal) (mps)`.

Ép thiết bị khác bằng biến môi trường `SER_DEVICE`:

```bash
SER_DEVICE=cpu streamlit run streamlit_app.py    # auto | cpu | mps | cuda
```

**Tốc độ đo thật** (cửa sổ 3 giây, MERaLiON-SER-v1):

| Thiết bị | Precision | Latency |
|---|---|---|
| Apple Silicon (`mps`) | float32 | **430 ms** |
| CPU | float32 | 980 ms |

Đo trên M5 / 24 GB. Lần đoán **đầu tiên** sau khi load chậm hơn nhiều
(~3 giây) vì Metal phải biên dịch kernel — từ lần thứ hai mới về con số
trên.

> **Vì sao Apple Silicon dùng float32 chứ không phải float16?**
> Metal vẫn phải rơi về CPU với một số phép fp16, nên half precision ở
> đây thường *chậm hơn* và đôi khi sai số. Máy Apple dùng unified memory
> nên cũng không thiếu bộ nhớ để phải tiết kiệm. Riêng CUDA thì fp16 là
> lãi thật nhờ tensor core.

#### Nếu có GPU NVIDIA

`requirements.txt` cài torch bản CPU. Muốn dùng CUDA thì cài wheel riêng:

```bash
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
```

Kiểm tra:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 2.4. Bật bóc băng (PhoWhisper)

Gạt công tắc **"📝 Bóc băng (PhoWhisper)"** trong sidebar. Lần đầu bật
sẽ tải model (~1 GB với bản `small`).

Chọn cỡ model ngay trong sidebar, hoặc đặt biến `ASR_MODEL_NAME`. Biến
này nhận **cả đường dẫn thư mục local** — đây là cách chắc ăn khi CDN
Xet của HuggingFace bị đứt giữa chừng:

```bash
ASR_MODEL_NAME=~/.cache/vser-models/PhoWhisper-small \
    streamlit run streamlit_app.py
```

Tải tay khi Xet lỗi (ép HTTP/1.1, có resume):

```bash
DEST=~/.cache/vser-models/PhoWhisper-small && mkdir -p $DEST
for f in config.json generation_config.json preprocessor_config.json \
         tokenizer.json tokenizer_config.json vocab.json merges.txt \
         normalizer.json added_tokens.json special_tokens_map.json \
         pytorch_model.bin; do
  curl -sL --http1.1 -C - --retry 8 --retry-all-errors -o "$DEST/$f" \
    "https://huggingface.co/vinai/PhoWhisper-small/resolve/main/$f"
done
```

### 2.5. Yêu cầu hệ thống tối thiểu

| Tài nguyên | Tối thiểu | Đề xuất |
|---|---|---|
| Python | 3.10 | 3.11 |
| RAM | 4 GB | 8 GB |
| Disk | 4 GB | 8 GB |
| GPU | không cần | Apple Silicon, hoặc NVIDIA 6 GB VRAM |
| Internet | cần (lần đầu load model) | ổn định |

### 2.6. Troubleshooting

**Lỗi `ModelLoadFailedError: needs >= 1.5 GiB free RAM`**:
- Đóng Chrome tabs, Spotify, Docker, etc.
- Hoặc thêm swap partition
- Hoặc dùng bản deploy trên HF Space (16 GB RAM free)

**Lỗi `OSError: [WinError 1314]` trên Windows**:
- Cài Microsoft C++ Build Tools
- Hoặc skip librosa → chỉ dùng soundfile (đơn giản hơn)

**Lỗi `OSError: cannot connect to localhost:8501`**:
- Cổng 8501 đã bị app khác chiếm
- Chạy với cổng khác: `streamlit run streamlit_app.py --server.port 8502`

**Lỗi model load quá chậm (~5 phút)**:
- Internet chậm, model 1.2 GB
- Dùng VPN hoặc download model thủ công qua HF CLI

---

## 3. Dùng như Python library

Ngoài UI, bạn có thể gọi model từ Python script.

### 3.1. Cú pháp cơ bản

```python
from src import inference

# 1. Warmup (lazy load, ~6s trên Apple Silicon / ~30s CPU)
info = inference.warmup()
print("Loaded:", info["model_id"], "on", info["device"])

# 2. Predict thẳng từ đường dẫn file (tự resample về 16 kHz mono)
result = inference.predict("my_recording.wav")
print("Cảm xúc:", result["label"], f"{result['confidence']:.0%}")
print("Toàn bộ phân phối:", result["class_scores"])
```

Muốn đưa waveform có sẵn thay vì đường dẫn:

```python
from src.audio import load_audio_mono_16k

waveform, sr = load_audio_mono_16k("my_recording.wav")
result = inference.predict_waveform(waveform, sr)
```

### 3.2. Cảm xúc theo thời gian

```python
from src import timeline

tl = timeline.analyze_timeline("my_recording.wav",
                               window_sec=3.0, hop_sec=1.5)
print(tl["summary"]["label"],
      f"nhất quán {tl['summary']['dominant_share']:.0%}")
for seg in tl["segments"]:
    print(f"{seg['start']:5.1f}-{seg['end']:5.1f}s  {seg['label']}")
```

`summary` lấy **trung bình vector xác suất** rồi mới argmax, chứ không
đếm phiếu nhãn từng cửa sổ — nhờ vậy một đoạn buồn nhè nhẹ suốt bài vẫn
đọc ra là buồn dù không cửa sổ nào để `sad` lên đầu.

### 3.3. Bóc băng

```python
from src import asr, timeline

r = asr.transcribe("my_recording.wav")
print(r["text"])
for seg in r["segments"]:          # cụm từ, gom từ mốc mức từ
    print(f"{seg['start']:5.1f}s  {seg['text']}")

# Ghép lời thoại vào từng cửa sổ cảm xúc
tl = timeline.analyze_timeline("my_recording.wav")
for e in timeline.align_transcript(tl["segments"], r["segments"]):
    print(f"{e['start']:4.1f}s  {e['label']:8} | {e['text']}")
```

### 3.4. Output schema của `predict()`

```python
{
    "label": str,                 # top-1 sau khi chuẩn hoá về ViSEC
                                  # ("other" nếu model chọn fearful/
                                  #  disgusted/surprised)
    "raw_label": str,             # nhãn thô của model (1 trong 7)
    "confidence": float,          # xác suất của nhãn top-1
    "class_scores": dict[str, float],   # softmax thật, 7 lớp, tổng = 1
    "latency_ms": int,
    "model_id": str,
    "device": str,                # "mps" | "cuda" | "cpu"
    "labels": list[str],
}
```

Dict còn vài khoá metadata khác (`provider`, `language`,
`vietnamese_verified`, `bench_score`…) — xem `_format_for_ui()` trong
`src/inference.py`.

### 3.5. Ví dụ: batch inference

```python
import csv
from pathlib import Path

from src import inference

inference.warmup()          # load 1 lần, dùng cho cả vòng lặp

with open("ket_qua.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["file", "cam_xuc", "do_tin_cay", "ms"])
    for path in sorted(Path("test_audio").glob("*.wav")):
        r = inference.predict(str(path))
        w.writerow([path.name, r["label"],
                    f"{r['confidence']:.4f}", r["latency_ms"]])
        print(f"{path.name}: {r['label']} ({r['confidence']:.0%})")
```

Nếu cần chạm thẳng vào adapter (bỏ qua lớp `inference`), lưu ý
`adapter.predict()` trả về **dataclass `RawPrediction`**, không phải
dict — truy cập bằng thuộc tính:

```python
adapter = inference.get_adapter()
wav, sr = inference.load_audio_mono_16k("a.wav")
pred = adapter.predict(wav, sr)
print(pred.label, pred.confidence, pred.class_scores)   # KHÔNG phải pred["label"]
```

Lưu ý: trong script chạy standalone `from src.inference import ...`,
cần chạy từ thư mục project root và `.venv` đã active.

---

## 4. Chạy benchmark

Để tái lập số liệu 40.25% / 40.70% trong README:

### 4.1. Yêu cầu

- Đã cài `requirements.txt`
- Ít nhất 4 GiB RAM free
- Internet để tải ViSEC dataset (~367 MB, lần đầu)

> ⚠️ **`bench/run_meralion.py` chưa hỗ trợ MPS** — nó chỉ nhận
> `--device cpu` hoặc `--device cuda`. Trên máy Apple Silicon,
> `--device auto` sẽ ra `cpu`, tức benchmark **không dùng GPU Metal**
> dù app thì có. Con số 40.25% trong README được đo trên máy CUDA khác
> (xem `device`/`dtype` trong `bench/results/scores.json`).

### 4.2. Chạy

```bash
# Đảm bảo đang ở project root và đã activate .venv

# Trên Apple Silicon / máy không có NVIDIA (chạy CPU, ~8-10 phút):
.venv/bin/python bench/run_meralion.py --per-class 100 --device cpu

# Trên máy có GPU NVIDIA (~5 phút):
.venv/bin/python bench/run_meralion.py \
    --per-class 100 --device cuda --dtype float16
```

Output (ví dụ trên máy CUDA):

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
- File size: ≤ 200 MB (giới hạn mặc định `st.file_uploader` của Streamlit)

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

1. Upload 1 file audio giọng **trung tính** (đọc tin tức chẳng hạn).
2. Kỳ vọng: `neutral` với confidence cao (~80-90%). Đây là lớp mạnh
   nhất của model.
3. Sau đó thử giọng vui / buồn rõ rệt. **Đừng ngạc nhiên nếu vẫn ra
   `neutral`** — model chỉ đạt 40.25% trên ViSEC và `happy`/`sad` là
   hai lớp yếu nhất (F1 ≈ 0.31-0.33), rất hay bị nuốt thành `neutral`.
   Đó là giới hạn đã biết của model, không phải lỗi cài đặt.
4. Nếu ngay cả giọng trung tính cũng ra kết quả lạ, mới nên nghi:
   - Audio quá nhỏ / quá to (clipping)
   - Nhiều tiếng ồn nền
   - File không thật sự chứa tiếng nói

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

Hiện tại chỉ có UI Streamlit — không có REST API tự động (khác với
Gradio). Muốn gọi model từ script/service khác, import trực tiếp
`src.inference.predict()` (xem mục 3 ở trên) hoặc tự viết một wrapper
FastAPI mỏng gọi cùng hàm đó. Webhook chưa có — nếu cần, sửa
`inference.predict()` thêm side-effect (HTTP POST đến URL).

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
