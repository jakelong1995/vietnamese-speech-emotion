# Kiến trúc hệ thống — Vietnamese Speech Emotion Recognition

> **Đối tượng**: developer muốn hiểu hoặc mở rộng codebase.
> **Đọc trước**: xem [README.md](../README.md) (overview ngắn).
> **Người dùng cuối**: xem [docs/USAGE.md](USAGE.md). Người thuyết trình: xem
> [docs/PRESENTATION.md](PRESENTATION.md).

---

## 1. Tổng quan 3 tầng

```
┌────────────────────────────────────────────────────────────────┐
│  Tầng 1 — PRESENTATION                                         │
│  app.py (Gradio)                                               │
│  - Markdown header / footer                                    │
│  - 3 tab input: Upload / Microphone / Samples                  │
│  - Result card + Bar plot                                      │
└─────────────────────────────┬──────────────────────────────────┘
                              │  audio_path (str)
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  Tầng 2 — INFERENCE                                            │
│  src/inference.py                                              │
│  - get_adapter() (singleton)                                   │
│  - warmup() / predict() / list_adapters() /                    │
│    bench_scores()                                              │
│                                                                │
│  src/adapters/                                                 │
│  - base.BaseAdapter (interface)                                │
│  - meralion.MeralionAdapter (concrete)                        │
└─────────────────────────────┬──────────────────────────────────┘
                              │  RawPrediction (label, scores, …)
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  Tầng 3 — MODEL                                                │
│  HuggingFace Transformers                                      │
│  - AutoFeatureExtractor                                        │
│  - AutoModelForAudioClassification                             │
│  Model: MERaLiON/MERaLiON-SER-v1 (~309M params + ~5M LoRA)    │
└────────────────────────────────────────────────────────────────┘
```

Ba tầng giao tiếp qua **2 interface** rất nhỏ:
- UI → Inference: chỉ gọi `inference.predict(audio_path)` và
  `inference.warmup()` (cho status banner).
- Inference → Model: chỉ gọi `adapter.predict(waveform, sample_rate)`
  qua interface `BaseAdapter`.

---

## 2. Module map

```
vietnamese-speech-emotion/
├── app.py                       # Gradio UI (~270 LOC)
├── requirements.txt             # CPU baseline — torch chưa có CUDA
├── setup_rtx4050.sh             # cài torch CUDA 12.1 cho dev GPU
├── README.md                    # entry chính cho user / HF Space
│
├── src/
│   ├── __init__.py              # re-export public API
│   ├── inference.py             # singleton wrapper (~175 LOC)
│   ├── audio.py                 # load + resample + format detection
│   ├── exceptions.py            # custom error hierarchy
│   └── adapters/
│       ├── __init__.py          # REGISTRY = {meralion-ser-v1: …}
│       ├── base.py              # BaseAdapter + ModelInfo + RawPrediction
│       └── meralion.py          # MeralionAdapter (HF loader + patch)
│
├── bench/
│   ├── run_meralion.py          # CLI benchmark
│   ├── visec.py                 # load + balance ViSEC subset
│   ├── metrics.py               # accuracy, F1, confusion matrix
│   └── results/
│       ├── scores.json          # ← SOURCE OF TRUTH (40.25%)
│       └── meralion_predictions.csv   # per-clip log (streaming-safe)
│
├── tests/
│   ├── conftest.py              # fixtures (stub REGISTRY, stub audio)
│   ├── test_inference.py        # 6 tests cho wrapper
│   ├── test_no_fabrication.py   # 7 invariant tests (no fake scores)
│   ├── test_audio.py            # 4 tests audio loading
│   └── test_visec_loader.py     # 3 tests benchmark subset
│
└── docs/
    ├── BENCHMARK.md             # methodology + per-class numbers
    ├── ARCHITECTURE.md          # file này
    ├── USAGE.md                 # user guide (end-to-end)
    └── PRESENTATION.md          # 25-slide script (Vietnamese)
```

---

## 3. Tầng Presentation — `app.py`

### 3.1. Cấu trúc Gradio Blocks

`app.py` xây dựng UI theo `gr.Blocks(...)` (không `Interface`) để có toàn
quyền kiểm soát layout. Cấu trúc:

```
gr.Blocks(theme=gr.themes.Soft(), css=APP_CSS)
├── gr.Markdown(HEADER_MD)              # tiêu đề + subtitle
├── gr.HTML(STATUS_HTML)                # status banner (model info)
├── gr.Tabs()
│   ├── Tab "📁 Upload"   → gr.Audio(type="filepath")
│   ├── Tab "🎙️ Mic"     → gr.Audio(type="filepath", sources="microphone")
│   └── Tab "🎵 Samples" → gr.Examples(samples=…)
├── gr.Button("Analyze", variant="primary")
├── gr.Row()
│   ├── gr.Column()  # emotion label, confidence, raw chip, latency
│   └── gr.Column()  # BarPlot class_scores
└── gr.Markdown(FOOTER_MD)
```

### 3.2. Design tokens (CSS variables)

`APP_CSS` (top của `app.py`) khai báo 6 biến màu + spacing + radius:

```css
:root {
  --blue:  #1d4ed8;   /* primary action */
  --green: #15803d;   /* success */
  --amber: #b45309;   /* warning / neutral */
  --red:   #b91c1c;   /* error / angry */
  --ink:   #162033;   /* main text */
  --muted: #64748b;   /* secondary text */
  --surface: #ffffff; /* card bg */
  --radius: 14px;
  --gap: 24px;
}
```

Các class sử dụng token:
- `.result-shell` — card bo góc, đổ bóng nhẹ, padding `--gap`
- `.emotion-label` — font clamp 36-72px, weight 700, màu theo emotion
- `.confidence-gauge` — vòng tròn `conic-gradient()` theo % confidence
- `.metric-chip` — chip nhỏ hiển thị "Raw label: surprised" hoặc "Latency: 72 ms"

### 3.3. Quy trình xử lý request

```python
def on_analyze_click(audio_path, history):
    info = inference.warmup()                    # lazy-load model lần đầu
    result = inference.predict(audio_path)       # RawPrediction dict
    top_label = result["label"]
    scores = result["class_scores"]              # {neutral: .4, …}
    confidence = scores[top_label]
    color = EMOTION_COLORS[top_label]            # hex code
    return (
        build_emotion_label_html(top_label, color),  # <div> HTML
        build_confidence_gauge_html(confidence, color),
        build_metric_chips_html(raw=…, latency_ms=…, …),
        build_barplot_df(scores),                     # pd.DataFrame
        append_to_history(history, top_label),
    )
```

Quan trọng: **`history` (lịch sử dự đoán) chỉ lưu trong Gradio client**,
không gửi lên server. Server stateless — mỗi request chỉ chứa audio + IDs.

### 3.4. Single-model — không có selector

Sau phase đơn giản hóa, `app.py` đã bỏ:
- `_leaderboard_html()` — bảng so sánh nhiều model
- `_recommendation_html()` — banner "bạn nên dùng model X"
- Model dropdown — không có nhiều model để chọn
- `_switch_model()` — chỉ có 1 model cố định

Kết quả: `app.py` ~270 dòng (giảm ~250 so với multi-model).

---

## 4. Tầng Inference — `src/inference.py`

### 4.1. Singleton pattern

`_active` là biến module-level, chỉ có 1 instance tại một thời điểm:

```python
_active: Optional[BaseAdapter] = None
_active_name: str = os.environ.get("SER_MODEL_NAME", "meralion-ser-v1")

def get_adapter() -> BaseAdapter:
    """Lazy-load active adapter, trả về cùng instance cho mọi caller."""
    global _active
    if _active is None:
        set_active(_active_name)        # tạo + load
    return _active
```

**Ý nghĩa**: nếu user gọi `predict()` 100 lần, model chỉ load 1 lần.
Đây là quan trọng vì MERaLiON mất ~9 giây trên GPU / ~30 giây trên CPU.

### 4.2. Các hàm public

| Hàm | Mục đích | Return |
|---|---|---|
| `warmup()` | Pre-load adapter, trả về `ModelInfo` để UI hiển thị status | `dict` |
| `predict(path)` | Chạy inference trên audio file | `dict` UI-shaped |
| `list_adapters()` | Liệt kê tên model có sẵn | `list[str]` |
| `bench_scores()` | Đọc `bench/results/scores.json` | `dict` |
| `set_active(name)` | Force chuyển model (chỉ dev dùng) | `None` |

### 4.3. Soft fallback khi host yếu

Trước đó `warmup()` raise `ModelLoadFailedError` ngay khi khởi động nếu
RAM < 1.5 GiB, khiến UI không render. Phase sửa lỗi đã đổi sang
"try best → catch → continue without crashing":

```python
def warmup() -> Dict[str, Any]:
    try:
        adapter = get_adapter()
    except ModelLoadFailedError as e:
        logger.warning("Adapter load failed: %s — UI continues.", e)
        return {"status": "unavailable", "reason": str(e)}
    return _as_info_dict(adapter.get_model_info())
```

UI đọc `status` field để quyết định hiển thị banner xám "model chưa sẵn
sàng" thay vì crash.

### 4.4. Source of truth cho status

Khi user mở app:
1. `app.py::warmup()` gọi `inference.warmup()` (background).
2. UI hiển thị spinner nếu chưa có status, banner xanh khi có.
3. Click "Analyze" sẽ trigger `inference.predict()`, lúc đó mới load thật
   (trừ khi warmup đã load rồi).

---

## 5. Tầng Adapters — `src/adapters/`

### 5.1. `BaseAdapter` (interface)

```python
class BaseAdapter(ABC):
    model_id: str          # canonical name trên HF Hub
    sample_rate: int       # input sample rate mà model muốn (16_000)
    labels: List[str]      # raw output labels (7 buckets)

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def predict(self, waveform, sample_rate) -> "RawPrediction": ...

    @abstractmethod
    def get_model_info(self) -> "ModelInfo": ...
```

### 5.2. `RawPrediction` schema

Dict trả về từ `predict()`. UI đọc trực tiếp, không chuyển đổi:

```python
{
    "label": "happy",          # top-1 raw label
    "scores": [                 # full 7-vector
        {"label": "neutral",   "score": 0.12},
        {"label": "happy",     "score": 0.68},
        ...
    ],
    "raw": "happy",             # trùng với label (giữ để tương thích)
    "latency_ms": 72,
    "device": "cuda",
    "dtype": "float16",
}
```

Quan trọng: **`scores` KHÔNG ĐƯỢC phải uniform, không NaN, không âm**.
Đây là invariant được test bởi `tests/test_no_fabrication.py`.

### 5.3. `MeralionAdapter` (concrete)

```python
class MeralionAdapter(BaseAdapter):
    model_id = "MERaLiON/MERaLiON-SER-v1"
    sample_rate = 16_000
    labels = ["neutral", "happy", "sad", "angry",
              "fearful", "disgusted", "surprised"]

    def load(self):
        # 1. Patch transformers meta-tensor bug (workaround upstream)
        # 2. AutoFeatureExtractor.from_pretrained(...)
        # 3. AutoModelForAudioClassification.from_pretrained(
        #        trust_remote_code=True)
        # 4. .to(self._device).to(self._dtype)
        # 5. .eval()

    def predict(self, waveform, sr):
        # 1. Resample nếu sr != 16_000 (librosa)
        # 2. features = fe(wav, sampling_rate=16_000, return_tensors="pt")
        # 3. features = {k: v.to(device) for k,v in features.items()}
        # 4. with torch.inference_mode(): logits = model(**features).logits
        # 5. probs = logits.softmax(dim=-1).squeeze().cpu().tolist()
        # 6. zip(labels, probs) → RawPrediction(...)
```

### 5.4. Trust-remote-code

Model MERaLiON có `auto_map` trong config.json — chứa custom code
(`MERaLiON_SER_Encoder`). Transformers cần `trust_remote_code=True` để
load nó. Đây là trade-off bảo mật chấp nhận được vì:
- Source code của model public trên HF Hub, ai cũng audit được.
- License MIT/Apache-2.0, không có hidden weight.

### 5.5. Patch `meta` tensor

Bug: trên một số phiên bản transformers, `from_pretrained(..., device_map="auto")`
gặp lỗi `"cannot copy from meta tensor"`. Workaround: load về CPU
trước, sau đó `.to(device)`. Đây là known issue của transformers < 4.42,
em đã patch trong `MeralionAdapter._safe_to_device()`.

---

## 6. Tầng Model — `MERaLiON/MERaLiON-SER-v1`

### 6.1. Kiến trúc gốc

```
┌──────────────────────────────────────────────────────┐
│  Whisper-Medium encoder (frozen, 309M params)        │
│  - 24 transformer layers                              │
│  - hidden_dim=1024, vocab_size=51865                  │
│  - Pre-trained: 680k giờ multilingual audio           │
└────────────────────┬─────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────┐
│  ECAPA-TDNN speaker embedder (frozen)                 │
│  - input: mel-spectrogram                             │
│  - output: 192-dim embedding                          │
└────────────────────┬─────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────┐
│  LoRA adapters (trainable, ~5M params)               │
│  - low-rank matrices trên Q,V của mỗi attn layer    │
│  - rank=8, alpha=16                                   │
└────────────────────┬─────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────┐
│  Classification head (trainable, ~5K params)         │
│  - Linear(1024+192, 7)                                │
│  - outputs: 7 logits → softmax                       │
└──────────────────────────────────────────────────────┘
```

### 6.2. 7 raw labels

MERaLiON-SER-v1 trả về đúng **7 labels cố định**, không có merging:

```
neutral · happy · sad · angry · fearful · disgusted · surprised
```

- **4 labels overlap với ViSEC**: `neutral / happy / sad / angry`
- **3 labels KHÔNG overlap**: `fearful / disgusted / surprised`
  - UI vẫn show trong BarPlot để user thấy full output của model
  - Benchmark ViSEC **không tính** 3 labels này (zero micro-averaged)
- **Quy tắc cứng**: KHÔNG đưa 3 labels này vào `neutral` hoặc đại
  lý. Raw output phải giữ nguyên — đây là thuộc tính quan trọng
  cho reproducibility và audit.

### 6.3. Vietnamese coverage

Model card ghi Vietnamese là "limited/secondary language". Hiện tại:
- 40.25% accuracy / 40.70% macro-F1 trên 400-clip balanced ViSEC
- Angry recall cao (~51%), happy/sad yếu (~30-36%)
- Không cross-validated chính thức với human labels

Trade-off: model có sẵn trên HF Hub, không cần train, không bị phụ
thuộc data private. Nhưng accuracy chưa đạt SOTA 65%+ của các model
được fine-tune specifically cho tiếng Việt.

---

## 7. Pipeline inference chi tiết

### 7.1. Sơ đồ tuần tự

```
User             Gradio UI              inference.py            MeralionAdapter       HuggingFace Model
 │                    │                       │                        │                       │
 │ upload audio.wav   │                       │                        │                       │
 ├───────────────────►│                       │                        │                       │
 │                    │ predict(path)         │                        │                       │
 │                    ├──────────────────────►│                        │                       │
 │                    │                       │ get_adapter()          │                       │
 │                    │                       ├───────────────────────►│                       │
 │                    │                       │      [loaded? ← singleton check]               │
 │                    │                       │◄───────────────────────┤                       │
 │                    │                       │ adapter.predict(wav, sr)                       │
 │                    │                       ├───────────────────────►│                       │
 │                    │                       │                        │ resample if needed    │
 │                    │                       │                        ├────►librosa           │
 │                    │                       │                        │ fe(wav, sr=16k)       │
 │                    │                       │                        ├────►AutoFeatureExtr…  │
 │                    │                       │                        │ model(**features)     │
 │                    │                       │                        ├──────────────────────►│
 │                    │                       │                        │   logits (7,)         │
 │                    │                       │                        │◄──────────────────────┤
 │                    │                       │                        │ softmax → probs       │
 │                    │                       │                        │ zip(labels, probs)    │
 │                    │                       │◄───────────────────────┤                       │
 │                    │                       │ RawPrediction dict     │                       │
 │                    │◄──────────────────────┤                        │                       │
 │ render bar plot +  │                       │                        │                       │
 │ emotion label      │                       │                        │                       │
 │◄───────────────────┤                       │                        │                       │
```

### 7.2. Performance budget

Trên RTX 4050 (sm_89), FP16:

| Bước | Latency |
|---|---|
| Audio load + format detect (`librosa.load`) | ~5 ms |
| Resample nếu cần | ~5 ms |
| `fe(wav, return_tensors="pt")` | ~3 ms |
| `.to(cuda)` transfer | ~2 ms |
| Model forward (Whisper + ECAPA + LoRA + head) | ~45 ms |
| Softmax + post-process | ~1 ms |
| CPU ↔ GPU transfer kết quả | ~2 ms |
| **Total / 10s clip** | **~63 ms** |
| **/ 30s clip** | **~150 ms** |

Trên CPU (7-thread i5, FP32): **~3 s / 10s clip** (40× chậm hơn GPU).

### 7.3. Memory budget

| Thành phần | RAM (CPU) | VRAM (GPU) |
|---|---|---|
| Model weights | 1.2 GB (FP32) | 0.6 GB (FP16) |
| Activations (10s clip) | ~200 MB | ~150 MB |
| Working buffer | ~300 MB | ~250 MB |
| **Total** | **~1.7 GB** | **~1.0 GB** |
| Free budget (16 GB CPU / 6 GB GPU) | ~14 GB | ~5 GB |

Dev host cần ≥ 1.5 GiB free RAM để load trên CPU. Nếu thiếu, app
không crash mà chỉ UI banner vàng "model chưa sẵn sàng".

---

## 8. Benchmark — `bench/`

### 8.1. Source of truth

Mọi con số trong `README.md`, `BENCHMARK.md`, slides đều lấy từ
**`bench/results/scores.json`**. Đây là file immutable — nếu muốn update
số liệu, phải chạy lại benchmark và ghi đè.

```json
{
  "meralion-ser-v1": {
    "accuracy": 0.4025,
    "macro_f1": 0.407,
    "num_samples": 400,
    "eligible_for_deploy": false,
    "device": "cuda",
    "dtype": "float16",
    "per_class_f1": {
      "happy":   0.3097,
      "neutral": 0.4952,
      "sad":     0.3259,
      "angry":   0.497
    },
    "notes": "Full ViSEC bench on device=cuda dtype=float16"
  }
}
```

`eligible_for_deploy: false` được set vì accuracy < 65% stop criterion.
Field này có thể được CI check để block merge.

### 8.2. CSV streaming (SIGKILL-safe)

`run_meralion.py` ghi prediction từng clip vào
`bench/results/meralion_predictions.csv` qua `fh.flush()`:

```python
with output_csv.open("a") as fh:
    writer = csv.writer(fh)
    if not header_written:
        writer.writerow(HEADER)
        header_written = True
    for clip in visec_iter:
        pred = adapter.predict(clip.wav, clip.sr)
        writer.writerow([clip.id, clip.label, pred.label,
                         pred.confidence, time.time() - t0])
        fh.flush()    # ← quan trọng: ghi xuống đĩa ngay
```

Nếu benchmark bị Ctrl+C / OOM kill giữa chừng, file CSV vẫn dùng được
cho debugging — không mất hết progress.

### 8.3. Subset balancing

```python
def load_visec(per_class: int = 100, seed: int = 0):
    ds = load_dataset("hustep-lab/ViSEC", split="train")
    by_label = defaultdict(list)
    for ex in ds:
        by_label[ex["emotion"]].append(ex)
    rng = random.Random(seed)
    return [
        rng.sample(items, per_class)
        for items in by_label.values()
    ]
```

`seed=0` cố định → mọi lần chạy đều trên cùng 400 clips → công bằng
so sánh.

---

## 9. Testing strategy

### 9.1. Các tier test

| Tier | File | Mục đích | Tốc độ |
|---|---|---|---|
| **Fabrication guards** | `test_no_fabrication.py` (7 tests) | Chặn việc giả scores / label merging | < 1 s |
| **Inference wrapper** | `test_inference.py` (6 tests) | Singleton, schema, error format | < 1 s |
| **Audio loading** | `test_audio.py` (4 tests) | Resample + format detect | < 2 s |
| **ViSEC loader** | `test_visec_loader.py` (3 tests) | Balanced subset, deterministic seed | < 5 s |
| **End-to-end (slow)** | `tests/manual/` | Real model + real audio (skip CI) | ~30 s |

Default `pytest -m "not slow"` chỉ chạy 4 tier đầu. Tổng ~20 tests,
chạy trong < 10 giây.

### 9.2. Fabrication guards (chi tiết)

File `test_no_fabrication.py` chặn nhiều kiểu fabrication:

```python
def test_class_scores_are_finite_in_unit_interval(rng):
    """Không có NaN, Inf, hoặc scores < 0 hoặc > 1."""
    ...

def test_class_scores_sum_to_1(rng):
    """Tổng scores = 1.0 ± 0.01, không phải uniform."""
    ...

def test_no_uniform_distribution(rng):
    """Distribution phải có variance > 0.001 (không flatten)."""
    ...

def test_raw_labels_not_merged_into_neutral():
    """7 labels phải giữ nguyên — không merge fearful/disgusted/surprised."""
    raw = ["neutral", "happy", "sad", "angry",
           "fearful", "disgusted", "surprised"]
    assert set(raw) == MeralionAdapter.labels
    ...
```

Mỗi test document lý do "vì sao đây là bug" trong docstring — để
contributor sau đọc hiểu context.

### 9.3. Stubbing cho tests

Vì test không nên download model thật (1.2 GB), `conftest.py` stub:

```python
@pytest.fixture(autouse=True)
def stub_meralion(monkeypatch):
    """Replace MeralionAdapter with a stub that returns deterministic logits."""
    class StubAdapter(BaseAdapter):
        model_id = "stub/meralion-ser-v1"
        sample_rate = 16_000
        labels = ["neutral","happy","sad","angry",
                  "fearful","disgusted","surprised"]

        def load(self): self._loaded = True
        def predict(self, wav, sr):
            return RawPrediction(
                label="happy",
                scores=[{"label": l, "score": random.random()}
                        for l in self.labels],
                raw="happy",
                latency_ms=1,
                device="cpu",
                dtype="float32",
            )
        def get_model_info(self): return ModelInfo(...)

    monkeypatch.setitem(inference.REGISTRY, "meralion-ser-v1", StubAdapter)
```

Nhờ Adapter pattern, stub thay thế được model thật không cần download.

---

## 10. CI / quality gates

Project không có CI chính thức nhưng conventions:

- **PR merge**: chạy `pytest -m "not slow"` phải pass 20/20 tests.
- **Documentation**: nếu thay đổi `app.py`, phải update screenshot trong
  `docs/PRESENTATION.md` (slide 15).
- **Numbers**: nếu thay đổi benchmark, update cả 3 file
  `README.md`, `docs/BENCHMARK.md`, `docs/PRESENTATION.md`.
- **No new deps**: PR thêm dependency phải được approve. Hiện tại chỉ
  torch / transformers / gradio / librosa / soundfile.

---

## 11. Extension points

### 11.1. Thêm model mới

```python
# src/adapters/my_model.py
class MyModelAdapter(BaseAdapter):
    model_id = "huggingface-id/my-ser-model"
    sample_rate = 16_000
    labels = [...]

    def load(self): ...
    def predict(self, wav, sr): ...
    def get_model_info(self): ...

# src/adapters/__init__.py
REGISTRY["my-model"] = MyModelAdapter
```

UI không cần sửa — chỉ cần adapter đăng ký vào REGISTRY.

### 11.2. Thêm input format

`src/audio.py::load_audio()` có thể mở rộng để hỗ trợ MIME type mới.

### 11.3. Thêm output channel

Nếu sau này muốn gửi kết quả qua webhook / lưu database,
sửa `inference.predict()` để thêm side-effect. Singleton pattern giữ
cho state không bị duplicate.

---

## 12. Triết lý thiết kế

1. **Đơn giản trước, tối ưu sau** — single-model trước, multi-model sau
   (nếu cần). Multi-model Space trước đó có 525 LOC app.py, giờ chỉ
   270.
2. **Adapter thay vì hard-code** — interface `BaseAdapter` cho phép
   plug-and-play model. Stub test dễ viết.
3. **Source-of-truth single file** — `bench/results/scores.json` là
   chỗ duy nhất chứa numbers. Doc chỉ mirror.
4. **No fabrication** — tests chặn việc uniform distribution, NaN, label
   merging. Mọi số liệu phải đo được.
5. **Honest limitations** — README nói rõ "stop criterion unmet, 40% chưa
   phải SOTA". Không oversell.
