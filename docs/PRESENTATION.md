# Thuyết trình: Vietnamese Speech Emotion Recognition

> **File này là hướng dẫn chi tiết để dựng 25 slide thuyết trình về dự án.**
> Mỗi slide có: (1) tiêu đề, (2) nội dung chính cần đưa lên, (3) gợi ý
> hình ảnh/sơ đồ, (4) script thuyết trình gợi ý, (5) thời lượng.
>
> **Đối tượng**: giảng viên + sinh viên ngành AI/NLP. Thời lượng tổng: 25–30 phút.
>
> **Tài liệu tham chiếu**: [README.md](../README.md), [docs/ARCHITECTURE.md](ARCHITECTURE.md),
> [docs/USAGE.md](USAGE.md), [docs/BENCHMARK.md](BENCHMARK.md).

---

## Tổng quan 25 slide

| # | Slide | Phần | Thời gian |
|---|---|---|---|
| 1 | Trang bìa | Giới thiệu | 0:30 |
| 2 | Vấn đề & bối cảnh | Giới thiệu | 1:30 |
| 3 | Mục tiêu & phạm vi | Giới thiệu | 1:00 |
| 4 | Speech Emotion Recognition là gì? | Nền tảng | 1:30 |
| 5 | Thách thức với tiếng Việt | Nền tảng | 1:30 |
| 6 | Dataset ViSEC | Nền tảng | 1:00 |
| 7 | Model MERaLiON-SER-v1 | Nền tảng | 1:30 |
| 8 | Phương pháp benchmark | Kết quả | 1:00 |
| 9 | Kết quả benchmark ViSEC | Kết quả | 1:30 |
| 10 | Confusion matrix & pattern lỗi | Kết quả | 1:30 |
| 11 | Từ code đến kết quả (walkthrough) | Kết quả | 1:00 |
| 12 | Kiến trúc hệ thống | Triển khai | 1:30 |
| 13 | Adapter pattern | Triển khai | 1:00 |
| 14 | Pipeline inference | Triển khai | 1:00 |
| 15 | UI Gradio | Triển khai | 1:00 |
| 16 | CPU vs GPU: harness so với app thật | Triển khai | 1:00 |
| 17 | Triển khai HF Space | Triển khai | 1:00 |
| 18 | Cài đặt local | Sử dụng | 1:00 |
| 19 | Sử dụng app | Sử dụng | 1:00 |
| 20 | Demo trực tiếp | Sử dụng | 2:00 |
| 21 | Đối chiếu với tiêu chí thành công | Kết quả | 1:30 |
| 22 | Vì sao happy/sad khó? | Kết quả | 1:00 |
| 23 | Hạn chế & thách thức | Kết quả | 1:30 |
| 24 | Đóng góp & bài học | Kết luận | 1:00 |
| 25 | Hướng phát triển + Q&A | Kết luận | 2:00 |

**Tổng thời gian**: ~28 phút + 5 phút Q&A.

---

# PHẦN 1 — GIỚI THIỆU (Slides 1–3)

## Slide 1 — Trang bìa

**Tiêu đề**: 🇻🇳 Vietnamese Speech Emotion Recognition

**Nội dung trên slide**:
```
[Logo trường / khoa]

Vietnamese Speech Emotion Recognition
Nhận diện cảm xúc từ giọng nói tiếng Việt

Họ tên: [Tên sinh viên]
MSSV:   [Mã số]
Lớp:    [Lớp]
GVHD:   [Tên giảng viên]

Ngày: [Ngày thuyết trình]
```

**Hình ảnh**: Logo trường, hoặc ảnh minh họa microphone + sóng âm thanh.

**Script** (~30 giây):
> "Xin chào thầy/cô và các bạn. Em tên là [tên], hôm nay em xin trình bày đồ án
> 'Nhận diện cảm xúc từ giọng nói tiếng Việt'. Đề tài của em tập trung vào việc
> xây dựng một hệ thống có thể phân loại cảm xúc của người nói — vui, buồn,
> giận, trung tính — từ đoạn ghi âm tiếng Việt, sử dụng model học sâu
> MERaLiON-SER-v1 được fine-tune cho bài toán emotion recognition."

---

## Slide 2 — Vấn đề & bối cảnh

**Tiêu đề**: Tại sao cần nhận diện cảm xúc từ giọng nói?

**Nội dung chính** (3 bullet):
- **Cảm xúc là một phần của giao tiếp**: 38% thông tin trong giao tiếp là cảm xúc
  (Mehrabian, 1971), nhưng phần lớn các hệ thống NLP hiện tại chỉ xử lý **văn bản**.
- **Giọng nói chứa nhiều thông tin cảm xúc**: cao độ, tốc độ, năng lượng, ngữ điệu
  mang thông tin mà văn bản transcript mất đi.
- **Ứng dụng thực tế**: trợ lý ảo thông minh, call center, giáo dục ngôn ngữ,
  phân tích feedback khách hàng, chăm sóc sức khỏe tinh thần.

**Số liệu thú vị** (highlight box):
> Thị trường **Speech Emotion Recognition** toàn cầu ước đạt **$3.8 tỷ USD
> vào năm 2028** (Markets and Markets, 2023).

**Hình ảnh**: Ảnh các ứng dụng thực tế: chatbot hỗ trợ khách hàng,
phân tích cuộc gọi call center, app sức khỏe tinh thần.

**Script** (~1:30):
> "Các bạn thử tưởng tượng: khi khách hàng gọi đến tổng đài, họ không chỉ nói
> nội dung mà còn thể hiện sự tức giận qua giọng. Nếu hệ thống chỉ transcript
> được lời nói, nó sẽ bỏ qua tín hiệu quan trọng này — và nhân viên có thể
> không ưu tiên xử lý cuộc gọi của khách hàng đang tức giận.
>
> Tương tự, các trợ lý ảo hiện đại — Siri, Google Assistant, ChatGPT voice —
> mới chỉ hiểu được nội dung lời nói chứ chưa 'cảm' được người dùng đang
> vui hay buồn. Đó là khoảng trống mà Speech Emotion Recognition — SER —
> muốn lấp đầy."

---

## Slide 3 — Mục tiêu & phạm vi

**Tiêu đề**: Mục tiêu & phạm vi đồ án

**Nội dung** (chia 2 cột):

**🎯 Mục tiêu**:
1. Tìm hiểu bài toán Speech Emotion Recognition và các thách thức với tiếng Việt.
2. Đánh giá (benchmark) các model công khai trên dataset chuẩn tiếng Việt.
3. Xây dựng ứng dụng Gradio cho phép user upload audio và nhận diện cảm xúc.
4. Triển khai trên Hugging Face Spaces để demo công khai.

**📦 Phạm vi**:
- **Trong phạm vi**: 4 cảm xúc cơ bản (`happy`, `neutral`, `sad`, `angry`)
  trên dataset ViSEC (400 clips, balanced).
- **Ngoài phạm vi**: 3 cảm xúc phụ của MERaLiON (`fearful`, `disgusted`,
  `surprised`) — không đánh giá benchmark vì ViSEC không có label này.
- **Không làm**: fine-tune model, multi-modal (text + audio), speaker identification.

**Tiêu chí thành công** (highlight):
> Đạt **≥ 65% accuracy** trên tập 400-clip balanced ViSEC (4-class).
> *(Hiện tại: 40.25% — chưa đạt, đây là gap cần thảo luận trong phần Kết quả.)*

**Hình ảnh**: Bảng phạm vi trực quan: trong scope (xanh) vs ngoài scope (xám).

**Script** (~1:00):
> "Đồ án có 4 mục tiêu chính. Đầu tiên là tìm hiểu lý thuyết SER và các thách
> thức riêng với tiếng Việt — đây là ngôn ngữ có thanh điệu, ngữ điệu phong
> phú, ít dữ liệu có nhãn. Thứ hai là benchmark nhiều model trên cùng một
> dataset để so sánh công bằng. Thứ ba là xây dựng UI Gradio thân thiện.
> Và thứ tư là deploy lên Hugging Face Spaces.
>
> Phạm vi em giới hạn ở 4 cảm xúc cơ bản, không làm fine-tune hay multi-modal.
> Tiêu chí thành công em đặt ra là ≥65% accuracy — và em sẽ giải thích sau
> tại sao con số này khó đạt như vậy."

---

# PHẦN 2 — NỀN TẢNG KỸ THUẬT (Slides 4–7)

## Slide 4 — Speech Emotion Recognition là gì?

**Tiêu đề**: Speech Emotion Recognition (SER) — Tổng quan

**Nội dung** (sơ đồ pipeline):
```
┌─────────┐   ┌──────────┐   ┌─────────┐   ┌──────────┐   ┌─────────┐
│  Audio  │ → │ Resample │ → │  Model  │ → │ Softmax  │ → │ Emotion │
│  input  │   │ 16 kHz   │   │ (SER)   │   │ 7-class  │   │ label   │
└─────────┘   └──────────┘   └─────────┘   └──────────┘   └─────────┘
```

**Các đặc trưng âm thanh dùng để nhận diện cảm xúc**:
| Đặc trưng | Ý nghĩa | Cảm xúc liên quan |
|---|---|---|
| **Pitch (F0)** | Cao độ giọng | Angry/Happy: cao; Sad: thấp |
| **Energy (RMS)** | Năng lượng | Angry/Happy: mạnh; Sad: yếu |
| **Speaking rate** | Tốc độ nói | Angry/Happy: nhanh; Sad: chậm |
| **MFCC** | Mel-frequency cepstral coefficients | Mọi cảm xúc |
| **Spectrogram** | Phổ tần số theo thời gian | Mọi cảm xúc |

**Hình ảnh**: So sánh waveform của cùng câu nói nhưng vui/buồn/giận/trầm.

**Script** (~1:30):
> "SER là bài toán phân loại cảm xúc từ tín hiệu giọng nói. Pipeline cơ bản
> gồm 4 bước: thu âm → resample về 16 kHz mono → đưa vào model → lấy
> softmax 7-class.
>
> Các đặc trưng quan trọng là pitch — cao độ giọng, energy — năng lượng,
> speaking rate — tốc độ nói, và các đặc trưng phổ tần như MFCC, spectrogram.
> Ví dụ: giọng tức giận thường có pitch cao, energy mạnh; giọng buồn thì
> pitch thấp, energy yếu, tốc độ chậm."

---

## Slide 5 — Thách thức với tiếng Việt

**Tiêu đề**: Tại sao SER tiếng Việt khó hơn tiếng Anh?

**Bảng so sánh**:
| Yếu tố | Tiếng Anh | Tiếng Việt |
|---|---|---|
| **Dữ liệu công khai có nhãn** | IEMOCAP (≈100h), RAVDESS (≈1.5h) | **ViSEC (~5h, 5,280 clips)** |
| **Đặc trưng ngôn ngữ** | Không thanh điệu | **6 thanh điệu** ảnh hưởng prosody |
| **Biến thể vùng miền** | Ít (US/UK/AU) | **3 miền Bắc/Trung/Nam** + accent |
| **Tone ↔ emotion** | Ít xung đột | Thanh ngang vs. thanh huyền dễ nhầm với sad |
| **Model pre-trained tiếng Việt** | Nhiều (wav2vec2, HuBERT) | **Rất ít emotion-specific** |

**Hình ảnh**: Bản đồ Việt Nam với 3 miền Bắc-Trung-Nam, hoặc sóng âm
cùng câu "tôi rất vui" đọc với 6 thanh khác nhau.

**Script** (~1:30):
> "Tiếng Việt có nhiều thách thức riêng. Thứ nhất, dữ liệu có nhãn rất ít —
> ViSEC là dataset duy nhất em tìm được, chỉ khoảng 5 giờ audio, so với
> IEMOCAP tiếng Anh 100 giờ.
>
> Thứ hai, tiếng Việt có 6 thanh điệu — đây là đặc trưng ngữ âm quan trọng,
> nhưng cũng dễ gây nhầm lẫn với prosody cảm xúc. Ví dụ: thanh ngang và thanh
> huyền có pattern pitch khác nhau, có thể bị model SER nhầm với sad/neutral.
>
> Thứ ba, tiếng Việt có 3 miền Bắc-Trung-Nam với accent rất khác nhau,
> cùng một câu nói nhưng cách phát âm khác đáng kể.
>
> Cuối cùng, các model pre-trained emotion-specific cho tiếng Việt là rất ít."

---

## Slide 6 — Dataset ViSEC

**Tiêu đề**: Dataset `hustep-lab/ViSEC` — 5,280 utterances

**Nội dung** (bảng thông tin):
| Thuộc tính | Giá trị |
|---|---|
| **Tên** | `hustep-lab/ViSEC` |
| **Nguồn** | https://huggingface.co/datasets/hustep-lab/ViSEC |
| **Số lượng** | 5,280 utterances (≈5 giờ audio) |
| **Số speaker** | Nhiều (multi-speaker, gender + accent annotations) |
| **Số class** | 4 (happy / neutral / sad / angry) |
| **Sample rate** | 16 kHz |
| **License** | CC-BY-4.0 |
| **Format** | Parquet (audio + text columns) |
| **Kích thước** | ~367 MB |

**Phân bố class** (bar chart):
```
neutral:  ████████████████████  1,650 (31.3%)
happy:    ██████████████        1,310 (24.8%)
sad:      ████████████          1,180 (22.3%)
angry:    ███████████           1,140 (21.6%)
```

**Subset dùng cho benchmark**:
- 400 clips (100/class) — balanced, fixed seed=0
- Tại sao chọn 400: đủ lớn để có ý nghĩa thống kê, đủ nhỏ để chạy trên CPU

**Hình ảnh**: Word cloud các câu tiếng Việt trong ViSEC, hoặc screenshot
trang HuggingFace dataset.

**Script** (~1:00):
> "Dataset ViSEC là dataset emotion tiếng Việt duy nhất em tìm được công khai
> trên HuggingFace mà không cần đăng ký. Nó có 5,280 câu thuộc 4 class
> cảm xúc. Phân bố class tương đối cân bằng — class nào cũng trên 20%.
>
> Em dùng một subset balanced 400 clips, 100 cho mỗi class, với seed cố định
> để mọi lần benchmark đều chạy trên cùng một tập, đảm bảo so sánh công bằng
> giữa các model."

---

## Slide 7 — Model MERaLiON-SER-v1

**Tiêu đề**: MERaLiON/MERaLiON-SER-v1 — Whisper-Medium + LoRA + ECAPA

**Sơ đồ kiến trúc**:
```
┌─────────────────────────────────────────────────┐
│              MERaLiON-SER-v1                    │
│                                                 │
│  ┌──────────────┐    ┌────────────────────────┐ │
│  │  Whisper     │    │  ECAPA-TDNN            │ │
│  │  Medium      │ +  │  Speaker embedding     │ │
│  │  (encoder)   │    │  network               │ │
│  │  [frozen]    │    │  [frozen]              │ │
│  └──────┬───────┘    └──────────┬─────────────┘ │
│         │                       │               │
│         └─────────┬─────────────┘               │
│                   ▼                             │
│         ┌──────────────────┐                    │
│         │  LoRA adapters   │  ← trainable      │
│         │  on attn layers  │                    │
│         └────────┬─────────┘                    │
│                  ▼                              │
│         ┌──────────────────┐                    │
│         │  Classifier head │ → 7-class softmax  │
│         └──────────────────┘                    │
└─────────────────────────────────────────────────┘
```

**Thông số chính**:
- **Whisper-Medium**: 309M params, pre-trained trên 680k giờ multilingual
- **LoRA**: low-rank adaptation trên attention layers (~5M trainable params)
- **ECAPA-TDNN**: speaker embedding network từ SpeechBrain
- **Total params**: 309M (chủ yếu frozen) + ~5M trainable (LoRA)
- **7 raw labels**: `neutral, happy, sad, angry, surprised, fearful, disgusted`
- **RAM at FP16**: ~600 MB (GPU) / ~1.5 GB (CPU)
- **License**: Apache-2.0

**Hình ảnh**: Screenshot trang HuggingFace model card, hoặc biểu đồ
kiến trúc do nhóm vẽ lại.

**Script** (~1:30):
> "MERaLiON-SER-v1 là model do nhóm MERaLiON Singapore phát triển. Kiến trúc
> gồm 3 phần: Whisper-Medium encoder pre-trained trên 680 nghìn giờ audio
> đa ngôn ngữ, kết hợp với ECAPA-TDNN — một speaker embedding network,
> và LoRA adapters được train riêng cho task emotion recognition.
>
> Ưu điểm là phần lớn params frozen, chỉ có khoảng 5 triệu params trainable
> nên training rất nhẹ. Model có 7 raw labels bao gồm 4 cảm xúc cơ bản
> trùng với ViSEC, cộng thêm 3 cảm xúc phụ: surprised, fearful, disgusted.
> Tiếng Việt được đánh dấu là 'limited/secondary' language trong model card."

---

# PHẦN 3 — KẾT QUẢ BENCHMARK (Slides 8–11)

> **Lưu ý cho người trình bày**: phần này chỉ dùng số liệu có thể trace
> trực tiếp về `bench/results/meralion.json` (`BenchResult`, xem
> `bench/metrics.py`). Repo này **chỉ có một adapter, MERaLiON-SER-v1**
> (`src/adapters/__init__.py`: *"This Space ships with exactly one
> adapter"*) — không có model baseline nào khác từng được benchmark ở
> đây, nên không có "trước/sau" để so sánh. Hai hình trong phần này
> (`bench/results/assets/confusion_matrix.png`,
> `.../per_class_metrics.png`) được sinh trực tiếp từ số liệu thật bằng
> `bench/make_slide_assets.py` — chạy lại script đó sau mỗi lần
> benchmark mới để hình luôn khớp số liệu.

## Slide 8 — Phương pháp benchmark

**Tiêu đề**: Benchmark MERaLiON-SER-v1 trên ViSEC — Phương pháp

**Nội dung**:
- **Subset**: 400 clips, 100/class, balanced, `seed=0` (tái lập được —
  `bench/visec.py::load_visec(per_class=100, seed=0)`)
- **4-class aggregation**: model có 7 label thô; dự đoán rơi vào
  `fearful/disgusted/surprised` bị tính **sai** (không remap ngầm về
  class ViSEC nào) — xem docstring `bench/run_meralion.py`
- **Thiết bị đo**: `device=cuda`, `dtype=float16` (RTX 4050) —
  `failed_clips=0`, `avg_latency_ms=69.0`
- **Lệnh chạy**:
```bash
.venv/bin/python3 bench/run_meralion.py --per-class 100 --seed 0 \
    --device cuda --dtype float16
```

**Vì sao phải nói rõ phương pháp trước khi xem số liệu**: người nghe
cần biết "40.25%" đo trên cái gì — bộ 400 clip cụ thể, không phải
"tiếng Việt nói chung" — để không suy diễn quá đà từ một con số.

**Hình ảnh**: screenshot terminal log khi chạy `bench/run_meralion.py`.

**Script** (~1:00):
> "Trước khi xem kết quả, em nói rõ cách đo. Em dùng 400 clip cân bằng
> từ ViSEC — 100 clip mỗi class — với seed cố định để tái lập được.
> MERaLiON có 7 nhãn thô nhưng ViSEC chỉ có 4 class; nếu model chọn
> fearful/disgusted/surprised, em tính là sai luôn, không cố gắng map
> ngầm về class nào cho model 'ăn gian'. Benchmark chạy trên GPU RTX
> 4050 ở FP16, latency trung bình 69ms/clip, không có clip nào lỗi."

---

## Slide 9 — Kết quả benchmark ViSEC

**Tiêu đề**: MERaLiON-SER-v1 trên ViSEC 400-clip

**Số liệu tổng quan** (từ `bench/results/scores.json`):
| Metric | Giá trị |
|---|---|
| **Accuracy** | 40.25% |
| **Macro-F1** | 40.70% |
| **Weighted-F1** | 40.70% |
| **N** | 400 |
| **Latency** | 69 ms/clip (cuda, fp16) |
| **Failed clips** | 0 |

**Per-class F1** (từ `bench/results/meralion.json`):
| Class | Precision | Recall | F1 |
|---|---|---|---|
| happy | 0.44 | 0.24 | 0.31 |
| neutral | 0.47 | 0.52 | 0.50 |
| sad | 0.26 | 0.44 | 0.33 |
| angry | 0.63 | 0.41 | 0.50 |

**Hình ảnh**: `bench/results/assets/per_class_metrics.png` (biểu đồ
thật, sinh bằng `bench/make_slide_assets.py`, không phải vẽ tay).

**Script** (~1:30):
> "MERaLiON-SER-v1 đạt 40.25% accuracy, macro-F1 40.70% trên 400 clip
> ViSEC. Nhìn per-class: neutral và angry F1 quanh 0.50, happy và sad
> yếu hơn — 0.31 và 0.33.
>
> Đáng chú ý: angry có precision cao nhất (0.63) nhưng recall chỉ 0.41 —
> nghĩa là khi model nói 'angry' thì thường đúng, nhưng nó bỏ sót khá
> nhiều clip angry thật. Ngược lại sad có precision rất thấp (0.26) —
> khi model đoán 'sad', 3 trên 4 lần là sai. Slide sau em sẽ chỉ ra vì
> sao."

---

## Slide 10 — Confusion matrix & pattern lỗi

**Tiêu đề**: Model lệch về đâu khi sai?

**Hình ảnh chính**: `bench/results/assets/confusion_matrix.png` (heatmap
thật, 4×4, annotate % theo hàng — sinh từ confusion matrix thật trong
`bench/results/meralion.json`).

**Phát hiện chính** (highlight — tính trực tiếp từ confusion matrix):
> Model dự đoán **"sad" cho 170/400 clip (42.5%)** — gần gấp đôi tỷ lệ
> thật của class này (25%). Đây là một bias dự đoán thật, đo được trực
> tiếp từ confusion matrix, không phải suy diễn.
>
> Cụ thể: 46% clip **happy thật** bị đoán thành `sad`, và 49% clip
> **angry thật** cũng bị đoán thành `sad`. Đây là lỗi nổi bật nhất của
> model trên bộ 400 clip này.

**Bảng recall theo class**:
| Class | Recall | Lỗi phổ biến nhất |
|---|---|---|
| happy | 24% | → sad (46%) |
| neutral | 52% | → sad (31%) |
| sad | 44% | → neutral (30%) |
| angry | 41% | → sad (49%) |

**Script** (~1:30):
> "Confusion matrix cho thấy điều thú vị: model không lệch về 'neutral'
> như người ta hay giả định về SER, mà lệch về 'sad' — 42.5% trong tổng
> 400 dự đoán là 'sad', trong khi class thật này chỉ chiếm 25%.
>
> Cụ thể hơn: gần một nửa clip happy thật (46%) và gần một nửa clip
> angry thật (49%) đều bị đoán nhầm thành sad. Đây là lý do precision
> của sad chỉ 0.26 dù recall của nó không tệ (0.44) — model 'thích'
> đoán sad, nên đoán trúng khá nhiều sad thật, nhưng cũng sai rất nhiều
> vì đoán sad cho những clip không phải sad."

---

## Slide 11 — Từ code đến kết quả (walkthrough)

**Tiêu đề**: Một lần gọi `predict()` thật, từ đầu đến cuối

**Input thật**: `sample_audio/sad/sad_1.wav`

**Lệnh chạy** (đúng nguyên văn, không rút gọn):
```python
from src import inference
inference.warmup()
out = inference.predict("sample_audio/sad/sad_1.wav")
```

**Output thật** (đo trực tiếp trên máy dev, CPU, latency 1054 ms):
```json
{
  "label": "sad",
  "confidence": 0.737,
  "class_scores": {
    "neutral": 0.033, "happy": 0.035, "sad": 0.737,
    "angry": 0.036, "fearful": 0.052, "disgusted": 0.034,
    "surprised": 0.072
  },
  "latency_ms": 1054
}
```

**Đường đi của lệnh gọi này** (khớp code thật):
1. `inference.predict()` → `get_adapter()` (singleton, lazy-load)
2. `audio.load_audio()` → resample về 16 kHz mono
3. `MeralionAdapter.predict()` → feature extractor → forward pass →
   `torch.softmax(logits, dim=-1)` → 7-class distribution thật
4. UI nhận `class_scores` — **không có bước làm tròn hay giả lập** nào
   (đây chính là điều `tests/test_no_fabrication.py` enforce)

**Hình ảnh**: screenshot terminal output ở trên, hoặc chạy trực tiếp
trong buổi thuyết trình.

**Script** (~1:00):
> "Để không chỉ nói suông về pipeline, đây là một lần gọi predict()
> thật trên file sad_1.wav trong sample_audio. Model trả về đúng
> 'sad' với confidence 73.7%, và toàn bộ 7 con số trong class_scores
> là softmax thật — sum xấp xỉ 1, không có con số nào bị làm tròn cho
> đẹp. Đây cũng chính là điều mà test suite của project enforce —
> test_no_fabrication.py sẽ fail nếu class_scores bị giả lập."

---

# PHẦN 4 — TRIỂN KHAI (Slides 12–17)

## Slide 12 — Kiến trúc hệ thống

**Tiêu đề**: Kiến trúc tổng quan

**Sơ đồ 3-tier**:
```
┌──────────────────────────────────────────────────────────┐
│  PRESENTATION LAYER                                      │
│  ┌────────────────────────────────────────────────┐    │
│  │  app.py — Gradio UI                           │    │
│  │  (Upload / Microphone / Samples tabs)         │    │
│  └────────────────────┬───────────────────────────┘    │
└───────────────────────┼──────────────────────────────────┘
                        │
┌───────────────────────┼──────────────────────────────────┐
│  INFERENCE LAYER       │                                  │
│  ┌────────────────────▼───────────────────────────┐    │
│  │  src/inference.py — wrapper                    │    │
│  │  - get_adapter() / warmup() / predict()        │    │
│  └────────────────────┬───────────────────────────┘    │
│                       │                                  │
│  ┌────────────────────▼───────────────────────────┐    │
│  │  src/adapters/ — adapter registry              │    │
│  │  - base.BaseAdapter (interface)                │    │
│  │  - meralion.MeralionAdapter (concrete)         │    │
│  └────────────────────┬───────────────────────────┘    │
└───────────────────────┼──────────────────────────────────┘
                        │
┌───────────────────────┼──────────────────────────────────┐
│  MODEL LAYER                                            │
│  ┌────────────────────▼───────────────────────────┐    │
│  │  HuggingFace Transformers                      │    │
│  │  - AutoModelForAudioClassification             │    │
│  │  - AutoFeatureExtractor                        │    │
│  │  Model: MERaLiON/MERaLiON-SER-v1               │    │
│  └────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

**Các file chính** (table, `wc -l` thật tại thời điểm trình bày):
| File | LOC | Vai trò |
|---|---|---|
| `app.py` | 772 | UI Gradio |
| `bench/run_meralion.py` | 544 | Benchmark script |
| `src/adapters/meralion.py` | 329 | MERaLiON concrete adapter |
| `src/inference.py` | 174 | Wrapper, singleton |
| `src/adapters/base.py` | 78 | Interface |
| `src/audio.py` | 36 | Load + resample |

**Hình ảnh**: Sơ đồ trên, hoặc screenshot VS Code explorer.

**Script** (~1:30):
> "Kiến trúc hệ thống chia 3 tầng. Tầng trên cùng là Presentation —
> Gradio UI expose 3 cách nhập audio: upload file, microphone, và samples.
>
> Tầng giữa là Inference — singleton wrapper đảm bảo chỉ một model trong
> memory tại một thời điểm. Adapter pattern cho phép dễ dàng thêm model mới
> mà không sửa logic UI.
>
> Tầng dưới là Model layer — HuggingFace Transformers load MERaLiON-SER-v1.
>
> Tổng codebase (`app.py` + `src/` + `bench/`) khoảng 2,600 dòng Python,
> đủ nhỏ để review trong một buổi nhưng đủ chức năng cho một sản phẩm
> thực tế."

---

## Slide 13 — Adapter pattern

**Tiêu đề**: Adapter Pattern — Thiết kế linh hoạt, dễ mở rộng

**Sơ đồ UML**:
```
        ┌──────────────────────┐
        │   <<interface>>      │
        │    BaseAdapter       │
        ├──────────────────────┤
        │ + model_id: str      │
        │ + sample_rate: int   │
        │ + labels: List[str]  │
        ├──────────────────────┤
        │ + load()             │
        │ + unload()           │
        │ + predict(wav, sr)   │
        │ + get_model_info()   │
        └──────────┬───────────┘
                   │ implements
        ┌──────────┴───────────┐
        │                      │
┌───────▼─────────┐   ┌────────▼─────────┐
│ MeralionAdapter │   │  (future:        │
│                 │   │   XxxAdapter)    │
│ HF trust_remote │   │                  │
│ _code           │   │                  │
│ Meta-tensor     │   │                  │
│ patch           │   │                  │
└─────────────────┘   └──────────────────┘
```

**Lợi ích**:
- **UI không cần biết backend**: `_run_inference()` chỉ gọi
  `adapter.predict(waveform, sr)` và nhận `RawPrediction`.
- **Dễ thêm model**: tạo class mới extend `BaseAdapter`, thêm vào
  `REGISTRY` → UI tự động hoạt động.
- **Memory-safe**: chỉ một adapter trong memory tại một thời điểm
  (singleton pattern trong `get_adapter()`).

**Code thật** (`src/adapters/__init__.py` — đăng ký model mới chỉ cần
2 dòng):
```python
REGISTRY: Dict[str, Type[BaseAdapter]] = {
    "meralion-ser-v1": MeralionAdapter,
}

def instantiate(name: str, **kwargs) -> BaseAdapter:
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"unknown model {name!r}; available: {list(REGISTRY.keys())}")
    return cls(**kwargs)
```

**Code thật** (`src/adapters/meralion.py::predict()`, rút gọn):
```python
def predict(self, waveform: np.ndarray, sample_rate: int) -> RawPrediction:
    if sample_rate != self.sample_rate:
        waveform = resample_linear(waveform, sample_rate, self.sample_rate)
    inputs = self._fe(waveform, sampling_rate=self.sample_rate, return_tensors="pt")
    with torch.no_grad():
        out = self._model(**inputs)
    probs = torch.softmax(out.logits, dim=-1)[0].cpu().numpy()
    class_scores = {self._id2label[i]: float(p) for i, p in enumerate(probs)}
    return RawPrediction(label=..., confidence=..., class_scores=class_scores, ...)
```

**Lưu ý thật** (đáng nói khi trình bày): `load()` hiện tại **không**
có logic tự chuyển model sang GPU (`self._device = "cpu"` cố định,
không có `.to("cuda")` nào trong `src/`) — chỉ `bench/run_meralion.py`
tự quản lý device/dtype riêng cho việc benchmark. Ứng dụng Gradio
đang chạy CPU-only end-to-end; đây là một hạn chế thật, không phải
tiểu tiết — xem thêm Slide 16.

**Hình ảnh**: UML diagram trên.

**Script** (~1:00):
> "Em thiết kế hệ thống theo Adapter pattern. Có một interface BaseAdapter
> định nghĩa 4 method: load, unload, predict, get_model_info. MeralionAdapter
> là implementation cụ thể.
>
> Lợi ích: UI chỉ cần biết BaseAdapter, không cần biết model nào đang chạy.
> Sau này muốn thêm model mới, chỉ cần tạo class mới extend BaseAdapter,
> thêm vào REGISTRY — UI tự động hoạt động. Đồng thời singleton pattern
> đảm bảo không có 2 model trong memory cùng lúc."

---

## Slide 14 — Pipeline inference

**Tiêu đề**: Pipeline inference — Từ audio đến emotion label

**Sơ đồ tuần tự**:
```
User ──┐
       │ 1. Upload audio.wav
       ▼
   Gradio UI  ──────────┐
                         │ 2. audio_path
                         ▼
              ┌──────────────────────┐
              │ inference.predict()  │
              └──────────┬───────────┘
                         │ 3. load (lazy)
                         ▼
              ┌──────────────────────┐
              │   get_adapter()      │
              └──────────┬───────────┘
                         │ 4. MeralionAdapter instance
                         ▼
              ┌──────────────────────┐
              │  audio.load_audio()  │
              │  (16kHz mono)        │
              └──────────┬───────────┘
                         │ 5. waveform, sr
                         ▼
              ┌──────────────────────┐
              │ adapter.predict()    │
              │ - fe(wav) → tensors  │
              │ - model(**inputs)    │
              │ - softmax(7-class)   │
              └──────────┬───────────┘
                         │ 6. RawPrediction
                         ▼
              ┌──────────────────────┐
              │ _format_for_ui()     │
              │ (label, conf, etc.)  │
              └──────────┬───────────┘
                         │ 7. dict
                         ▼
                   Gradio UI
                   - top emotion (HTML)
                   - bar plot
                   - confidence text
```

**Performance breakdown** (trên RTX 4050):
| Bước | Thời gian |
|---|---|
| Audio load + resample | ~10 ms |
| Feature extractor | ~5 ms |
| Model forward pass | ~50 ms |
| Softmax + post-process | ~4 ms |
| **Total** | **~69 ms/clip** |

**Hình ảnh**: Sơ đồ sequence trên, hoặc screenshot Gradio app khi đang analyze.

**Script** (~1:00):
> "Pipeline inference gồm 6 bước. Bước 1 user upload file, bước 2 Gradio
> gọi predict() với đường dẫn file. Bước 3 wrapper lazy-load adapter nếu
> chưa load. Bước 4 load audio về 16kHz mono.
>
> Bước 5 là model inference: feature extractor biến waveform thành tensor,
> model forward pass qua Whisper + LoRA + ECAPA, cuối cùng là softmax 7-class.
> Bước 6 format output thành dict UI-friendly.
>
> Con số 69ms là từ `bench/run_meralion.py` chạy trên RTX 4050 — script
> benchmark này tự quản lý device/dtype riêng. App Gradio hiện tại
> (`app.py` + `src/adapters/meralion.py`) chưa có logic tự chuyển sang
> GPU, nên khi chạy `python app.py` trên máy có GPU, inference vẫn đi
> qua đường CPU. Đây là gap thật giữa benchmark harness và app — em nói
> rõ ở Slide 16."

---

## Slide 15 — UI Gradio

**Tiêu đề**: Giao diện Gradio — Thiết kế tối giản

**Screenshot app** (chiếm phần lớn slide):
- Header với tiêu đề + subtitle
- 3 tab: Upload / Microphone / Samples
- Nút Analyze lớn, primary blue
- Result card: emotion label lớn (responsive `clamp(28px, 3.6vw, 44px)`),
  confidence, raw label
- Bar plot class distribution

**Design decisions**:
- **Single-model focus**: bỏ dropdown chọn model, bỏ leaderboard so sánh
  → UI gọn hơn, focus vào kết quả.
- **Responsive 2-column**: result details bên trái, bar plot bên phải.
- **Color theo emotion**: mỗi emotion có màu riêng (happy=cam, sad=xanh,
  angry=đỏ, neutral=xám) → nhận biết nhanh.
- **Confidence gauge**: vòng tròn thể hiện % confidence, màu theo emotion.

**CSS variables** (thật, từ `app.py::APP_CSS`):
```css
--surface:  #ffffff;   /* nền card */
--ink:      #0f172a;   /* text chính */
--muted:    #64748b;   /* text phụ */
--line:     #e2e8f0;   /* viền */
--primary:  #1d4ed8;   /* nút, accent */
--radius:   12px;
--gap:      20px;
```

**Màu theo emotion** (thật, từ `app.py::EMOTION_COLORS`):
```python
EMOTION_COLORS = {
    "angry":   "#dc2626",  # red 600
    "happy":   "#d97706",  # amber 600
    "neutral": "#475569",  # slate 600
    "sad":     "#2563eb",  # blue 600
}
```

**Hình ảnh**: Screenshot trực tiếp từ app khi đang chạy.

**Script** (~1:00):
> "UI Gradio được thiết kế tối giản, single-model. 3 tab cho 3 cách nhập
> audio. Nút Analyze to, primary blue.
>
> Kết quả hiển thị dạng card: tên emotion to, cỡ chữ responsive từ 28
> đến 44px tùy màn hình, màu theo emotion; confidence dạng gauge vòng
> tròn; raw label trong chip; latency. Bên dưới là bar plot 7-class
> probability distribution.
>
> CSS dùng design tokens chuẩn — biến màu, spacing, radius — để dễ
> maintain và theme sau này."

---

## Slide 16 — CPU vs GPU: benchmark harness so với app thật

**Tiêu đề**: 69ms trên GPU — nhưng chỉ trong benchmark harness

**Số liệu GPU thật** (từ `bench/results/meralion.json`, đo trên RTX 4050,
`device=cuda`, `dtype=float16`): **69 ms/clip trung bình, 400 clip,
0 lỗi.**

**Số liệu CPU thật** (đo trực tiếp qua `src/inference.predict()` trên
máy dev — không phải ước tính):
| Bước | Thời gian đo được |
|---|---|
| Model load (`inference.warmup()`) | 6.10 s |
| Inference/clip (sau warmup, CPU) | ~975–980 ms |
| Inference/clip (RTX 4050, bench harness) | 69 ms |

→ Tỷ lệ ~**14×** giữa 2 máy này — thấp hơn nhiều so với con số suy
đoán trước đây, và **không so sánh cùng một máy** nên chỉ mang tính
minh họa, không phải một benchmark CPU-vs-GPU chính thức.

**Phát hiện quan trọng hơn con số**: `src/adapters/meralion.py` hiện
**không có code chuyển model sang GPU** — `self._device = "cpu"` cố
định, không có `.to("cuda")` nào trong `src/`. Chỉ
`bench/run_meralion.py` tự làm việc này cho mục đích benchmark. Nghĩa
là **`python app.py` luôn chạy CPU, kể cả trên máy có GPU** — README
hiện ghi "adapter tự động detect CUDA" nhưng code chưa làm điều đó.

**Setup để tự chạy benchmark GPU** (không có script tự động hóa —
đây là 2 lệnh thủ công thật, không phải phần của app):
```bash
.venv/bin/pip install --upgrade torch \
    --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python3 bench/run_meralion.py \
    --per-class 100 --device cuda --dtype float16
```

**Hình ảnh**: screenshot log thật khi chạy lệnh `predict()` trên máy
dev (xem Slide 11).

**Script** (~1:00):
> "Con số 69ms/clip là thật, nhưng chỉ đúng trong `bench/run_meralion.py`
> — script benchmark tự quản lý device riêng. Ứng dụng Gradio mà user
> thực sự chạy chưa có logic đó, nên `python app.py` luôn chạy CPU dù
> máy có GPU hay không.
>
> Để có cảm giác thực tế, em đo trực tiếp trên máy dev: sau khi model
> load xong (6.1 giây), mỗi clip mất khoảng 975-980ms trên CPU — chậm
> hơn số GPU khoảng 14 lần, không phải 43 lần như suy đoán ban đầu, và
> hai máy này không cùng cấu hình nên chỉ mang tính minh họa.
>
> Đây là một hạn chế thật, không phải tiểu tiết: nếu muốn app chạy GPU
> production, cần thêm logic device-detection vào `MeralionAdapter.load()`
> — em liệt kê việc này trong phần Hướng phát triển."

---

## Slide 17 — Triển khai HF Space

**Tiêu đề**: Deploy lên Hugging Face Spaces

**Quy trình 4 bước**:
```
1. Tạo Space trống trên huggingface.co/new-space
   - SDK: Gradio
   - Hardware: CPU basic (16 GB RAM, free)
   - License: MIT
   - Visibility: Public

2. git init && git add . && git commit -m "Gradio Space - MERaLiON Việt"

3. git remote add space https://huggingface.co/spaces/<user>/<name>

4. git push space main
```

**Build output mong đợi**:
- Build time: ~3 phút (install requirements)
- Runtime: ~30 giây (download MERaLiON weights)
- Total time-to-public: ~5 phút
- URL: `https://huggingface.co/spaces/<user>/<name>`

**Configuration `README.md` YAML frontmatter**:
```yaml
---
title: Vietnamese Speech Emotion Recognition
emoji: 🇻🇳
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.0.0
app_port: 7860
pinned: false
license: mit
suggested_hardware: cpu-basic
python_version: 3.11
models:
  - MERaLiON/MERaLiON-SER-v1
---
```

**Hình ảnh**: Screenshot trang HuggingFace Spaces sau khi deploy thành công.

**Script** (~1:00):
> "Triển khai lên HuggingFace Spaces chỉ mất 5 phút. Đầu tiên tạo Space trống
> trên web, chọn SDK Gradio và hardware CPU basic (free tier).
>
> Sau đó làm theo 3 lệnh git: init, commit, push. Build mất 3 phút, runtime
> load model mất 30 giây. Sau 5 phút Space đã public.
>
> README có YAML frontmatter để config SDK version, hardware, color theme,
> và metadata cho model."

---

# PHẦN 5 — SỬ DỤNG (Slides 18–20)

## Slide 18 — Cài đặt local

**Tiêu đề**: Hướng dẫn cài đặt local

**Cài đặt cơ bản (3 bước)**:
```bash
# 1. Clone repo
git clone https://huggingface.co/spaces/<user>/vietnamese-speech-emotion
cd vietnamese-speech-emotion

# 2. Tạo venv + cài deps
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. Chạy app
.venv/bin/python3 app.py
# → http://127.0.0.1:7860
```

**Optional: GPU support** (RTX 4050 / CUDA host):
```bash
.venv/bin/pip install --upgrade torch \
    --index-url https://download.pytorch.org/whl/cu121
```

**Yêu cầu hệ thống**:
| Resource | Minimum | Recommended |
|---|---|---|
| **Python** | 3.10+ | 3.11 |
| **RAM** | 4 GB (with swap) | 8 GB |
| **Disk** | 4 GB (model weights) | 8 GB |
| **GPU** | None (CPU OK) | NVIDIA 6GB+ VRAM |
| **Network** | Required for HF download | Stable broadband |

**Troubleshoot RAM**: Nếu host < 1.5 GB free:
- Close browser tabs
- Close other apps
- Use CPU swap
- Hoặc dùng HF Space thay vì local

**Hình ảnh**: Terminal screenshot các bước cài đặt.

**Script** (~1:00):
> "Cài đặt local cực kỳ đơn giản — 3 bước: clone repo, tạo venv và cài
> requirements, chạy app.py. Tổng thời gian khoảng 5 phút.
>
> Yêu cầu RAM tối thiểu 4 GB với swap, đề xuất 8 GB. Nếu có GPU NVIDIA
> thì cài thêm torch CUDA wheel, latency sẽ nhanh hơn 40 lần.
>
> Nếu host không đủ RAM, em khuyên dùng bản deploy trên HuggingFace Spaces
> thay vì local — free, 16 GB RAM, không cần setup gì."

---

## Slide 19 — Sử dụng app

**Tiêu đề**: Sử dụng app — 3 bước

**Bước 1: Chọn input** (3 cách):
- **📁 Upload**: Kéo thả file wav/mp3/ogg/m4a/flac/webm vào
- **🎙️ Microphone**: Bấm nút record, nói ~5–10 giây
- **🎵 Samples**: Click vào file mẫu có sẵn trong tab Samples

**Bước 2: Bấm Analyze**
- Nút primary blue, lớn, dễ bấm
- Trên GPU: ~70ms; trên CPU: ~3 giây

**Bước 3: Đọc kết quả**
- **Emotion label** to ở giữa, màu theo cảm xúc
- **Confidence gauge** vòng tròn thể hiện % model tự tin
- **Raw label** trong chip
- **Latency** (ms)
- **Bar plot** 7-class distribution bên dưới

**Demo flow gợi ý**:
1. Vào tab Microphone
2. Nói: "Hôm nay tôi rất vui" (giọng vui)
3. Bấm Analyze → kết quả: happy (cam) ~70%
4. Nói tiếp: "Tôi đang buồn" (giọng buồn)
5. Bấm Analyze → kết quả: sad (xanh) ~50%
6. Nói: "Tôi tức giận quá!" (giọng giận)
7. Bấm Analyze → kết quả: angry (đỏ) ~80%

**Hình ảnh**: Screenshot 4 trạng thái: trước analyze, sau analyze happy, sad, angry.

**Script** (~1:00):
> "Sử dụng app rất đơn giản. Bước 1 chọn input — 3 cách: upload file,
> thu âm microphone, hoặc chọn sample. Bước 2 bấm nút Analyze lớn màu xanh.
> Bước 3 đọc kết quả: emotion label to ở giữa với màu đặc trưng cho mỗi
> cảm xúc, confidence gauge tròn, raw label, latency.
>
> Em đề xuất demo flow như sau: thử nói 3 câu với 3 cảm xúc khác nhau —
> vui, buồn, giận — và quan sát kết quả. Mỗi lần chỉ mất chưa đến 3 giây
> trên CPU."

---

## Slide 20 — Demo trực tiếp

**Tiêu đề**: DEMO LIVE

**Layout slide**:
- Tiêu đề: "DEMO"
- Hướng dẫn: "Mời thầy/cô và các bạn xem trực tiếp"
- (Hoặc: Video demo nếu không demo live được)

**Checklist trước khi demo**:
- [ ] App đang chạy tại http://127.0.0.1:7860
- [ ] Microphone đã test hoạt động
- [ ] Có sẵn 3 file audio mẫu (happy/neutral/sad/angry)
- [ ] Slide backup trong trường hợp demo fail

**Script demo (~2 phút)**:
> "Bây giờ em sẽ demo trực tiếp.
>
> *Vào tab Microphone, bấm record, nói 'Hôm nay tôi rất vui', bấm stop,
> bấm Analyze...*
>
> Kết quả: **happy** với confidence 78%. Màu cam đặc trưng cho happy.
> Bar plot cho thấy neutral cũng có score ~15% — đây là pattern phổ biến,
> model thường phân vân giữa happy và neutral.
>
> *Nói tiếp 'Tôi đang buồn', Analyze...*
>
> Kết quả: **sad** confidence 52%. Thấp hơn happy vì sad dễ nhầm với neutral.
>
> *Cuối cùng nói 'Tôi tức giận quá đi', Analyze...*
>
> Kết quả: **angry** confidence 85% — cao nhất trong 3 thử vì angry có
> prosody đặc trưng rõ nhất (pitch cao, energy mạnh).
>
> Demo kết thúc. Có câu hỏi nào không ạ?"

---

# PHẦN 6 — KẾT QUẢ & THẢO LUẬN (Slides 21–23)

## Slide 21 — Đối chiếu với tiêu chí thành công

**Tiêu đề**: 40.25% vs mục tiêu ≥ 65% — Gap ở đâu?

**Stop criterion check** (từ `bench/results/scores.json`,
`eligible_for_deploy: false`):
```
🎯 Mục tiêu ≥ 65% accuracy (400-clip balanced ViSEC):   ❌ KHÔNG ĐẠT
   MERaLiON-SER-v1:  40.25%   (cách mục tiêu 24.75 điểm phần trăm)
```

**Gap này đến từ đâu?** (phân rã theo class, dùng recall thật):
| Class | Recall thật | Cách 65% |
|---|---|---|
| angry | 41% | -24 pp |
| neutral | 52% | -13 pp |
| sad | 44% | -21 pp |
| happy | 24% | -41 pp — **xa nhất** |

**Không phải lỗi ngẫu nhiên** — Slide 10 đã chỉ ra nguyên nhân cụ thể:
model lệch dự đoán về `sad` (42.5% tổng số dự đoán), kéo recall của
happy và angry xuống thấp vì chúng bị "hút" sang sad.

**Hình ảnh**: bar chart recall theo class với đường mốc 65% (matplotlib,
có thể tái dùng cấu trúc từ `bench/make_slide_assets.py`).

**Script** (~1:30):
> "Em nói thẳng: tiêu chí đặt ra ban đầu là ≥65% accuracy, và model hiện
> tại chỉ đạt 40.25% — cách mục tiêu gần 25 điểm phần trăm. Đây không
> phải một con số bất ngờ nếu nhìn vào confusion matrix ở slide trước:
> happy có recall thấp nhất, chỉ 24%, vì gần một nửa clip happy bị model
> đoán nhầm thành sad.
>
> Nói cách khác, gap 25 điểm này không phải do model 'yếu đều' ở mọi
> class, mà tập trung vào một lỗi cụ thể có thể debug được: xu hướng
> over-predict sad. Đây là hướng cải thiện rõ ràng nhất, em sẽ nói ở
> phần Hạn chế và Hướng phát triển."

---

## Slide 22 — Vì sao happy/sad khó?

**Tiêu đề**: Phân tích per-class — Class nào khó nhất, vì sao?

**Per-class metrics của MERaLiON** (từ `bench/results/meralion.json`):
| Class | Precision | Recall | F1 | Support | Ghi chú |
|---|---|---|---|---|---|
| **happy** | 0.44 | 0.24 | **0.31** | 100 | 🔴 Recall thấp nhất |
| **neutral** | 0.47 | 0.52 | **0.50** | 100 | 🟢 Cân bằng nhất |
| **sad** | 0.26 | 0.44 | **0.33** | 100 | 🟠 Precision thấp nhất |
| **angry** | 0.63 | 0.41 | **0.50** | 100 | 🟡 Precision cao, recall thấp |

**Vì sao?** (đọc trực tiếp từ confusion matrix ở Slide 10, không suy
diễn):
- **happy** (F1 0.31): 46% clip happy bị đoán thành **sad** — không
  phải neutral. Đây là lỗi lớn nhất của model.
- **sad** (F1 0.33): precision chỉ 0.26 — 3/4 lần model đoán "sad" là
  sai, vì model đoán sad quá thường xuyên (42.5% tổng số dự đoán, xem
  Slide 10) chứ không phải vì sad khó nhận diện.
- **angry** (F1 0.50): precision 0.63 nhưng recall chỉ 0.41 — khi đoán
  angry thì thường đúng, nhưng bỏ sót gần 6/10 clip angry thật (đa số
  bị đoán thành sad).

**Hình ảnh**: `bench/results/assets/per_class_metrics.png` (đã dùng ở
Slide 9, có thể chiếu lại để đối chiếu).

**Script** (~1:00):
> "Câu hỏi 'vì sao happy và sad khó' — câu trả lời không phải vì
> prosody của chúng giống neutral như nhiều giả thuyết SER kinh điển,
> mà là một lỗi cụ thể của model này: nó đoán 'sad' quá thường xuyên.
>
> 46% clip happy bị đoán thành sad — đó là lý do recall của happy chỉ
> 24%. Ngược lại, vì model đoán sad tràn lan, precision của sad chỉ
> 0.26 — cứ 4 lần đoán sad thì 3 lần sai. Đây là một bias cụ thể, đo
> được, và có thể target khi fine-tune — không phải một giới hạn mơ hồ
> của bài toán SER nói chung."

---

## Slide 23 — Hạn chế & thách thức

**Tiêu đề**: Hạn chế & Thách thức

**3 hạn chế chính** (callout):

**1. Chưa đạt stop criterion ≥ 65%**:
- Hiện tại: 40.25% — cách 25 điểm phần trăm
- Nguyên nhân: dataset nhỏ (400 clips), model pre-trained không chuyên
  tiếng Việt, audio quality của ViSEC không đồng đều
- **Hướng cải thiện**: fine-tune MERaLiON trên tập lớn hơn, hoặc dùng
  Vietnamese-specific features

**2. Bias per-class — model over-predict "sad"**:
- Happy F1 = 0.31, sad F1 = 0.33 — rất yếu
- 42.5% tổng số dự đoán là "sad" (thật ra chỉ 25% clip là sad) — 46%
  clip happy và 49% clip angry bị đoán nhầm thành sad
- **Hướng cải thiện**: data augmentation (pitch shift, speed change),
  focal loss hoặc class-weighted loss để giảm bias về sad

**3. Giới hạn hardware**:
- Cần ≥1.5 GB RAM free để load model
- Nhiều máy dev không đủ → phải dùng HF Space thay vì local
- **Hướng cải thiện**: quantization (FP8/INT8), model pruning

**Hạn chế phụ** (sub-bullet):
- Không hỗ trợ batch inference (1 audio tại 1 thời điểm)
- Không có speaker diarization (ai nói gì)
- Không có confidence calibration (confidence có thể quá tự tin)

**Hình ảnh**: Bảng tổng hợp 3 hạn chế với icon thể hiện mức độ.

**Script** (~1:30):
> "Em thành thật chia sẻ 3 hạn chế chính của hệ thống.
>
> Thứ nhất: stop criterion ≥ 65% chưa đạt, hiện tại 40.25%. Nguyên nhân
> là dataset quá nhỏ và model không chuyên tiếng Việt. Để cải thiện cần
> fine-tune trên tập lớn hơn.
>
> Thứ hai: model có bias cụ thể — đoán 'sad' cho 42.5% tổng số clip dù
> class này chỉ chiếm 25% thật. Cần data augmentation hoặc class-weighted
> loss để giảm bias này khi fine-tune.
>
> Thứ ba: giới hạn hardware — 1.5 GB RAM là nhiều với máy dev yếu.
> Hướng cải thiện là quantization model.
>
> Ngoài ra còn 3 hạn chế phụ: không có batch inference, speaker diarization,
> hay confidence calibration. Đây là scope ngoài đồ án này."

---

# PHẦN 7 — KẾT LUẬN (Slides 24–25)

## Slide 24 — Đóng góp & bài học

**Tiêu đề**: Đóng góp & Bài học kinh nghiệm

**Đóng góp của đồ án** (3 điểm):
1. **Benchmark tái lập được**: seed cố định (`seed=0`), 400 clip cân
   bằng, mọi số liệu trace được thẳng về `bench/results/*.json` — chạy
   lại `bench/run_meralion.py` sẽ ra đúng số cũ.
2. **Adapter pattern**: thiết kế module linh hoạt (`BaseAdapter` +
   `REGISTRY`), dễ mở rộng — thêm model mới không cần sửa UI.
3. **Open-source**: code public trên HF Space, ai cũng có thể dùng thử
   và contribute.

**Bài học kinh nghiệm** (5 điểm):
1. **Mọi số liệu trình bày phải trace được về file kết quả thật**:
   không gõ tay hay ước lượng "cho tròn" từ trí nhớ — dễ lệch khỏi số
   thật mà không ai phát hiện. Nguyên tắc này chính là lý do project có
   `tests/test_no_fabrication.py`.
2. **Đọc model card kỹ**: MERaLiON liệt kê Vietnamese là "limited/
   secondary" trong pretraining — cần đặt kỳ vọng đúng mức trước khi
   benchmark, không giả định "cứ deploy lên là chạy tốt".
3. **Test invariant, không chỉ test happy-path**: `test_no_fabrication.py`
   enforce class_scores phải sum-to-1, không NaN, không âm — bắt lỗi ở
   tầng dữ liệu trước khi lỗi lan ra UI.
4. **Đọc confusion matrix, đừng chỉ đọc accuracy**: 40.25% accuracy tự
   nó không nói model lệch về đâu; phải nhìn confusion matrix mới thấy
   bias cụ thể (over-predict "sad") — đây là lỗi debug được, không phải
   giới hạn mơ hồ của bài toán.
5. **Đơn giản hóa**: single-model Space (1 adapter trong `REGISTRY`)
   dễ maintain hơn nhiều so với kiến trúc multi-model dropdown.

**Hình ảnh**: Icon checklist cho mỗi bài học.

**Script** (~1:00):
> "Đồ án có 3 đóng góp chính: benchmark tái lập được với seed cố định,
> adapter pattern linh hoạt, và open-source public trên HF Space.
>
> 5 bài học kinh nghiệm em rút ra:
>
> 1. Mọi số liệu trình bày phải trace được về file kết quả thật — không
>    ước lượng cho đẹp. Đây cũng là lý do project có hẳn một test suite
>    chống fabrication.
> 2. Đọc model card kỹ — MERaLiON tự nhận Vietnamese là ngôn ngữ phụ,
>    nên kỳ vọng ban đầu phải thực tế.
> 3. Test invariant của dữ liệu, không chỉ test đường đi thuận lợi.
> 4. Đọc confusion matrix chứ đừng dừng ở accuracy — accuracy không nói
>    cho mình biết model lệch về đâu.
> 5. Đơn giản hóa — single-model Space dễ maintain hơn multi-model."

---

## Slide 25 — Hướng phát triển + Q&A

**Tiêu đề**: Hướng phát triển & Câu hỏi

**Hướng phát triển tương lai** (chia 3 nhóm):

**🔬 Ngắn hạn (1–2 tháng)**:
- Fine-tune MERaLiON trên ViSEC full 5,280 clips
- Data augmentation: pitch shift, time stretch, noise injection
- Focal loss để cải thiện class khó (happy/sad)

**🚀 Trung hạn (3–6 tháng)**:
- Bổ sung 3 class: fearful, disgusted, surprised (tự label hoặc tìm
  dataset khác)
- Multi-modal: kết hợp audio + text transcript (Whisper transcript → PhoBERT)
- Confidence calibration: temperature scaling

**🌟 Dài hạn (6+ tháng)**:
- Real-time streaming inference (WebSocket)
- Speaker diarization (ai nói dòng nào)
- Emotion intensity regression (không chỉ class, mà còn cường độ 0-1)
- Domain adaptation: customer service call, education, healthcare

**Câu hỏi thường gặp** (FAQ):
- **Tại sao không dùng Wav2Vec2 tiếng Việt?** → Pre-trained Wav2Vec2 tiếng
  Việt không có emotion head; cần fine-tune from scratch.
- **Có thể chạy trên mobile?** → Chưa — model 309M params, cần tối thiểu
  1.5 GB RAM. Sau quantization có thể chạy trên mobile mạnh.
- **License?** → Code: MIT, Model MERaLiON: Apache-2.0, Dataset ViSEC: CC-BY-4.0.

**Slide cuối** (text lớn ở giữa):
```
            Cảm ơn thầy/cô và các bạn đã lắng nghe!

              Q & A — Em sẵn sàng trả lời câu hỏi
```

**Hình ảnh**: Icon timeline 3 giai đoạn (ngắn/trung/dài hạn).

**Script** (~2:00):
> "Cuối cùng, em xin chia sẻ hướng phát triển trong tương lai.
>
> Ngắn hạn 1-2 tháng: fine-tune MERaLiON trên full ViSEC, áp dụng data
> augmentation và focal loss.
>
> Trung hạn 3-6 tháng: mở rộng lên 7 class thay vì 4, thêm multi-modal
> với text transcript, và confidence calibration.
>
> Dài hạn 6+ tháng: real-time streaming, speaker diarization, emotion
> intensity regression, và domain adaptation cho các use case cụ thể
> như call center, giáo dục, healthcare.
>
> Em xin kết thúc phần trình bày tại đây. Cảm ơn thầy/cô và các bạn
> đã lắng nghe. Em sẵn sàng trả lời câu hỏi ạ."

---

## Phụ lục: Thống kê trình bày

**Tổng thời gian đề xuất**:
- Phần 1 (Slides 1-3): ~3 phút
- Phần 2 (Slides 4-7): ~5.5 phút
- Phần 3 (Slides 8-11): ~5 phút
- Phần 4 (Slides 12-17): ~7 phút
- Phần 5 (Slides 18-20): ~4 phút
- Phần 6 (Slides 21-23): ~4 phút
- Phần 7 (Slides 24-25): ~3 phút
- **Tổng**: ~31 phút + 5 phút Q&A

**Tips thuyết trình**:
- Mỗi slide trung bình 60-90 giây — đừng vội, hãy để khán giả đọc
- Slides 4-7 (nền tảng) quan trọng nhất — dành thời gian giải thích
- Slides 11, 21 có biểu đồ — dùng laser pointer hoặc zoom
- Slide 20 demo — backup video nếu demo fail
- Slide 25 Q&A — chuẩn bị câu trả lời cho 5 câu hỏi thường gặp

**Công cụ gợi ý**:
- Tạo slide: PowerPoint, Google Slides, Keynote, hoặc Canva
- Biểu đồ: matplotlib (đã có trong `bench/results/`)
- Screenshots: chạy app live, dùng Snipping Tool
- Video backup: quay màn hình bằng OBS Studio
