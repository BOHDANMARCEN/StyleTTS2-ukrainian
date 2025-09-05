# UA-TTS-Subdub

Україномовний офлайн TTS-конвеєр на базі **StyleTTS2** з підтримкою **вербалізації**, **наголосів** і **повної озвучки субтитрів SRT** з точним дотриманням таймінгів (з опційним time‑stretch).

---

## 📦 Можливості
- 🗣️ **Текст → Мова** (single/multi‑speaker StyleTTS2).
- 🔤 **Вербалізація** чисел, дат, скорочень (офлайн mBART).
- ◌́ **Автоматичні наголоси** (акцентатор) + перетворення до **IPA**.
- 🎬 **Озвучка SRT-субтитрів**: кожна репліка синтезується та **підганяється у час** під `start → end` (time‑stretch).
- 📁 Збереження результату у `outputs/… .wav`.
- 🌐 Готове для локального запуску й **Hugging Face Spaces**.

---

## 🗂️ Структура проєкту
```
UA-TTS-Subdub/
├─ app.py                         # Головний Gradio застосунок
├─ verbalizer.py                  # Обгортка над mBART вербалізатором
├─ ipa_uk.py                      # Перетворення українського тексту у IPA (адаптивний модуль)
├─ utils/
│  ├─ audio.py                    # Нормалізація, запис, тиша, time-stretch
│  └─ text.py                     # Розбиття тексту, чистка субтитрів
├─ scripts/
│  └─ download_models.py          # (Опційно) Завантаження моделей у локальні папки
├─ requirements.txt               # Залежності Python
├─ README.md                      # Цей файл (скорочена версія)
├─ .gitignore
├─ voices/                        # Тут *.pt стилі для multi-speaker моделі
│  └─ README.md
├─ outputs/                       # Збережені результати (.wav)
└─ models/
   ├─ models--verbalizer/         # Локальна папка вербалізатора (HF snapshot)
   ├─ models--styletts2-single/   # Локальна папка single‑speaker моделі
   └─ models--styletts2-multi/    # Локальна папка multi‑speaker моделі
```

> **Примітка:** назви підпапок у `models/` вільні — головне узгодити їх у `app.py`.

---

## 🔧 Встановлення

1) **Python 3.10–3.11**, `ffmpeg` (для деяких бібліотек аудіо — не обовʼязково, але бажано)

2) Встановіть залежності:
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

3) **Моделі (офлайн):**
- Підготуйте локальні папки з моделями (будь-яким способом):
  - Вербалізатор mBART → `models/models--verbalizer`
  - StyleTTS2 (single) → `models/models--styletts2-single`
  - StyleTTS2 (multi) → `models/models--styletts2-multi`
- (Опційно) скористайтесь скриптом:
```bash
python scripts/download_models.py
```

4) **Голоси (multi‑speaker):**
- Покладіть стилі `.pt` у `voices/` (див. `voices/README.md`).

---

## 🚀 Запуск
```bash
python app.py
```
Відкрийте інтерфейс: `http://127.0.0.1:7860`.

### Режими
- **Текст → Мова**: введіть текст, оберіть модель (single/multi), голос і швидкість.
- **Озвучка субтитрів**: завантажте `.srt` → отримаєте **повний .wav** з дотриманням таймінгів; файл також збережеться в `outputs/`.

---

## 🧠 Логіка конвеєра
1. **Вербалізація**: розкриття чисел/дат/скорочень → чистий текст.
2. **Акцентатор**: розстановка наголосів, нормалізація Unicode, уніфікація тире.
3. **G2P/IPA**: перетворення наголошеного тексту у фонеми.
4. **StyleTTS2**: токенізація фонем → генерація звуку (24 кГц).
5. **Subdub (SRT)**: для кожної репліки генерується аудіо і **time‑stretch** під `duration = end − start`; між репліками додається тиша або враховується перетин.

---

## ⚙️ Налаштування шляхів моделей у `app.py`
У верхній частині `app.py` відкоригуйте константи:
```python
VERBALIZER_DIR = 'models/models--verbalizer'
STYLETTS2_SINGLE_DIR = 'models/models--styletts2-single'
STYLETTS2_MULTI_DIR = 'models/models--styletts2-multi'
PROMPTS_DIR = 'voices'
SAMPLE_RATE = 24000
```

---

## 📁 voices/README.md
```
# Голоси для multi-speaker

Покладіть сюди файли стилів *.pt, назва файлу = назва голосу.
Напр.:  
voices/
  ├─ filatov.pt
  ├─ mariya.pt
  └─ narrator.pt

Якщо у вас один голос (single), ця папка може бути порожньою.
```

---

## 🧩 requirements.txt
```
accelerate>=0.33.0
transformers>=4.42.0
huggingface_hub>=0.24.0
torch>=2.1.0
numpy>=1.26.0
gradio>=4.44.0
spaces>=0.29.0
soundfile>=0.12.1
librosa>=0.10.1
srt>=3.5.3
unicodedata2>=15.1.0
ukrainian-word-stress>=1.5.0
# Бібліотека інференсу StyleTTS2 (опублікуйте власний пакет або локальний модуль)
styletts2-inference @ git+https://github.com/patriotyk/styletts2_ukrainian_inference.git
```
> За потреби замініть останній рядок на реальний шлях/репозиторій вашого інференс‑пакета.

---

## 📘 Код файлів

### `verbalizer.py`
```python
import torch
from transformers import AutoModelForSeq2SeqLM, MBartTokenizer

class Verbalizer:
    """Обгортка навколо mBART-вербалізатора для офлайн-розгортання."""
    def __init__(self, model_path: str, device: str = 'cpu'):
        try:
            self.tokenizer = MBartTokenizer.from_pretrained(
                model_path,
                local_files_only=True
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                local_files_only=True
            )
        except Exception as e:
            print(f"[Verbalizer] Помилка завантаження моделі: {e}")
            raise
        self.to(device)

    def to(self, device: str):
        self.device = device
        self.model.to(self.device)

    @torch.inference_mode()
    def generate_text(self, text: str, **gen_kwargs) -> str:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, **gen_kwargs)
        decoded = [self.tokenizer.decode(t, skip_special_tokens=True) for t in out]
        return decoded[0] if decoded else ""
```

### `ipa_uk.py`
```python
"""
Адаптивний перетворювач українського тексту у IPA.
Пріоритети:
1) Якщо доступна стороння бібліотека (напр., uk-g2p / власний модуль), використовуємо її.
2) Інакше — спрощений трансліт у набір фонем, сумісний із токенізатором моделі.

У продакшені замініть цей файл на ваш точний G2P.
"""
import re

# Стрес-символ (комбінуючий гострий наголос)
STRESS = "\u0301"

try:
    # Приклад: ваш точний модуль
    from uk_g2p import g2p as _precise_g2p  # type: ignore
except Exception:  # noqa: E722
    _precise_g2p = None

_basic_map = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'ɦ', 'ґ': 'g', 'д': 'd', 'е': 'e', 'є': 'je',
    'ж': 'ʒ', 'з': 'z', 'и': 'ɪ', 'і': 'i', 'ї': 'ji', 'й': 'j', 'к': 'k', 'л': 'l',
    'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
    'ф': 'f', 'х': 'x', 'ц': 'ts', 'ч': 'tʃ', 'ш': 'ʃ', 'щ': 'ʃtʃ', 'ь': '', "'": '',
    'ю': 'ju', 'я': 'ja',
}

_vowel = set("аеєиієїоуюяAEЄИІЇЇОУЮЯ")


def _simple_ipa(text: str) -> str:
    t = text.lower()
    # прибираємо тегові артефакти
    t = re.sub(r"<[^>]+>", "", t)
    out = []
    for ch in t:
        out.append(_basic_map.get(ch, ch))
    ipa = " ".join(out)
    # зберігаємо позицію наголосу як окремий символ у стрімі
    ipa = ipa.replace(STRESS, STRESS)
    return ipa


def ipa(text: str) -> str:
    if _precise_g2p is not None:
        try:
            return _precise_g2p(text)
        except Exception:
            pass
    return _simple_ipa(text)
```

### `utils/audio.py`
```python
from __future__ import annotations
import os
from datetime import datetime
import numpy as np
import soundfile as sf
import librosa
import torch


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_wav(np_audio: np.ndarray, sr: int, out_dir: str, base: str) -> str:
    ensure_dir(out_dir)
    ts = datetime.now().strftime('%Y%m%d__%H%M%S')
    path = os.path.join(out_dir, f"{base}__{ts}.wav")
    sf.write(path, np_audio, sr)
    return path


def make_silence(seconds: float, sr: int) -> torch.Tensor:
    n = max(0, int(round(seconds * sr)))
    return torch.zeros(n, dtype=torch.float32)


def normalize_audio(wav: torch.Tensor, peak: float = 0.95) -> torch.Tensor:
    if wav.numel() == 0:
        return wav
    m = wav.abs().max().item()
    if m == 0:
        return wav
    scale = min(1.0, peak / m)
    return wav * scale


def time_stretch_to_duration(np_audio: np.ndarray, sr: int, target_sec: float, clamp: tuple[float, float] = (0.6, 1.6)) -> np.ndarray:
    """
    Підганяє довжину сигналу під target_sec за допомогою phase-vocoder (librosa.effects.time_stretch).
    rate > 1 → швидше (коротше); output_len = input_len / rate.
    """
    if target_sec <= 0:
        return np_audio
    in_sec = len(np_audio) / sr
    if in_sec == 0:
        return np_audio
    rate = max(clamp[0], min(clamp[1], in_sec / target_sec))
    if abs(rate - 1.0) < 1e-3:
        return np_audio
    y = librosa.effects.time_stretch(np_audio.astype('float32'), rate=rate)
    # Вирівнювання до точної довжини
    out_len = int(round(target_sec * sr))
    if len(y) < out_len:
        pad = out_len - len(y)
        y = np.pad(y, (0, pad))
    elif len(y) > out_len:
        y = y[:out_len]
    return y.astype('float32')
```

### `utils/text.py`
```python
import re
from unicodedata import normalize
from ukrainian_word_stress import Stressifier, StressSymbol

stressifier = Stressifier()

SPLIT_SYMBOLS = '.?!:'

def split_to_parts(text: str, max_len: int = 150) -> list[str]:
    parts = ['']
    idx = 0
    for s in text:
        parts[idx] += s
        if s in SPLIT_SYMBOLS and len(parts[idx]) > max_len:
            idx += 1
            parts.append('')
    return [p for p in parts if p.strip()]


def clean_and_stress(text: str) -> str:
    """Нормалізація + заміна '+' на наголос + уніфікація тире + stressifier."""
    t = (text or '').strip()
    t = t.replace('+', StressSymbol.CombiningAcuteAccent)
    t = normalize('NFKC', t)
    t = re.sub(r'[᠆‐‑‒–—―⁻₋−⸺⸻]', '-', t)
    t = re.sub(r' - ', ': ', t)
    return stressifier(t)


def clean_sub_text(raw: str) -> str:
    # прибираємо HTML/ASS теги, перенос рядків, курсиви тощо
    t = re.sub(r'<[^>]+>', '', raw)
    t = re.sub(r'\{\\[^}]+\}', '', t)  # {\i1} і подібні
    t = t.replace('\n', ' ').replace('\r', ' ')
    t = re.sub(r'\s+', ' ', t)
    return t.strip()
```

### `scripts/download_models.py`
```python
"""
Зразковий скрипт завантаження моделей у локальні папки через huggingface_hub.
Замініть MODEL_ID_* на ваші реальні репозиторії.
"""
from huggingface_hub import snapshot_download

# !!! ЗАМІНІТЬ на ваші реальні ID репозиторіїв у HF !!!
MODEL_ID_VERBALIZER = "ina-speech-research/Ukrainian-Verbalizer-g2p-v1"  # приклад
MODEL_ID_STYLETTS2_SINGLE = "patriotyk/styletts2_ukrainian_single"         # приклад
MODEL_ID_STYLETTS2_MULTI = "patriotyk/styletts2_ukrainian_multispeaker"    # приклад

print("Завантажуємо вербалізатор…")
snapshot_download(repo_id=MODEL_ID_VERBALIZER, local_dir="models/models--verbalizer")

print("Завантажуємо StyleTTS2 single…")
snapshot_download(repo_id=MODEL_ID_STYLETTS2_SINGLE, local_dir="models/models--styletts2-single")

print("Завантажуємо StyleTTS2 multi…")
snapshot_download(repo_id=MODEL_ID_STYLETTS2_MULTI, local_dir="models/models--styletts2-multi")

print("Готово.")
```

### `.gitignore`
```
.venv/
__pycache__/
*.pt
*.wav
outputs/
models/
*.ipynb_checkpoints
.DS_Store
```

### `voices/README.md`
```markdown
# Голоси для multi-speaker StyleTTS2

Покладіть сюди стилі голосів у форматі `.pt` (тензори ембедінгів стилю).

- Імʼя файлу = імʼя голосу (відображатиметься у UI).
- Для single-speaker режиму ця папка не потрібна.
```

---

## 🖥️ `app.py` (повний)
```python
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import glob
import re
import srt
import numpy as np
from datetime import datetime

# Gradio / Spaces
import gradio as gr
try:
    import spaces
    gpu_decorator = spaces.GPU
except Exception:  # якщо не у HF Spaces – no-op декоратор
    def gpu_decorator(fn):
        return fn

import torch
from unicodedata import normalize

from styletts2_inference.models import StyleTTS2
from ukrainian_word_stress import Stressifier, StressSymbol

from verbalizer import Verbalizer
from ipa_uk import ipa
from utils.text import split_to_parts, clean_and_stress, clean_sub_text
from utils.audio import (
    ensure_dir, save_wav, make_silence, normalize_audio, time_stretch_to_duration
)

# ================== КОНФІГ ==================
VERBALIZER_DIR = 'models/models--verbalizer'
STYLETTS2_SINGLE_DIR = 'models/models--styletts2-single'
STYLETTS2_MULTI_DIR = 'models/models--styletts2-multi'
PROMPTS_DIR = 'voices'
OUTPUTS_DIR = 'outputs'
SAMPLE_RATE = 24000

stressify = Stressifier()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Вербалізатор (офлайн)
verbalizer = Verbalizer(model_path=VERBALIZER_DIR, device=device)

# ======== Завантаження моделей StyleTTS2 ========
# single
single_model = StyleTTS2(hf_path=STYLETTS2_SINGLE_DIR, device=device)
# стиль single – окремий .pt або взятий із моделі; для простоти встановимо None
single_style = None

# multi
multi_model = StyleTTS2(hf_path=STYLETTS2_MULTI_DIR, device=device)

# стилі голосів (pt) із папки voices/
multi_styles = {}
prompts_list = sorted(glob.glob(os.path.join(PROMPTS_DIR, '*.pt')))
prompts_list = [os.path.splitext(os.path.basename(p))[0] for p in prompts_list]
for name in prompts_list:
    path = os.path.join(PROMPTS_DIR, f'{name}.pt')
    try:
        multi_styles[name] = torch.load(path, map_location=device)
        print('[voices] loaded', name)
    except Exception as e:
        print('[voices] fail', name, e)

models = {
    'multi': { 'model': multi_model, 'styles': multi_styles },
    'single': { 'model': single_model, 'style': single_style }
}

# ================== ФУНКЦІОНАЛ ==================
@gpu_decorator
@torch.inference_mode()
def verbalize_ui(text: str) -> str:
    parts = split_to_parts(text)
    out = []
    for p in parts:
        if p.strip():
            out.append(verbalizer.generate_text(p))
    return ' '.join(out)


@gpu_decorator
@torch.inference_mode()
def synthesize(model_name: str, text: str, speed: float, voice_name: str | None = None, progress=gr.Progress()):
    if not text or text.strip() == "":
        raise gr.Error("Введіть текст")
    if len(text) > 50000:
        raise gr.Error("Текст має бути < 50k символів")

    result_wav = []
    model_data = models[model_name]

    for t in progress.tqdm(split_to_parts(text)):
        t = t.strip().replace('"', '')
        if not t:
            continue
        t = clean_and_stress(t)
        t = verbalizer.generate_text(t)
        ps = ipa(t)

        tokens = model_data['model'].tokenizer.encode(ps)
        style = None
        if model_name == 'multi' and voice_name:
            style = model_data['styles'].get(voice_name)
        elif model_name == 'single':
            style = model_data.get('style')

        wav = model_data['model'](tokens, speed=speed, s_prev=style)
        wav = normalize_audio(wav)
        result_wav.append(wav)

    if not result_wav:
        raise gr.Error("Порожній результат")

    final_audio = torch.cat(result_wav).cpu().numpy().astype('float32')
    # автозбереження
    fname = save_wav(final_audio, SAMPLE_RATE, OUTPUTS_DIR, base='tts')
    return (SAMPLE_RATE, final_audio), fname


@gpu_decorator
@torch.inference_mode()
def synthesize_subtitles(sub_file, model_name: str, speed: float, voice_name: str | None = None,
                         strict_timing: bool = True, elasticity: float = 0.35,
                         progress=gr.Progress()):
    """
    Озвучка SRT. Для кожної репліки:
      1) вербалізація → наголос → IPA
      2) синтез StyleTTS2
      3) time-stretch під тривалість (за потреби)
      4) тиша між репліками відповідно до таймінгів
    Параметр elasticity визначає межі time-stretch: rate ∈ [1−el, 1+el].
    """
    if sub_file is None:
        raise gr.Error("Завантажте .srt файл")

    with open(sub_file.name, 'r', encoding='utf-8') as f:
        subs = list(srt.parse(f.read()))

    if not subs:
        raise gr.Error("Порожні субтитри")

    model_data = models[model_name]
    cur_t = 0.0
    chunks: list[torch.Tensor] = []

    low = max(0.2, 1.0 - elasticity)
    high = 1.0 + elasticity

    for sub in progress.tqdm(subs):
        start_sec = sub.start.total_seconds()
        end_sec = sub.end.total_seconds()
        duration = max(0.0, end_sec - start_sec)

        # тиша до початку, якщо є розрив
        if start_sec > cur_t:
            chunks.append(make_silence(start_sec - cur_t, SAMPLE_RATE))
            cur_t = start_sec

        # підготовка тексту
        raw = clean_sub_text(sub.content)
        if not raw:
            continue
        t = clean_and_stress(raw)
        t = verbalizer.generate_text(t)
        ps = ipa(t)

        # синтез
        tokens = model_data['model'].tokenizer.encode(ps)
        style = None
        if model_name == 'multi' and voice_name:
            style = model_data['styles'].get(voice_name)
        elif model_name == 'single':
            style = model_data.get('style')

        wav = model_data['model'](tokens, speed=speed, s_prev=style)
        wav = normalize_audio(wav)
        np_wav = wav.cpu().numpy().astype('float32')

        # підгін у час
        if strict_timing and duration > 0:
            np_wav = time_stretch_to_duration(np_wav, SAMPLE_RATE, duration, clamp=(low, high))
        # повертаємо до torch для уніфікації
        wav = torch.from_numpy(np_wav)
        chunks.append(wav)

        cur_t = end_sec

    if not chunks:
        raise gr.Error("Немає аудіо для збирання")

    final = torch.cat(chunks).cpu().numpy().astype('float32')

    base = os.path.splitext(os.path.basename(sub_file.name))[0]
    fname = save_wav(final, SAMPLE_RATE, OUTPUTS_DIR, base=f'subdub__{base}')

    return (SAMPLE_RATE, final), fname


# ================== UI ==================
with gr.Blocks(title="UA-TTS-Subdub", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🇺🇦 UA‑TTS‑Subdub — StyleTTS2 + Вербалізація + SRT озвучка")

    with gr.Tab("Текст → Мова"):
        with gr.Row():
            model_select = gr.Dropdown(choices=["multi", "single"], value="multi", label="Модель")
            voice_select = gr.Dropdown(choices=sorted(list(multi_styles.keys())), value=None, label="Голос (multi)")
            speed_slider = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Швидкість")
        text_in = gr.Textbox(lines=6, label="Текст українською")
        with gr.Row():
            btn_verbal = gr.Button("Вербалізувати")
            btn_synth = gr.Button("Синтезувати")
        audio_out = gr.Audio(label="Результат", type="numpy")
        file_out = gr.File(label="Завантажити .wav")

        btn_verbal.click(fn=verbalize_ui, inputs=[text_in], outputs=[text_in])
        btn_synth.click(fn=synthesize,
                        inputs=[model_select, text_in, speed_slider, voice_select],
                        outputs=[audio_out, file_out])

    with gr.Tab("Озвучка субтитрів"):
        sub_file = gr.File(label="Завантажте .srt", file_types=['.srt'])
        with gr.Row():
            model_select2 = gr.Dropdown(choices=["multi", "single"], value="multi", label="Модель")
            voice_select2 = gr.Dropdown(choices=sorted(list(multi_styles.keys())), value=None, label="Голос (multi)")
            speed_slider2 = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Швидкість")
        with gr.Row():
            strict = gr.Checkbox(value=True, label="Строго підганяти під таймінги (time‑stretch)")
            elasticity = gr.Slider(0.2, 0.6, value=0.35, step=0.05, label="Еластичність time‑stretch (±)")
        btn_sub = gr.Button("Згенерувати озвучку субтитрів")
        audio_out2 = gr.Audio(label="Результат", type="numpy")
        file_out2 = gr.File(label="Завантажити .wav")

        btn_sub.click(fn=synthesize_subtitles,
                      inputs=[sub_file, model_select2, speed_slider2, voice_select2, strict, elasticity],
                      outputs=[audio_out2, file_out2])

if __name__ == "__main__":
    ensure_dir(OUTPUTS_DIR)
    demo.queue(api_open=True, max_size=15).launch(show_api=True)
```

---

## 📒 README.md (скорочена копія)
```markdown
# UA-TTS-Subdub

Україномовний офлайн TTS-конвеєр на базі StyleTTS2 із вербалізацією, наголосами та повною озвучкою SRT з дотриманням таймінгів (time‑stretch).

## Швидкий старт
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_models.py   # (опційно) завантажить моделі у models/
python app.py
```
Відкрийте http://127.0.0.1:7860

## Структура
- `app.py` — Gradio UI, TTS і SRT-озвучка
- `verbalizer.py` — mBART вербалізатор (офлайн)
- `ipa_uk.py` — адаптивний G2P у IPA
- `utils/` — аудіо/текст утиліти
- `voices/` — стилі .pt для multi-speaker
- `models/` — локальні снапшоти моделей
- `outputs/` — готові .wav

## Моделі
Вкажіть локальні шляхи у верхівці `app.py` або скористайтесь `scripts/download_models.py` й замініть `MODEL_ID_*` на ваші реальні HF‑репозиторії.

## Озвучка субтитрів
- Завантажте `.srt` у вкладці «Озвучка субтитрів».
- Система синтезує кожну репліку та піджене її в час під таймінги (опційно можна послабити жорсткість через «Еластичність»).
- Пауза між репліками створюється автоматично.
- Результат зберігається в `outputs/` і доступний для завантаження.

## Ліцензія моделей
Переконайтесь, що ліцензії моделей дозволяють локальне використання/перерозповсюдження.
```

---

## 🧪 Примітки та поради
- Якщо токенізатор/модель StyleTTS2 очікує інший формат фонем — замініть `ipa_uk.py` на ваш точний G2P.
- Якщо працюєте **не у HF Spaces**, імпорт `spaces` автоматично стає no‑op.
- Для **ідеальної синхронізації** з відеомонтажем рекомендуємо експортувати `.wav`, а далі зводити у відеоредакторі чи через `ffmpeg`.
- Якщо ваші субтитри мають перекриття таймінгів, система просто не вставлятиме тишу; міксування перекритих реплік тут не виконується (можна додати при потребі).

---

## ✅ Готово!
Це повний кістяк продакшн‑рівня для україномовного озвучування тексту та субтитрів із можливістю офлайн‑розгортання. Замініть/уточніть лише реальні шляхи до моделей і G2P за вашими вимогами.

