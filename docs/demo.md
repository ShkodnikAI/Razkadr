# Демо-набор: полный цикл за 5 минут

Демо не требует ни видео, ни ключей: синтетическая фикстура наряда №034 генерирует видео с известной разметкой, а пайплайн работает на детерминированных fake-провайдерах №033. Все команды ниже выполнены дословно из корня репозитория (Python 3.12, чистый venv, `pip install -e .`); вывод — реальный прогон, без сокращений.

## 1. Системные зависимости

```bash
sudo apt-get install -y ffmpeg libreoffice poppler-utils
```

## 2. Установка и минимальный конфиг

```bash
git clone https://github.com/ShkodnikAI/Razkadr.git && cd Razkadr
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Минимальный `razkadr.toml` (fake-провайдеры; для боевых провайдеров см. `docs/config.md`):

```toml
runs_dir = "runs"

[asr]
provider = "fake"
model = "fake-nova-2"

[vlm]
provider = "fake"
model = "fake-vision-1"

[llm]
provider = "fake"
model = "fake-gpt-4o-mini"

[embeddings]
provider = "fake"
model = "fake-embed-v2"
dim = 8
```

## 3. Синтетическое видео с аудио

```bash
python tests/fixtures/make_fixtures.py demo-fixture
ffmpeg -y -i demo-fixture/scenes.mp4 \
  -f lavfi -i "sine=frequency=440:sample_rate=44100:duration=30" \
  -shortest -c:v copy -c:a aac -b:a 96k demo-video.mp4
```

## 4. Сборка деки

```bash
razkadr build demo-video.mp4 --preset recipe --lang ru
```

Ожидаемый вывод (реальный прогон 2026-10-17):

```
run 20261017T100403Z_c07d79: ingest ok, frames ok, transcript ok, chunks ok, ocr ok, index ok, select ok, text ok, layout ok, render ok, html ok, pdf ok
  video:      320x240 @ 10 fps, 30 s
  storyboard: 4 episodes (hero 1, ingredients 1, prep 2), 4 slides, 1 warning(s)
  package:    runs/20261017T100403Z_c07d79
    pptx  runs/20261017T100403Z_c07d79/deck.pptx
    pdf   runs/20261017T100403Z_c07d79/deck.pdf
    html  runs/20261017T100403Z_c07d79/deck.html
    json  runs/20261017T100403Z_c07d79/storyboard.json
  cost:       0 USD (31 calls, 0 cache hits)
  time:       ingest 1.026s, frames 0.558s, transcript 0.003s, chunks 0.002s, ocr 3.003s, index 2.003s, layout 0.002s, render 0.057s, html 0.001s, pdf 0.784s
  dir:        runs/20261017T100403Z_c07d79
```

Что посмотреть:

- `runs/<run_id>/deck.pptx` (и `deck.pdf`) — дека на эталонном шаблоне: титул, hero с подписью, ингредиенты, шаги, подача;
- `runs/<run_id>/deck.html` — откройте в браузере: таймкоды кликабельны;
- `runs/<run_id>/storyboard.json` — Storyboard: эпизоды с текстом и таймкодами.

## 5. Что ожидать и чего не ждать

- Предупреждение `role 'steps': 0 episode(s) selected, preset minimum is 3` — норма на 30-секундной фикстуре: ролей `steps` нужно минимум 3 с шагом 15 с, синтетике их взять неоткуда. На реальном рецепте распределение будет полным.
- Тексты эпизодов — детерминированные заглушки («fake description …»): fake-провайдеры служат для механики, не для качества; боевые тексты дают реальные провайдеры по BYOK (`docs/config.md`).
- `--resume <run_id>` после правки `storyboard.json` пересобирает деку, сохраняя правки и не пересчитывая неизменное; повторные вызовы бесплатны (кэш вызовов).

## 6. Проверка артефактов схемами

```bash
razkadr validate runs/<run_id>/storyboard.json
razkadr validate --kind slides runs/<run_id>/slides.json
razkadr validate --kind run runs/<run_id>/run.json
```

Каждая команда печатает `ok`. Полный каталог запуска и формат каждого артефакта — `docs/artifacts.md`.
