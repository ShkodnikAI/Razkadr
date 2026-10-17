# Замеры провайдеров (наряд 004)

Скрипты измеряют задержку, списанное использование (usage) и стоимость в USD для
кандидатов ADR-0006: ASR, эмбеддинги, VLM (описания/OCR кадров), LLM (сжатие/перевод).
Тарифы — публичные цены из `tariffs.json` (источники и дата проверки внутри).
Все ключи берутся строго из переменных окружения, в вывод ключи маскируются.

## Переменные окружения

| Провайдер | Переменная |
|---|---|
| Gemini API (transcribe / flash-lite / flash / embedding) | `GEMINI_API_KEY` |
| Deepgram | `DEEPGRAM_API_KEY` |
| ElevenLabs | `ELEVENLABS_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| Cohere | `COHERE_API_KEY` |
| Jina AI | `JINA_API_KEY` |

## Проверка готовности без сети

```bash
.venv/bin/python scripts/providers/measure.py --dry-run
```

## Команды замеров

Подставьте свои пути. `--repeat 3` даёт min/median/max задержки; стоимость считается
по usage каждого вызова. Результаты пишутся в `--out` (по умолчанию
`runs/providers/results.json`, каталог `runs/` не коммитится).

### ASR — 1 минута видео с речью (+ опциональный эталонный текст для WER)

```bash
.venv/bin/python scripts/providers/measure.py --task asr \
  --provider gemini-transcribe --provider elevenlabs-scribe \
  --provider deepgram-nova3 --provider openai-transcribe \
  --audio /path/to/one-minute-speech.mp4 \
  --reference-text "$(cat /path/to/reference.txt)" \
  --repeat 3 --out runs/providers/asr.json
```

### Эмбеддинги — 20 кадров + 20 текстов (размерности для ADR-0002)

```bash
.venv/bin/python scripts/providers/measure.py --task embed \
  --provider openai-text-embedding-3-small \
  --provider cohere-embed-v4 --provider jina-embeddings-v4 \
  $(printf -- '--image %s ' /path/to/frames/*.jpg | head -c -1) \
  --text "Доведите масло до кипения" --text "Bake for 25 minutes" \
  --repeat 3 --out runs/providers/embed.json
```

### VLM — описания/OCR кадров (проверка ожидаемого текста на кадре)

```bash
.venv/bin/python scripts/providers/measure.py --task vlm \
  --provider gemini-flashlite --provider openai-luna \
  --image /path/to/frame-01.png --image /path/to/frame-02.png \
  --expected-text "FLOUR 500 G" \
  --repeat 3 --out runs/providers/vlm.json
```

### LLM — сжатие/перевод (тестовые фразы или фрагменты транскрипта)

```bash
.venv/bin/python scripts/providers/measure.py --task llm \
  --provider openai-luna --provider gemini-flashlite --provider gemini-flash \
  --text "Доведите до кипения, помешивая, около двух минут." \
  --repeat 3 --out runs/providers/llm.json
```

## Что дальше с результатами

1. Заполнить таблицу замеров в `docs/providers.md` (колонки из вывода `results.json`).
2. Обновить ADR-0006 фактическими числами и принять/скорректировать выбор.
3. Провести live-проверку выбранных адаптеров (наряды 037–039).
