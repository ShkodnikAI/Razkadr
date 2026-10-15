# Артефакты стадий и каталог запуска

Каждый запуск конвейера живёт в собственном каталоге `runs/<run_id>/`. Стадия читает артефакты предыдущих и пишет свои; форматы зафиксированы JSON Schema (draft 2020-12), файлы схем — `schemas/<kind>.v1.json`, копии поставляются с пакетом (`src/razkadr/schemas/`, синхронность проверяет тест). Зафиксировав форматы до реализации стадий, можно разрабатывать и принимать стадии независимо (наряды №5–11).

- Валидация: `razkadr validate --kind <kind> <файл>`; без `--kind` файл проверяется как Storyboard (наряд №002).
- Виды: `storyboard`, `source`, `frames`, `transcript`, `chunks`, `ocr`, `index`, `run`.
- Примеры: `examples/artifacts/` — по одному небольшому валидному примеру на каждый артефакт.
- Каждая схема несёт `schema_version` (semver, см. «Совместимость»).

## Структура каталога запуска

```
runs/<run_id>/
  run.json            # run: версии, конфиг-хэш, стадии, вызовы провайдеров     (kind=run)
  source.json         # source: путь/URL, sha256, длительность, fps, разрешение, дорожки (kind=source)
  normalized.mp4      # нормализованное видео (пишет ingest)
  audio.wav           # аудиодорожка для ASR (пишет ingest)
  frames/
    *.jpg             # выбранные кадры
    ../frames.json    # frames: t, file, scene_id, sharpness, phash, exposure; rejected+reject_reason для отбракованных (kind=frames)
  transcript.json     # transcript: сегменты (t_start, t_end, text, lang, speaker?) + язык (kind=transcript)
  chunks.json         # chunks: тематические чанки (id, t_start, t_end, text, lang, segment_ids — привязка к сегментам транскрипта) (kind=chunks)
  ocr.json            # ocr: текст на кадрах (frame, t, text, lang) (kind=ocr)
  index/
    manifest.json     # index: провайдер, модель, версия, dim, число векторов (kind=index)
    vectors.bin       #payload векторов; путь фиксируется в manifest.vectors_file
  storyboard.json     # storyboard (наряд №002), создаётся отбором (волна 2)
```

`run_id` — имя каталога и поле `run_id` в `run.json` (одинаковые). Пути внутри артефактов (`frames[].file`, `vectors_file`) задаются относительно каталога запуска: каталог можно перемещать целиком.

## Что пишет и читает каждая стадия

| Стадия (наряд) | Читает | Пишет |
|---|---|---|
| Ingest (№005) | исходный файл или URL | `source.json`, `normalized.mp4`, `audio.wav` |
| Кадры (№006) | `normalized.mp4` | `frames/*.jpg`, `frames.json` |
| Транскрипт (№007) | `audio.wav` | `transcript.json` |
| Чанки (№008) | `transcript.json` | `chunks.json` |
| OCR (№009) | `frames.json`, `frames/*.jpg` | `ocr.json` |
| Индекс (№010) | `frames.json`, `chunks.json`, `ocr.json` | `index/manifest.json`, `index/vectors.bin` |
| Отбор (№014, волна 2) | `index/`, `frames.json`, `chunks.json`, `ocr.json` | `storyboard.json` (черновик без текста) |
| Текст (№017–019, волна 3) | `storyboard.json`, `transcript.json` | `storyboard.json` (с текстом) |
| Сборка (№021–028, волна 4) | `storyboard.json`, кадры | PPTX/PDF/HTML/JSON |
| RunContext (№036) | — | `run.json`, кэш вызовов, логи |

## Схема зависимостей стадий

```
Ingest ──→ Кадры ──┬──→ OCR ────┐
   │               │            ├──→ Индекс ──→ Отбор ──→ Текст ──→ Сборка
   ├──→ Транскрипт ┴→ Чанки ────┘           (Storyboard: черновик → с текстом)
   └────────── RunContext ведёт run.json через все стадии
```

Порядок «Кадры/Транскрипт → Чанки → OCR → Индекс» параллелится на независимых ветках (Кадры и Транскрипт не зависят друг от друга). Каждая стрелка = ссылка на артефакт из таблицы выше; стадии не общаются иначе, кроме как через артефакты в каталоге запуска.

## Формат run.json — учёт вызовов провайдеров

`run.json` ведёт RunContext (наряд №036). Запись `provider_calls[]` обязательными полями фиксирует: `stage`, `provider`, `model`, `units[]` (`kind`: `seconds|tokens|images`, `value`), `cost` (`amount`, `currency` ISO 4217, `tariff_date`), `cache_hit` — этого достаточно, чтобы на шлюзе B посчитать стоимость минуты видео и долю попаданий в кэш без доступа к провайдерам. Один вызов может занимать несколько единиц (VLM: `images` + `tokens`). Ключи API в `run.json` и прочих артефактах запрещены (см. `AGENTS.md` и наряд №033).

Стадии в `stages[]`: `name`, `status` (`ok|failed|skipped|cached`), опционально `started_utc`, `duration_s`, `error` (для `failed`). Статус `cached` означает: артефакт взят из предыдущего запуска без повторных платных вызовов.

## Правила совместимости

- Версия артефакта — `schema_version` в самом файле, semver (`1.0` ≡ `1.0.0`); валидатор принимает только свою major-версию и отвергает чужую с понятным сообщением.
- **MINOR** — новые необязательные поля или новые значения enum: старый код читает новый файл; стадии обязаны игнорировать неизвестные поля, а не падать.
- **MAJOR** — несовместимые изменения: новый файл схемы `schemas/<kind>.v2.json`, мигратор, обновление всех стадий, пишущих/читающих вид.
- Семантические проверки сверх схем: уникальность `id` и `t_end >= t_start` для `chunks`/`transcript`/`episodes` Storyboard; время в пределах `duration_s` источника — для Storyboard; major-версия — для всех видов. Всё остальное — уровень схемы, без «свободных» полей (`additionalProperties: false` везде, кроме описанных в схемах исключений).
- Проверки дат (`created_utc`, `started_utc`, `tariff_date`) синтаксические (ISO 8601): календарная валидность (например, 32 января) схемой не отвергается — осознанное упрощение, чтобы не добавлять зависимости от rfc3339-валидатора.
- sha256 в `source.json` — хэш файла `path` (нормализованного видео): это ключ кэша для всех стадий; смена источника меняет хэш и инвалидирует кэш запуска.

## Связь со Storyboard

`storyboard.json` в каталоге запуска подчиняется схеме `storyboard.v1.json` (наряд №002, описание — `docs/storyboard.md`). Storyboard — единственный артефакт, который пользователь правит руками; стадии обязаны сохранять пользовательские правки (`user_edited`).
