<p align="center"><img src="docs/brand/logo-horizontal.png" alt="Razkadr" width="460"></p>

# Razkadr

**Video in, deck out.** Сервис превращает видео в презентацию с доказательной привязкой к оригиналу: находит качественные кадры по описанию, готовит к каждому короткий текст и собирает слайды (PPTX, PDF, Google Slides) с таймкодами.

Название от «раскадровка»: центральный объект системы — **Storyboard** (JSON), в котором человек может поправить любой кадр и текст до сборки документа.

```
Источник → Индексатор → Отбор → Текст → Раскладка → Рендер/Экспорт
                           ↘        ↓        ↙
                         STORYBOARD (JSON)
```

Первый тестовый сценарий: **видео рецептов** (пресет `recipe`).

## Статус

**CLI-ядро MVP собрано (волны 0–5).** Одна команда `razkadr build` ведёт видео до пакета результатов (PPTX + PDF + HTML + Storyboard JSON) — отобранные кадры, тексты эпизодов, раскладка, рендер и экспорт работают на детерминированных fake-провайдерах без ключей и готовы к реальным API через BYOK. Приёмка — шлюзы владельца (docs/PLAN.md); реальные тарифы и эталонный набор — наряды №004/№040.

## Быстрый старт

Нужны системные зависимости: **ffmpeg** (нормализация видео), **LibreOffice** (`soffice` — экспорт PDF), **poppler-utils** (`pdftoppm` — визуальные тесты). Например, для Debian/Ubuntu:

```bash
sudo apt-get install -y ffmpeg libreoffice poppler-utils
```

Установка и проверка (Python 3.12+):

```bash
git clone https://github.com/ShkodnikAI/Razkadr.git && cd Razkadr
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
razkadr --version
```

Минимальный `razkadr.toml` в корне репозитория (детерминированные fake-провайдеры — работает без ключей и без сети):

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

Сборка деки из видео (пресет `recipe`, язык вывода RU):

```bash
razkadr build video.mp4 --preset recipe --lang ru
```

Команда кладёт пакет в каталог запуска `runs/<run_id>/`:

```
runs/<run_id>/
  deck.pptx          # дека на эталонном шаблоне, со спикерскими заметками
  deck.pdf           # та же дека PDF (нужен LibreOffice)
  deck.html          # самодостаточная HTML-дека с таймкод-ссылками
  storyboard.json    # Storyboard: кадры, тексты, таймкоды — правьте руками и пересобирайте
  slides.json, select.json, text.json, run.json   # промежуточные артефакты и учёт
```

Правку Storyboard можно внести руками (`storyboard.json`) и повторить `razkadr build <видео> --resume <run_id>`: правки пользователя сохраняются, неизменённые стадии не пересчитываются, повторы не тарифицируются повторно (кэш вызовов).

Готовую деку можно выложить в Google Slides с конвертацией: `razkadr export-gslides runs/<run_id>/deck.pptx --out gslinks.json` — нужен OAuth-токен владельца (scope drive.file), механика и конфиг `[google]` — в [docs/render.md](docs/render.md) и [docs/config.md](docs/config.md).

Без ключей пайплайн работает на fake-провайдерах из конфига выше (детерминированные заглушки №033). Для боевых вызовов подключите своих провайдеров через BYOK — реальные секции `[asr]`/`[vlm]`/`[llm]`/`[embeddings]` и `[keys]` описаны в [docs/config.md](docs/config.md) (ключи не логируются и не попадают в артефакты).

## Демо без видео (5 минут)

Нет видео под рукой — сгенерируйте синтетическую фикстуру с известной разметкой (наряд №034) и прогоните полный цикл:

```bash
python tests/fixtures/make_fixtures.py demo-fixture
ffmpeg -y -i demo-fixture/scenes.mp4 \
  -f lavfi -i "sine=frequency=440:sample_rate=44100:duration=30" \
  -shortest -c:v copy -c:a aac -b:a 96k demo-video.mp4
razkadr build demo-video.mp4 --preset recipe --lang ru
```

Подробности и ожидаемый вывод — [docs/demo.md](docs/demo.md); экономика минуты — [docs/cost.md](docs/cost.md).

## Разработка

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Документы

| Файл | Что это |
|---|---|
| [docs/spec.md](docs/spec.md) | Техническое задание v0.5 |
| [docs/PLAN.md](docs/PLAN.md) | План: волны нарядов и шлюзы приёмки |
| [docs/config.md](docs/config.md) | Конфигурация: razkadr.toml, ключи/BYOK, тарифы, кэш |
| [docs/artifacts.md](docs/artifacts.md) | Артефакты стадий, каталог запуска, пакет build |
| [docs/cost.md](docs/cost.md) | Себестоимость минуты: методика и измерения |
| [docs/demo.md](docs/demo.md) | Демо-набор: фикстура, команды, ожидаемый вывод |
| [docs/providers.md](docs/providers.md) | Провайдеры: BYOK, кэш вызовов, учёт стоимости |
| [docs/templates.md](docs/templates.md) | Пользовательские PPTX-шаблоны |
| [docs/adr/](docs/adr/) | Архитектурные решения |
| [docs/limitations.md](docs/limitations.md) | Честные границы |
| [docs/naryady/](docs/naryady/) | Наряды (задачи с контрактом) |
| [PROJECT_SCHEME.md](PROJECT_SCHEME.md) | Схема работы над проектом |

## Репозиторий

https://github.com/ShkodnikAI/Razkadr. Задачи (наряды) оформляются по шаблону `.github/ISSUE_TEMPLATE/naryad.md`, файлы нарядов лежат в `docs/naryady/`.

## Как работаем

Атомарные задачи — **наряды** с контрактом и критериями приёмки; наряды группируются в **волны**, каждая волна заканчивается **шлюзом приёмки** (см. `docs/PLAN.md`). Решения фиксируются в ADR.

## Лицензия

Apache License 2.0, см. [LICENSE](LICENSE). Шрифт Cinzel (в контурах логотипа) распространяется по SIL OFL; см. `docs/brand/README.md`.
