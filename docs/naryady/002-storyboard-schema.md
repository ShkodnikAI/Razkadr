# Наряд №002 — JSON Schema Storyboard v1.0

**Статус:** open
**Волна:** 0
**Стадия конвейера:** контракты (все стадии)
**Версия схемы Storyboard:** 1.0
**Связанные ADR:** —
**Зависит от:** №001

## Контракт
```
schemas/storyboard.v1.json + examples/storyboard.valid.json
→ `razkadr validate examples/storyboard.valid.json` печатает "ok", код 0

examples/storyboard.invalid.json (например, нет episodes[0].frame.path)
→ код 1, сообщение с путём ошибки: episodes[0].frame.path: required
```

## Scope
- `schemas/storyboard.v1.json` по структуре из `docs/spec.md` §3.1: `schema_version`, `project`, `sources`, `episodes` (role, source_id, t_start, t_end, frame, alternatives, text, user_edited).
- Команда `razkadr validate`.
- Примеры: валидный, валидный с правками пользователя (`user_edited: true`), невалидный.
- Правила схемы: `t_end >= t_start` (проверка в валидаторе), режимы текста `verbatim|condensed|described`, языки кодами BCP-47.
- Тесты `tests/test_schema.py`.

## Non-goals
- Генерация данных, логика стадий.
- Схемы промежуточных артефактов индексатора (создаются нарядами волны 1).

## Acceptance criteria
- [ ] Примеры из контракта проходят
- [ ] Тесты на валидный, невалидный и граничные случаи (пустые episodes, неизвестный режим текста)
- [ ] CI зелёный
- [ ] Схема описана в `docs/` (поля, правила `user_edited`)
- [ ] PR содержит `Closes #002`

## Assumptions / Open questions
- Версионирование схемы: `schema_version` по semver, обратная совместимость в рамках major.
