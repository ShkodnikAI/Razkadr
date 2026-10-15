# Наряд №002 — JSON Schema Storyboard v1.0 и команда validate

**Волна:** 0 (Фундамент)  
**Стадия:** contracts  
**Размер:** M  
**Исполнитель:** agent  
**Зависит от:** #1  
**Связанные ADR:** —  
**Статус:** open

## Контекст
Storyboard — центральный объект системы (см. `docs/spec.md` §3.1). Его схема должна быть зафиксирована до написания стадий: все стадии читают и пишут этот формат.

## Контракт
```
examples/storyboard.valid.json
→ `razkadr validate examples/storyboard.valid.json`  печатает "ok", код выхода 0

examples/storyboard.invalid.json   (нет episodes[0].frame.path)
→ код выхода 1, в stderr: "episodes[0].frame.path: required"
```

## Scope
- `schemas/storyboard.v1.json` по структуре из `docs/spec.md` §3.1: `schema_version`, `project`, `sources`, `episodes` (role, source_id, t_start, t_end, frame, alternatives, text, user_edited).
- Семантические проверки сверх схемы: уникальность id, `source_id` ссылается на существующий источник, `t_end >= t_start`, время в пределах `duration_s` источника, если оно задано.
- Команда `razkadr validate <файл>` с понятными сообщениями (JSON-путь и причина).
- Три примера в `examples/`: валидный, валидный с правками пользователя (`user_edited: true`), невалидный.
- Описание схемы в `docs/storyboard.md`: поля, режимы текста `verbatim|condensed|described`, язык кодами BCP-47, правила `user_edited`, правила версионирования (`schema_version` по semver, совместимость в пределах major).

## Non-goals
- Генерация данных и логика стадий.
- Схемы промежуточных артефактов индексатора (наряд №35).

## Acceptance criteria
- [ ] Пример из контракта проходит (оба случая)
- [ ] Тесты: валидный, невалидный, граничные случаи (пустой `episodes`, неизвестный режим текста, `t_end < t_start`, неизвестный `source_id`, дубликат id)
- [ ] Сообщения об ошибках содержат JSON-путь
- [ ] `docs/storyboard.md` описывает все поля
- [ ] CI зелёный

## Assumptions / Open questions
- Валидатор на `jsonschema` (уже в dev-зависимостях); перенос в основные зависимости допустим и должен быть отмечен в PR.

## Проверка аудитора
- Прогнать оба примера руками и проверить текст ошибки.
- Проверить, что семантические проверки покрыты тестами, а не только схема.
- Убедиться, что схема не допускает неизвестных полей на верхнем уровне без осознанного решения.

---
Агент: прочитай `AGENTS.md`. PR содержит `Closes #2`, раздел Assumptions и Unresolved; числа только из реальных запусков.
