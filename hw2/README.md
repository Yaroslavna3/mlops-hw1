# HW2: Gitflow

Домашнее задание посвящено работе с ветками, Pull Request и релизными тегами.

## Схема ветвления

| Ветка | Назначение |
| --- | --- |
| `main` | Стабильные релизы |
| `develop` | Интеграция изменений |
| `feature/*` | Новая функциональность |
| `release/*` | Подготовка релиза |
| `hotfix/*` | Срочные исправления |

## Выполненная работа

1. Создана ветка `develop` от `main`.
2. В ветке `feature/add-median` добавлена функция `median` и тесты.
3. Feature-ветка слита в `develop` через Pull Request.
4. В ветке `release/0.2.0` подготовлен релиз `v0.2.0`.
5. В ветке `hotfix/0.2.1` улучшена обработка пустого ввода.
6. Hotfix выпущен с тегом `v0.2.1`.
7. Изменения из `main` синхронизированы с `develop`.

## Pull Requests

- `feature/add-median` → `develop`
- `release/0.2.0` → `main`
- `hotfix/0.2.1` → `main`
- `main` → `develop`

## Проверка

```powershell
git branch -a
git tag -n
git log --graph --oneline --decorate --all
```
