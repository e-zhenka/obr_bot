# Прототип "Чат-бот для Курса Английского Языка"

Это Flask-приложение представляет собой прототип ассистента для помощи в изучении английского языка. Бот использует модели LLM для классификации запросов и генерации ответов на основе материалов курса.

## Функциональность

- Классификация пользовательских запросов с помощью Gemma 3B
- Генерация ответов с использованием DeepSeek Chat
- Работа с различными типами контента:
  - Лекционные материалы
  - Практические упражнения
  - Экзаменационные вопросы
  - Общая информация о курсе

## Технологический стек

- Python 3.9
- Flask 3.0.2
- OpenAI API (через OpenRouter)
- Docker

## Установка и запуск

## Настройка окружения
1.  Клонируйте репозиторий
2.  Скопируйте файл .env_example в .env: ```cp .env_example .env```
3.  Отредактируйте .env файл, добавив свой OPENROUTER_API_KEY

### Локальный запуск

2. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # для Linux/Mac
venv\Scripts\activate     # для Windows
```
3. Установите зависимости:
```bash
pip install -r requirements.txt
```
4. Создайте файл .env и добавьте API ключ:

### Запуск через Docker

1. Убедитесь, что у вас установлены Docker и Docker Compose
2. Создайте файл .env с API ключом OPENROUTER_API_KEY=
3. Запустите контейнер:
```bash
docker-compose up --build
```
