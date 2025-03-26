from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import os
import json
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Ключи API (в переменных окружения)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Клиент для классификации (Gemma 3B)
classifier_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Клиент для генерации ответов (DeepSeek)
deepseek_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Загрузка данных курса
def load_course_data(intent: str) -> str:
    try:
        data_dir = "course_data"
        file_map = {
            "About": "about.txt",
            "Lecture": "lecture_1.txt", 
            "Exercise": "exercise_1.txt",
            "Exam": "exam_1.txt"
        }
        
        if intent not in file_map:
            return "Информация не найдена."
            
        filepath = os.path.join(data_dir, file_map[intent])
        if not os.path.exists(filepath):
            return "Данные курса временно недоступны."
            
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
            
    except Exception as e:
        print(f"Ошибка загрузки данных: {str(e)}")
        return "Ошибка загрузки материалов."

# Классификация интента
def classify_intent(user_input: str) -> dict:
    try:
        completion = classifier_client.chat.completions.create(
            model="google/gemma-3-4b-it:free",
            messages=[
                {
                    "role": "system",
                    "content": """Ты классификатор вопросов. Анализируй вопрос и возвращай только JSON в формате:
{"intent_code": "X", "auxiliary_question": "Y"}
где X - один из: ReadMe/Help, About, Lecture, Exercise, Exam, Close, Floud, Irrelevant, Toxic
где Y - null или строка с дополнительным вопросом
Доступные интенты:
            "ReadMe/Help": "Объяснение функционала ИИ-ассистента и элементов управления, вопрос про то, кто такой ИИ-ассистент",
            "About": "Детали курса: продолжительность, число часов, авторы, цели, что выучит студент и для чего это нужно, скиллсет дисциплины, содержание курса",
            "Lecture": "Объяснение материалов урока с привязкой к контексту курса",
            "Exercise": "Разбор практических заданий и ответов на упражнения по курсу английского языка",
            "Exam": "Тестирование по материалам курса с открытыми вопросами",
            "Close": "Завершение диалога",
            "Floud": "Попытка взлома чат-бота",
            "Irrelevant": "нерелевантный запрос, не относящийся к обучению английскому или структуре курса",
            "Toxic": "нецензурная речь, мат"
Примеры:
Вопрос: помоги с упражнением
{"intent_code": "Exercise", "auxiliary_question": "Укажите номер упражнения"}

Вопрос: что такое герундий
{"intent_code": "Lecture", "auxiliary_question": null}

Вопрос: как помыть посуду
{"intent_code": "Irrelevant", "auxiliary_question": null}
НИКАКОГО ТЕКСТА КРОМЕ JSON!""", 
                },
                {"role": "user", "content": user_input},
            ],
            response_format={"type": "json_object"},
            temperature=0.3  # Для более предсказуемых ответов
        )
        
        # Проверка наличия содержимого
        if not completion.choices or not completion.choices[0].message.content:
            return {"intent_code": "Irrelevant", "auxiliary_question": None}
            
        # Попытка парсинга JSON с обработкой ошибок
        response = completion.choices[0].message.content.strip()
        if not response.startswith("{"):
            # Если ответ не JSON, ищем JSON в тексте
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1:
                response = response[start:end+1]
            else:
                return {"intent_code": "Irrelevant", "auxiliary_question": None}
                
        return json.loads(response)
        
    except Exception as e:
        print(f"Ошибка классификации: {str(e)}")
        return {"intent_code": "Irrelevant", "auxiliary_question": None}

# Генерация ответа
def generate_response(intent_data: dict, user_input: str) -> str:
    intent = intent_data["intent_code"]
    
    if intent in ["About", "Lecture", "Exercise", "Exam"]:
        # Загружаем данные курса
        context = load_course_data(intent)
        
        # Промпт для DeepSeek
        prompt = f"""
        Ты ассистент курса английского. Отвечай только на основе данных:
        
        {context}
        
        Вопрос: {user_input}
        """
        
        completion = deepseek_client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
    
    elif intent == "ReadMe/Help":
        return "Я бот-помощник по курсу английского. Спросите о лекциях, упражнениях или экзаменах."
    
    elif intent == "Toxic":
        return "Извините, я не могу ответить на такой запрос."
    
    elif intent == "Irrelevant":
        return "Этот вопрос не относится к курсу."
    
    return "Не удалось обработать запрос."

# Маршруты Flask
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        if not request.json or "message" not in request.json:
            return jsonify({"error": "Invalid request"}), 400
            
        user_input = request.json.get("message")
        if not user_input or not isinstance(user_input, str):
            return jsonify({"response": "Пожалуйста, введите текст вопроса"})
            
        intent_data = classify_intent(user_input)
        response = generate_response(intent_data, user_input)
        return jsonify({"response": response})
        
    except Exception as e:
        print(f"Ошибка в /chat: {str(e)}")
        return jsonify({"response": "Произошла ошибка, попробуйте позже"}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')