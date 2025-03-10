import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import numpy as np
import tensorflow as tf
import train
import threading

try:
    # Загружаем сохраненную модель
    model = tf.keras.models.load_model('Best_points.h5')
except FileNotFoundError:
    train.train_model()
    model = tf.keras.models.load_model('Best_points.h5')


# Функция для предсказания буквы
def predict_letter(image):
    # Преобразуем изображение в формат, который может обработать модель
    # image.save("./img0.png")
    image = image.resize((28, 28)).convert('L').rotate(-90, expand=True).transpose(
        Image.FLIP_LEFT_RIGHT)  # Изменяем размер и переводим в grayscale
    image_array = np.array(image) / 255.0  # Нормализуем
    image_array = image_array.reshape(1, 28, 28, 1)  # Добавляем размерности для модели

    # Получаем предсказание
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class[0]


def recognize_letter():
    # Создаем черное изображение
    image = Image.new('L', (200, 200), 'black')
    draw = ImageDraw.Draw(image)

    # Получаем все объекты с тегом 'line'
    line_ids = canvas.find_withtag('line')
    # print(f"Line IDs: {line_ids}")  # Отладочный вывод

    # Извлекаем координаты всех линий
    for line_id in line_ids:
        coords = canvas.coords(line_id)
        # print(f"Coordinates for line {line_id}: {coords}")  # Отладочный вывод

        # Рисуем линию на изображении
        if coords:  # Проверяем, что координаты не пустые
            draw.line(coords, fill='white', width=20)

    # Сохраняем изображение для проверки
    # image.save("debug_image.png")

    # Предсказываем букву
    letter = predict_letter(image)
    # print(f"Predicted letter: {chr(letter + 96)}")
    result_label.config(text=f"Predicted letter: {chr(letter + 96)}")


# Функция для очистки холста
def clear_canvas():
    canvas.delete('all')


# Создаем окно
window = tk.Tk()
window.title("Draw a Letter")
# Создаем холст для рисования
canvas = tk.Canvas(window, width=200, height=200, bg='black')
canvas.pack()

# Создаем Label для отображения предсказанной буквы
result_label = tk.Label(window, text="Predicted: ", font=("Arial", 16))
result_label.pack()

# Переменные для рисования
drawing = False
last_x, last_y = None, None


# Функция для начала рисования
def start_drawing(event):
    global drawing, last_x, last_y
    drawing = True
    last_x, last_y = event.x, event.y


def draw(event):
    global last_x, last_y
    if drawing:
        # Создаем линию и добавляем её в группу 'line'
        canvas.create_line((last_x, last_y, event.x, event.y), fill='white', width=20, tags='line')
        last_x, last_y = event.x, event.y
        last_x, last_y = event.x, event.y
        recognize_letter()


# Функция для завершения рисования
def stop_drawing(event):
    global drawing
    drawing = False


def check_thread_status(training_thread):
    if training_thread.is_alive():
        # Если поток ещё работает, проверяем снова через 100 мс
        window.after(100, check_thread_status, training_thread)
    else:
        # Включаем кнопку после завершения обучения
        train_button.config(state=tk.NORMAL)

        # Показываем сообщение о конце обучения
        status_label.config(text="Training completed!", font=("Arial", 20))

        # Загружаем график из файла
        plot_image = Image.open("training_plot.png")
        plot_image = plot_image.resize((300, 300))  # Изменяем размер изображения
        plot_photo = ImageTk.PhotoImage(plot_image)

        # Отображаем график в GUI
        plot_label.config(image=plot_photo)
        plot_label.image = plot_photo  # Сохраняем ссылку, чтобы изображение не удалилось сборщиком мусора


def train_model_from_gui():
    # Получаем выбранную функцию активации
    activation = activation_var.get()
    # Получаем количество нейронов из поля ввода
    try:
        hidden_units = int(hidden_units_entry.get())
    except ValueError:
        hidden_units = 128
    print(activation)
    print(hidden_units)
    # Отключаем кнопку, чтобы предотвратить повторное нажатие
    train_button.config(state=tk.DISABLED)

    # Показываем сообщение о начале обучения
    status_label.config(text="Training started...", font=("Arial", 20))

    # Вызываем функцию train_model с выбранными параметрами
    training_thread = threading.Thread(
        target=lambda: train.train_model(activation=activation, hidden_units=hidden_units))
    training_thread.start()

    # Начинаем проверку состояния потока
    window.after(100, check_thread_status, training_thread)


# Привязываем события
canvas.bind('<Button-1>', start_drawing)  # Нажатие левой кнопки мыши
canvas.bind('<B1-Motion>', draw)  # Движение мыши с зажатой левой кнопкой
canvas.bind('<ButtonRelease-1>', stop_drawing)  # Отпускание левой кнопки мыши

# Кнопка для очистки
clear_button = tk.Button(window, text="Clear", command=clear_canvas)
clear_button.pack()

# Создаем выпадающий список для выбора функции активации
activation_var = tk.StringVar(value='relu')  # Значение по умолчанию
activation_label = tk.Label(window, text="Activation Function:")
activation_label.pack()
activation_menu = tk.OptionMenu(window, activation_var, 'relu', 'sigmoid', 'tanh')
activation_menu.pack()

# Создаем поле ввода для количества нейронов
hidden_units_label = tk.Label(window, text="Number of Hidden Units:")
hidden_units_label.pack()
hidden_units_entry = tk.Entry(window)
hidden_units_entry.pack()

# Кнопка для запуска обучения модели
train_button = tk.Button(window, text="Train Model", command=lambda: train_model_from_gui())
train_button.pack()

# Окно статуса
status_label = tk.Label(window, text='Press train model!')
status_label.pack()

# Label для отображения графика
plot_label = tk.Label(window)
plot_label.pack()

# Запускаем окно
window.mainloop()
