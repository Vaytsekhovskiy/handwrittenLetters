import kagglehub
import pandas as pd  # для обработки данных
import numpy as np  # для линейной алгебры
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt  # для создания графиков
import seaborn as sns

path = kagglehub.dataset_download("crawford/emnist")

testing_letter = pd.read_csv(path + '/emnist-letters-test.csv')
training_letter = pd.read_csv(path + '/emnist-letters-train.csv')

y1 = np.array(training_letter.iloc[:, 0].values)  # Берём первый столбец всех строк (метка класса)
x1 = np.array(training_letter.iloc[:, 1:].values)  # Все строки, все столбцы кроме первого столбца
# testing_letter
y2 = np.array(testing_letter.iloc[:, 0].values)
x2 = np.array(testing_letter.iloc[:, 1:].values)

# Приводим данные к диапазону [0,1]
train_images = x1 / 255.0
test_images = x2 / 255.0

train_images_number = train_images.shape[0]  # кол-во изображений в обучающей выборке
train_images_height = 28
train_images_width = 28
train_images_size = train_images_height * train_images_width

train_images = train_images.reshape(train_images_number, train_images_height, train_images_width, 1)
# кол-во изображений, высота, ширина, 1 = кол-во каналов (1 т.к. чёрно белое)

test_images_number = test_images.shape[0]
test_images_height = 28
test_images_width = 28
test_images_size = test_images_height * test_images_width

test_images = test_images.reshape(test_images_number, test_images_height, test_images_width, 1)

# Количество классов (26 букв английского алфавита + спец символы)
number_of_classes = 37

y1 = tf.keras.utils.to_categorical(y1, number_of_classes)
y2 = tf.keras.utils.to_categorical(y2, number_of_classes)

train_x, test_x, train_y, test_y = train_test_split(train_images, y1, test_size=0.2, random_state=42)


def train_model(activation='relu', hidden_units=128):
    model = tf.keras.Sequential([
        # tf.keras.layers.Conv2D(32, 3, input_shape=(28, 28, 1)),
        # tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(512, activation=activation),
        tf.keras.layers.Dense(hidden_units, activation=activation),
        tf.keras.layers.Dense(number_of_classes, activation='softmax')
    ])

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    MCP = ModelCheckpoint('Best_points.h5', verbose=1, save_best_only=True, monitor='val_accuracy', mode='max')
    ES = EarlyStopping(monitor='val_accuracy', min_delta=0, verbose=0, restore_best_weights=True, patience=3,
                       mode='max')
    RLP = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, min_lr=0.0001)

    # объект, который содержит информацию о процессе обучения (например, значения потерь и точности на каждой эпохе).
    history = model.fit(train_x, train_y, epochs=10, validation_data=(test_x, test_y), callbacks=[MCP, ES, RLP])

    # Получаем количество эпох, на которых обучалась модель
    # history.history['accuracy'] содержит значения точности на обучающих данных для каждой эпохи
    q = len(history.history['accuracy'])

    # Устанавливаем размер графика (10x10 дюймов)
    plt.figsize = (10, 10)

    sns.lineplot(x=range(1, 1 + q), y=history.history['accuracy'], label='Accuracy')
    sns.lineplot(x=range(1, 1 + q), y=history.history['val_accuracy'], label='Val_Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('Accuray')
    plt.legend()

    # Сохраняем график в файл
    plot_path = "training_plot.png"
    plt.savefig(plot_path)
    plt.close()  # Закрываем график, чтобы освободить память

    return plot_path  # Возвращаем путь к файлу с графиком
