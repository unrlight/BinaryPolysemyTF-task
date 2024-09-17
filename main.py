import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Подготовка данных
data = pd.read_excel('data.xlsx')
data = data.dropna()
x_train = data[['x1', 'x2', 'x3', 'x4', 'x5']].values
y_train = data['F'].values

# Настройки для экспериментов
neuron_counts = [3, 4, 5, 32, 1024]
activations = ['tanh', 'sigmoid', 'relu']
epochs = 1500

# Вывод весов
def print_weights(model, output_weights_data):
    weights = model.layers[0].get_weights() # Получаем веса 1 спрятанного слоя
    if weights:
        output_weights_data += ("### Веса 1 спрятанного слоя")
        output_weights_data += (f"\nВеса: \n{weights[0]}")
        output_weights_data += (f"\nСмещения: \n{weights[1]}\n\n")
        text_file = open("Output.txt", "a", encoding="utf-8")
        text_file.write(output_weights_data)
        text_file.close()

    weights = model.layers[1].get_weights() # Получаем веса выходного слоя
    if weights:
        output_weights_data += ("### Веса выходного слоя")
        output_weights_data += (f"\nВеса: \n{weights[0]}")
        output_weights_data += (f"\nСмещения: \n{weights[1]}\n\n")
        text_file = open("Output.txt", "a", encoding="utf-8")
        text_file.write(output_weights_data)
        text_file.close()

# Функция для создания и обучения модели
def create_and_train_model(neurons, activation, epochs):
    output_weights_data = f"\n###### Запуск модели с {neurons} нейронами и функцией активации {activation}"
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(5,)),
        tf.keras.layers.Dense(neurons, activation=activation),
        tf.keras.layers.Dense(1, activation=activation)
    ])
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(x_train, y_train, epochs=epochs, verbose=0)
    print_weights(model, output_weights_data)
    return history.history['loss']

# Построение графика потерь
plt.figure(figsize=(12, 8))
colors = ['r', 'g', 'b', 'm', 'c']

for i, neurons in enumerate(neuron_counts):
    for activation in activations:
        if activation == 'tanh':
            linestyle = '--'
        if activation == 'sigmoid':
            linestyle = '-'
        if activation == 'relu':
            linestyle = 'dotted'
        loss = create_and_train_model(neurons, activation, epochs)
        plt.plot(loss, color=colors[i], linestyle=linestyle, label=f'{activation}, {neurons} нейрона')

plt.ylabel('Loss')
plt.xlabel('Эпохи')
plt.legend()
plt.show()
