import csv
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import LambdaCallback


# class callback(Callback):
#     def __init__(self, model, index_to_char, char_dim):
#         self.model = model
#         self.index_to_char = index_to_char
#         self.char_dim = char_dim

#     def


def read_names(file_name):
    print("Reading names")
    with open(file_name, newline='') as csv_file:
        reader = csv.reader(csv_file)
        return [name_list[0] for name_list in list(reader)]


def get_unique_char(unique_names):
    print("Get unique characters")
    unique_characters = []
    for name in unique_names:
        for character in name:
            if character not in unique_characters:
                unique_characters.append(character)
    return sorted(unique_characters)


def convert_char_to_index(unique_characters):
    # Convert from character to index
    return dict((unique_characters[i], i) for i in range(0, len(unique_characters)))


def convert_index_to_char(unique_characters):
    # Convert from index to character
    return dict((i, unique_characters[i]) for i in range(0, len(unique_characters)))


def get_max_char(unique_names):
    # this will be the number of time steps in the RNN
    
    return len(max(unique_names, key=len))


def get_num_training_examples(names):
    # number of elements in the list of names, this is the number of training examples
    return len(names)


def get_char_dim(char_to_index):
    return len(char_to_index)


def get_inputs(m, max_char, char_dim, char_to_index, unique_names):
    X = np.zeros((m, max_char, char_dim))
    Y = np.zeros((m, max_char, char_dim))

    for i in range(m):
        name = list(unique_names[i])
        for j in range(len(name)):
            X[i, j, char_to_index[name[j]]] = 1
            if j < len(name) - 1:
                Y[i, j, char_to_index[name[j + 1]]] = 1

    return X, Y


def define_and_compile_model(max_char, char_dim):
    # Define and compile model
    model = Sequential(
        [
            LSTM(128, input_shape=(max_char, char_dim), return_sequences=True),
            # LSTM(128, input_shape=(max_char, char_dim), return_sequences=True),
            Dense(char_dim, activation="softmax"),
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
    return model


def make_name(model, max_char, char_dim, index_to_char):
    name = []
    x = np.zeros((1, max_char, char_dim))
    end = False
    i = 0

    while end == False:
        predicted = model.predict(x)
        predicted_sliced = predicted[0, i]
        probs = list(predicted_sliced)
        probs = probs / np.sum(probs)
        index = np.random.choice(range(char_dim), p=probs)
        if i == max_char - 2:
            character = "."
            end = True
        else:
            character = index_to_char[index]
        name.append(character)
        x[0, i + 1, index] = 1
        i += 1
        if character == ".":
            end = True

    name = [char for char in name if char != "."]
    return name


def generate_name_loop(epoch, _):
    if epoch % 25 == 0:
        print("Names generated after epoch %d:" % epoch)
        for i in range(5):
            make_name(model, index_to_char, char_dim)


def train_model(model, X, Y, with_callback=False):
    if with_callback:
        name_generator = LambdaCallback(on_epoch_end=generate_name_loop)
        model.fit(
            X, Y, batch_size=64, epochs=400, callbacks=[name_generator], verbose=0
        )
    else:
        model.fit(X, Y, batch_size=64, epochs=5, verbose=1)
    return model


def generate_names(model, max_char, index_to_char, char_dim, unique_names):
    generated_names = []
    for i in range(0, 10):
        name = make_name(model, max_char, char_dim, index_to_char)
        name = "".join(name)
        generated_names.append(name)
        if name not in unique_names:
            generated_names.append(name)
    return list(set(generated_names))


def main():
    file_name = "names.csv"
    unique_names = read_names(file_name)

    unique_characters = get_unique_char(unique_names)
    
    char_to_index = convert_char_to_index(unique_characters)
    index_to_char = convert_index_to_char(unique_characters)
    max_char = get_max_char(unique_names)
    print(max_char)
    num_training_examples = get_num_training_examples(unique_names)
    char_dim = get_char_dim(char_to_index)
    X, Y = get_inputs(
        num_training_examples, max_char, char_dim, char_to_index, unique_names
    )
    model = define_and_compile_model(max_char, char_dim)
    model = train_model(model, X, Y, with_callback=False)
    generated_names = generate_names(
        model, max_char, index_to_char, char_dim, unique_names
    )
    print(generated_names)


if __name__ == "__main__":
    main()
