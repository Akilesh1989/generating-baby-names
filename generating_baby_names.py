import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import LambdaCallback


def preprocess(names):
    print("Preprocessing names")
    unique_names = list(set(names))
    names = []
    for name in unique_names:
        if name.endswith("."):
            names.append(name.lower())
        else:
            name = name + "."
            names.append(name.lower())

    unique_characters = []
    for name in names:
        for character in name:
            if character not in unique_characters:
                unique_characters.append(character)

    unique_characters = sorted(unique_characters)

    # Convert from character to index
    char_to_index = dict(
        (unique_characters[i], i) for i in range(0, len(unique_characters))
    )

    # Convert from index to character
    index_to_char = dict(
        (i, unique_characters[i]) for i in range(0, len(unique_characters))
    )

    # maximum number of characters in Pokémon names
    # this will be the number of time steps in the RNN
    max_char = len(max(names, key=len))

    # number of elements in the list of names, this is the number of training examples
    m = len(names)

    # number of potential characters, this is the length of the input of each of the RNN units
    char_dim = len(char_to_index)
    X = np.zeros((m, max_char, char_dim))
    Y = np.zeros((m, max_char, char_dim))

    for i in range(m):
        name = list(names[i])
        for j in range(len(name)):
            X[i, j, char_to_index[name[j]]] = 1
            if j < len(name) - 1:
                Y[i, j, char_to_index[name[j + 1]]] = 1

    return max_char, char_dim, index_to_char, char_to_index, X, Y


def build_model(max_char, char_dim):
    print("Building model")
    model = Sequential(
        [
            LSTM(128, input_shape=(max_char, char_dim), return_sequences=True),
            # LSTM(128, input_shape=(max_char, char_dim), return_sequences=True),
            Dense(char_dim, activation="softmax"),
        ]
    )

    model.compile(loss="categorical_crossentropy", optimizer="adam")
    return model


def make_name(model):
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
    # print("".join(name))
    return name


def train_model(model, X, Y, epochs):
    print("Training model")
    def generate_name_loop(epoch, _):
        if epoch % 25 == 0:
            print("Names generated after epoch %d:" % epoch)
            for i in range(5):
                make_name(model)

    name_generator = LambdaCallback(on_epoch_end=generate_name_loop)
    ## Training the model
    model.fit(X, Y, batch_size=64, epochs=epochs, verbose=0)
    return model


def make_name_with_starting_letter(model, first_letter, second_letter, max_char, char_dim, index_to_char, char_to_index):
    name = []
    x = np.zeros((1, max_char, char_dim))
    if first_letter:
        x[0, 0, char_to_index[first_letter]] = 1
        first_character = index_to_char[char_to_index[first_letter]]
        name.append(first_character)
    if second_letter:
        x[0, 1, char_to_index[second_letter]] = 1
        second_character = index_to_char[char_to_index[second_letter]]
        name.append(second_character)
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


def main(names, first_letter, second_letter, minimum_name_length, num_names, epochs):
    max_char, char_dim, index_to_char, char_to_index, X, Y = preprocess(names)
    model = build_model(max_char, char_dim)
    model = train_model(model, X, Y, epochs)

    generated_names = []
    name_count = 0
    while True:
        name = make_name_with_starting_letter(
            model, first_letter, second_letter, max_char, char_dim, index_to_char, char_to_index
        )
        if len(name) < minimum_name_length:
            continue
        name = "".join(name)
        name = name + "."
        if name not in names:
            generated_names.append(name)
        len_names = len(generated_names)
        if name_count != len_names:
            name = name.replace(".", "")
            # print(f"{len_names}. {name}")
        if len(generated_names) < num_names:
            name_count = len_names
            continue
        else:
            break

    return generated_names


if __name__ == "__main__":
    names = [
        "Mason",
        "Logan",
        "Alexander",
        "Ethan",
        "Jacob",
        "Michael",
        "Daniel",
        "Henry",
        "Jackson",
        "Sebastian",
        "Aiden",
        "Matthew",
        "Samuel",
        "David",
        "Joseph",
        "Carter",
        "Owen",
        "Wyatt",
        "John",
        "Jack",
        "Luke",
        "Jayden",
        "Dylan",
        "Grayson",
        "Levi",
        "Issac",
        "Gabriel",
        "Julian",
        "Mateo",
        "Anthony",
        "Jaxon",
        "Lincoln",
        "Joshua",
        "Christopher",
        "Andrew",
        "Theodore",
        "Caleb",
        "Ryan",
        "Asher",
        "Nathan",
        "Thomas",
        "Leo",
        "Isaiah",
        "Charles",
        "Josiah",
        "Hudson",
        "Christian",
        "Hunter",
        "Connor",
        "Eli",
        "Ezra",
        "Aaron",
        "Landon",
        "Adrian",
        "Jonathan",
        "Nolan",
        "Jeremiah",
        "Easton",
        "Elias",
        "Colton",
        "Cameron",
        "Carson",
        "Robert",
        "Angel",
        "Maverick",
        "Nicholas",
        "Dominic",
        "Jaxson",
        "Greyson",
        "Adam",
        "Ian",
        "Austin",
        "Santiago",
        "Jordan",
        "Cooper",
        "Brayden",
        "Roman",
        "Evan",
        "Ezekiel",
        "Xavier",
        "Jose",
        "Jace",
        "Jameson",
        "Leonardo",
        "Bryson",
        "Axel",
        "Everett",
        "Parker",
        "Kayden",
        "Miles",
        "Sawyer",
        "Jason",
        "Sam",
        "Chris",
    ]

    # The characters should be enclosed in single quotes.
    # If you do not want to assign a first or second letter, replace 'a' or 'k' with None
    first_letter = None
    second_letter = None
    minimum_name_length = 4
    num_names = (
        10  # change this to the desired number of names you want to see in the output.
    )
    epochs = 1000

    generated_names = main(names, first_letter, second_letter, minimum_name_length, num_names, epochs)
    for name in generated_names:
        print(name.replace(".", ""))
