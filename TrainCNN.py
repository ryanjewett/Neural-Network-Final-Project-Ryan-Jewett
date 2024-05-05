import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def main():
    data = np.load(r"C:\Users\ryanj\Documents\CPE_4093\imageData2.npz")

    X_train = data['arr_0']
    X_test = data['arr_1']
    Y_train = data['arr_2']
    Y_test = data['arr_3']

    X_train = X_train/255
    X_test = X_test/255

    print(f'shape of X_train {X_train.shape}')
    print(f'shape of X_test {X_test.shape}')
    print(f'shape of Y_train {Y_train.shape}')
    print(f'shape of Y_test {Y_test.shape}')

    print(X_train)
    print(Y_train)

    model_small = Sequential()

    model_small.add(Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model_small.add(MaxPooling2D((2, 2)))
    model_small.add(Conv2D(32, (3,3), activation='relu'))
    model_small.add(MaxPooling2D((2, 2)))
    model_small.add(Flatten())
    model_small.add(Dense(32, activation='relu'))
    model_small.add(Dropout(0.5))
    model_small.add(Dense(10, activation='softmax'))

    model_small.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print(model_small.summary())

    history = model_small.fit(X_train, Y_train, epochs=5, batch_size=32, validation_data=(X_test, Y_test), shuffle=True)

    # Evaluate the model
    test_loss, test_accuracy = model_small.evaluate(X_test, Y_test)
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_accuracy}')

    model_small.save('model.h5')

if __name__ == "__main__":
    main()
