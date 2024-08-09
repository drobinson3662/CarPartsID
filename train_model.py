import tensorflow as tf
from keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from data_loader import load_data
from keras._tf_keras.keras.callbacks import EarlyStopping, LearningRateScheduler


def build_model(input_shape, num_classes):
    # Create a new model
    model = Sequential()

    # Add a convolution layer, 32 filters, filter 3px x 3px in size
    # Max Pooling Layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))

    # Flattens channel value to single value
    model.add(Flatten())

    # Add dense layers
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def lr_schedule(epoch, lr):
    return lr * 0.95


if __name__ == '__main__':
    base_dir = 'dataset'
    img_size = (224, 224)
    batch_size = 32
    train_gen, val_gen, test_gen = load_data(base_dir, img_size, batch_size)

    model = build_model((224, 224, 3), len(train_gen.class_indices))
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)

    history = model.fit(
        train_gen,
        epochs=30,
        validation_data=val_gen,
        callbacks=[early_stopping, lr_scheduler],
    )

    model.save('car_parts_model.h5')