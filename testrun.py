
from keras._tf_keras.keras.models import load_model
from data_loader import load_data

if __name__ == '__main__':
    base_dir = 'dataset'
    img_size = (224, 224)
    batch_size = 32
    _, _, test_gen = load_data(base_dir, img_size, batch_size)

    model = load_model('car_parts_model.h5')

    loss, accuracy = model.evaluate(test_gen)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
