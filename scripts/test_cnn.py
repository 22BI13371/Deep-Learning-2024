import keras
import matplotlib.pyplot as plt

mnist_cnn_model = "saved_models/predict_number_cnn_model.keras"
model = keras.saving.load_model(mnist_cnn_model)

model.summary()

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

(train_images, test_images) = (train_images/255.0, test_images/255.0)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)