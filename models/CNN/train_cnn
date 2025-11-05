import tensorflow as tf
from tensorflow.keras import layers, models
from utils import load_data, plot_metrics, save_model

def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val), (x_test, y_test), class_names = load_data()
    model = build_cnn(x_train.shape[1:], len(class_names))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"CNN Test Accuracy: {test_acc:.4f}")
    save_model(model, "CNN")
    plot_metrics(history, "CNN")
