import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from utils import load_data, plot_metrics, save_model

def build_mobilenetv2(input_shape, num_classes):
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val), (x_test, y_test), class_names = load_data()
    model = build_mobilenetv2(x_train.shape[1:], len(class_names))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"MobileNetV2 Test Accuracy: {test_acc:.4f}")
    save_model(model, "MobileNetV2")
    plot_metrics(history, "MobileNetV2")
