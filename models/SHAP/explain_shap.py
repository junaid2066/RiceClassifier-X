import shap
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils import load_data

if __name__ == "__main__":
    (_, _), (_, _), (x_test, y_test), class_names = load_data()
    model = load_model("models/CNN/cnn_model.h5")
    background = x_test[np.random.choice(x_test.shape[0], 50, replace=False)]
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(x_test[:10])
    shap.image_plot(shap_values, x_test[:10], show=False)
    plt.savefig("results/SHAP/shap_example.png")
