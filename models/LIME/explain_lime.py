import lime
import lime.lime_image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils import load_data

if __name__ == "__main__":
    (_, _), (_, _), (x_test, y_test), class_names = load_data()
    model = load_model("models/CNN/cnn_model.h5")
    explainer = lime.lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(x_test[0].astype('double'), model.predict, top_labels=1, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    plt.imshow(temp)
    plt.title("LIME Explanation")
    plt.savefig("results/LIME/lime_example.png")
