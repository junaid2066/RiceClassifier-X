import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory

def get_datasets(processed_dir='data/processed', image_size=(224,224), batch_size=32):
    train_dir = os.path.join(processed_dir, 'train')
    val_dir = os.path.join(processed_dir, 'val')
    test_dir = os.path.join(processed_dir, 'test')

    train_ds = image_dataset_from_directory(train_dir,
                                            labels='inferred',
                                            label_mode='categorical',
                                            image_size=image_size,
                                            batch_size=batch_size,
                                            shuffle=True)
    val_ds = image_dataset_from_directory(val_dir,
                                          labels='inferred',
                                          label_mode='categorical',
                                          image_size=image_size,
                                          batch_size=batch_size,
                                          shuffle=False)
    test_ds = image_dataset_from_directory(test_dir,
                                           labels='inferred',
                                           label_mode='categorical',
                                           image_size=image_size,
                                           batch_size=batch_size,
                                           shuffle=False)
    class_names = train_ds.class_names
    return train_ds, val_ds, test_ds, class_names

def compile_and_train(model, train_ds, val_ds, epochs=10, save_path=None):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    if save_path:
        model.save(save_path)
    return history

def plot_history(history, out_dir, model_name):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.plot(history.history.get('accuracy', []), label='train_acc')
    plt.plot(history.history.get('val_accuracy', []), label='val_acc')
    plt.title(f'{model_name} accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(os.path.join(out_dir, f'{model_name}_accuracy.png'))
    plt.close()

    plt.figure()
    plt.plot(history.history.get('loss', []), label='train_loss')
    plt.plot(history.history.get('val_loss', []), label='val_loss')
    plt.title(f'{model_name} loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(os.path.join(out_dir, f'{model_name}_loss.png'))
    plt.close()
