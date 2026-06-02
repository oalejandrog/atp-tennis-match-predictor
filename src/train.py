import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

EPOCHS = 30
BATCH_SIZE = 32

def load_processed_data(path='data/processed/tennis_data.npz'):
    data = np.load(path)
    return (data['X_train'], data['y_train']), (data['X_val'], data['y_val']), (data['X_test'], data['y_test'])

def build_shallow_mlp(input_shape):
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

def build_dropout_mlp(input_shape):
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

def build_batch_norm_mlp(input_shape):
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

def train_and_evaluate():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_processed_data()
    input_shape = (X_train.shape[1],)
    
    model_configs = [
        ('Shallow_MLP', build_shallow_mlp),
        ('Dropout_MLP', build_dropout_mlp),
        ('BatchNorm_MLP', build_batch_norm_mlp)
    ]
    
    histories = {}
    test_results = {}
    
    os.makedirs('results', exist_ok=True)
    
    for name, builder in model_configs:
        print(f"Training {name}...")
        model = builder(input_shape)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0
        )
        
        histories[name] = history.history
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        test_results[name] = accuracy
        print(f"{name} Test Accuracy: {accuracy:.4f}")
        
    save_training_plots(histories)
    save_results_summary(test_results)

def save_training_plots(histories):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for name, hist in histories.items():
        plt.plot(hist['val_accuracy'], label=name)
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for name, hist in histories.items():
        plt.plot(hist['val_loss'], label=name)
    plt.title('Validation Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png')

if __name__ == "__main__":
    train_and_evaluate()
for name, acc in results.items():
            f.write(f"{name}: {acc:.4f}\n")

if __name__ == "__main__":
    train_and_evaluate()
