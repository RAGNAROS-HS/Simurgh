import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def load_data(filepath):
    """
    Loads the dataset from a pickle file and prepares it for the neural network.
    Assumes the DataFrame contains 'bitboard' (features) and 'is_reachable' (labels).
    """
    df = pd.read_pickle(filepath)
    
    # Stack the list of arrays into a single numpy array
    # bitboards are shape (13, 8, 8)
    X = np.stack(df['bitboard'].values)
    
    # Labels are binary (1 for reachable, 0 for unreachable)
    y = df['is_reachable'].values.astype(np.float32)
    
    return X, y

def build_model(input_shape=(13, 8, 8)):
    """
    Builds a Convolutional Neural Network for classifying chess board states.
    Uses data_format='channels_first' to match the PyTorch-like (channels, rank, file) format.
    """
    model = keras.Sequential([
        # Input layer mapping to our 13 planes, 8x8 board
        layers.Input(shape=input_shape),
        
        # Convolutional layers
        # Using channels_first since our shape is (13, 8, 8) instead of the default (8, 8, 13)
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same", data_format="channels_first"),
        layers.BatchNormalization(axis=1),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same", data_format="channels_first"),
        layers.BatchNormalization(axis=1),
        
        # Pooling to reduce spatial dimensions
        layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_first"),
        
        # More capacity
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same", data_format="channels_first"),
        layers.BatchNormalization(axis=1),
        layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_first"),
        
        layers.Flatten(),
        
        # Fully connected layers
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        
        # Output layer for binary classification (Reachable / Unreachable)
        layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def main():
    DATASET_DIR = "datasets"
    
    train_path = os.path.join(DATASET_DIR, "train_dataset.pkl")
    val_path = os.path.join(DATASET_DIR, "val_dataset.pkl")
    test_path = os.path.join(DATASET_DIR, "test_dataset.pkl")
    
    # Ensure datasets exist before proceeding
    if not os.path.exists(train_path):
        print(f"Dataset not found at {train_path}. Please run datasetHandler.py first.")
        return

    # 1. Load Data
    print("Loading data...")
    X_train, y_train = load_data(train_path)
    X_val, y_val = load_data(val_path)
    X_test, y_test = load_data(test_path)
    
    print(f"Train shapes: X = {X_train.shape}, y = {y_train.shape}")
    print(f"Val shapes  : X = {X_val.shape}, y = {y_val.shape}")
    print(f"Test shapes : X = {X_test.shape}, y = {y_test.shape}")
    
    # 2. Build Model
    model = build_model()
    model.summary()
    
    # 3. Configure Callbacks
    # Save the best model during training
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath="best_simurgh_model.keras", 
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1
    )
    
    # Stop training early if validation loss stops improving
    early_stopping_cb = keras.callbacks.EarlyStopping(
        patience=5, 
        restore_best_weights=True,
        monitor="val_loss",
        verbose=1
    )
    
    # 4. Train the Model
    print("\nStarting training...")
    batch_size = 64
    epochs = 30
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint_cb, early_stopping_cb],
        verbose=1 # Show progress bar
    )
    
    # 5. Evaluate on Test Set
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss    : {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}\n")
    
    # 6. Detailed Metrics
    print("Generating detailed evaluation metrics...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_classes = (y_pred_probs > 0.5).astype(int)
    
    print("--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred_classes)
    print(cm)
    
    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Unreachable', 'Reachable'], 
                yticklabels=['Unreachable', 'Reachable'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    PLOT_DIR = "plots"
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOT_DIR, "confusion_matrix.png"))
    plt.close()
    print(f"Confusion matrix plot saved to {os.path.join(PLOT_DIR, 'confusion_matrix.png')}")
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred_classes, target_names=["Unreachable (0)", "Reachable (1)"]))
    
    try:
        roc_auc = roc_auc_score(y_test, y_pred_probs)
        print(f"ROC AUC Score: {roc_auc:.4f}")
    except ValueError:
        print("ROC AUC could not be calculated (needs both classes in test set).")

if __name__ == "__main__":
    main()
