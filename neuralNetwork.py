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
    
    # Return the full dataframe so we have access to metadata like perturbation_type
    return df

def extract_features_labels(df):
    """"
    Extracts features and labels from the df loaded by load_data
    """
    # Stack the list of arrays into a single numpy array
    # bitboards are shape (13, 8, 8)
    X = np.stack(df['bitboard'].values)
    
    # Labels are binary (1 for reachable, 0 for unreachable)
    y = df['is_reachable'].values.astype(np.float32)
    
    return X, y
    
    # Labels are binary (1 for reachable, 0 for unreachable)
    y = df['is_reachable'].values.astype(np.float32)
    
    return X, y

def resnet_block(inputs, filters, stride=1):
    shortcut = inputs
    
    # First conv
    x = layers.Conv2D(filters, kernel_size=(3, 3), strides=stride, padding="same", data_format="channels_first")(inputs)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.Activation("relu")(x)
    
    # Second conv
    x = layers.Conv2D(filters, kernel_size=(3, 3), strides=1, padding="same", data_format="channels_first")(x)
    x = layers.BatchNormalization(axis=1)(x)
    
    # Shortcut path
    if stride != 1 or inputs.shape[1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=(1, 1), strides=stride, padding="same", data_format="channels_first")(inputs)
        shortcut = layers.BatchNormalization(axis=1)(shortcut)
        
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x

def build_model(input_shape=(13, 8, 8)):
    """
    Builds a Chess-ResNet-20 Convolutional Neural Network.
    Uses data_format='channels_first' to match the format (channels, rank, file).
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial Convolution
    x = layers.Conv2D(16, kernel_size=(3, 3), strides=1, padding="same", data_format="channels_first")(inputs)
    x = layers.BatchNormalization(axis=1)(x)
    x = layers.Activation("relu")(x)
    
    # Stage 1: 3 blocks (6 layers) with 16 filters, 8x8 resolution
    x = resnet_block(x, 16, stride=1)
    x = resnet_block(x, 16, stride=1)
    x = resnet_block(x, 16, stride=1)
    
    # Stage 2: 3 blocks (6 layers) with 32 filters. Downsample to 4x4
    x = resnet_block(x, 32, stride=2)
    x = resnet_block(x, 32, stride=1)
    x = resnet_block(x, 32, stride=1)
    
    # Stage 3: 3 blocks (6 layers) with 64 filters. Downsample to 2x2
    x = resnet_block(x, 64, stride=2)
    x = resnet_block(x, 64, stride=1)
    x = resnet_block(x, 64, stride=1)
    
    # Final Classifier
    x = layers.GlobalAveragePooling2D(data_format="channels_first")(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
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
    train_df = load_data(train_path)
    val_df = load_data(val_path)
    test_df = load_data(test_path)
    
    X_train, y_train = extract_features_labels(train_df)
    X_val, y_val = extract_features_labels(val_df)
    X_test, y_test = extract_features_labels(test_df)
    
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
    
    # 4.5 Plot Training History
    print("\nSaving training history plots...")
    PLOT_DIR = "plots"
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # Plot Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy Progress')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "training_accuracy.png"))
    plt.close()
    
    # Plot Loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Progress')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "training_loss.png"))
    plt.close()
    
    # 5. Evaluate on Test Set
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss    : {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}\n")
    
    # 6. Detailed Metrics
    print("Generating detailed evaluation metrics...")
    y_pred_probs_full = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs_full, axis=-1)
    y_pred_probs = y_pred_probs_full[:, 1]
    
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
    cls_report = classification_report(y_test, y_pred_classes, target_names=["Unreachable (0)", "Reachable (1)"], output_dict=True)
    print(classification_report(y_test, y_pred_classes, target_names=["Unreachable (0)", "Reachable (1)"]))
    
    # Save Classification Report plot
    cls_df = pd.DataFrame(cls_report).transpose()
    # Drop "support" row from plot to avoid messing up the color scale
    cls_df_plot = cls_df.drop('support', axis=1, errors='ignore')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(cls_df_plot, annot=True, cmap="YlGnBu", fmt='.4f')
    plt.title('Classification Report Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "classification_report.png"))
    plt.close()
    print(f"Classification report plot saved to {os.path.join(PLOT_DIR, 'classification_report.png')}")

    print("\n--- Per-Perturbation Analysis ---")
    # Attach predictions back to test dataframe
    test_df['predicted_class'] = y_pred_classes
    test_df['predicted_prob'] = y_pred_probs
    
    # Group by perturbation type 
    for p_type, group in test_df.groupby('perturbation_type'):
        # For 'none', these are our reachable boards
        is_reachable_target = 1 if p_type == 'none' else 0
        target_name = 'Reachable' if is_reachable_target == 1 else f'Unreachable ({p_type})'
        
        # Since 'none' perturbations are all class 1 and others are class 0
        # Creating a standard 2x2 confusion matrix won't work perfectly per-group since there is only 1 true class in each group
        # but we can still show how many were classified as Reachable vs Unreachable
        
        predictions = group['predicted_class'].values
        total = len(predictions)
        pred_reachable = np.sum(predictions == 1)
        pred_unreachable = np.sum(predictions == 0)
        
        accuracy = pred_reachable / total if is_reachable_target == 1 else pred_unreachable / total
        
        print(f"[{p_type}] Count: {total} | Accuracy: {accuracy:.4f}")
        
        # Let's plot what it predicted for this specific perturbation type
        # We'll create a 1x2 bar chart instead of a heatmap since the Y-axis (True Label) is always just 1 singular value
        plt.figure(figsize=(6, 4))
        sns.barplot(x=['Unreachable', 'Reachable'], y=[pred_unreachable, pred_reachable], hue=['Unreachable', 'Reachable'], palette=['red', 'green'], legend=False)
        plt.title(f'Predictions for: {p_type}\n(True Label: {target_name}) | Acc: {accuracy:.2f}')
        plt.ylabel('Count')
        
        safe_ptype = str(p_type).replace(" ", "_").replace("/", "_").lower()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"predictions_{safe_ptype}.png"))
        plt.close()

    
    try:
        roc_auc = roc_auc_score(y_test, y_pred_probs)
        print(f"ROC AUC Score: {roc_auc:.4f}")
    except ValueError:
        print("ROC AUC could not be calculated (needs both classes in test set).")

if __name__ == "__main__":
    main()
