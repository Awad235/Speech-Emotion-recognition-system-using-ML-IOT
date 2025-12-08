import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, Dense, Dropout, 
    BatchNormalization, Input, GaussianNoise, LayerNormalization,
    GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# Load data
print("Loading data...")
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")
emotion_labels = np.load("label_encoder_classes.npy")

print("\nData shapes:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# Data augmentation function
def augment_data(X, y):
    print("\nAugmenting training data...")
    X_aug = []
    y_aug = []
    
    for i in range(len(X)):
        # Original sample
        X_aug.append(X[i])
        y_aug.append(y[i])
        
        # Add noise
        noise = np.random.normal(0, 0.01, X[i].shape)
        X_aug.append(X[i] + noise)
        y_aug.append(y[i])
        
        # Time stretching (fixed version)
        time_steps = X[i].shape[0]
        feature_dim = X[i].shape[1]
        stretched = np.zeros((time_steps, feature_dim))
        
        for feature in range(feature_dim):
            stretched[:, feature] = np.interp(
                np.linspace(0, time_steps-1, time_steps),
                np.arange(time_steps),
                X[i][:, feature]
            )
        
        X_aug.append(stretched)
        y_aug.append(y[i])
        
    return np.array(X_aug), np.array(y_aug)

# Augment training data
X_train, y_train = augment_data(X_train, y_train)

print("\nAfter augmentation:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

# Build model with improvements
model = Sequential([
    # Input layer with noise for regularization
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    GaussianNoise(0.02),
    
    # First Conv block
    Conv1D(32, 7, padding='same', activation='relu', kernel_regularizer=l2(0.01)),
    LayerNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),
    
    # Second Conv block
    Conv1D(64, 5, padding='same', activation='relu', kernel_regularizer=l2(0.01)),
    LayerNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),
    
    # Third Conv block
    Conv1D(128, 3, padding='same', activation='relu', kernel_regularizer=l2(0.01)),
    LayerNormalization(),
    MaxPooling1D(2),
    Dropout(0.3),
    
    # Global pooling to reduce parameters
    GlobalAveragePooling1D(),
    
    # Dense layers
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    LayerNormalization(),
    Dropout(0.5),
    
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    LayerNormalization(),
    Dropout(0.5),
    
    # Output layer
    Dense(y_train.shape[1], activation='softmax')
])

# Compile with label smoothing
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "emotion_model_fixed_v3.keras",
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Train
print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# Plot training history
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.show()

# Evaluate
print("\nEvaluating model...")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Show confusion matrix
print("\nGenerating confusion matrix...")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_true_classes, y_pred_classes)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=emotion_labels))

# Show example predictions
print("\nExample predictions (first 5 test samples):")
test_preds = model.predict(X_test[:5])
for i, pred in enumerate(test_preds):
    emotion = emotion_labels[np.argmax(pred)]
    confidence = pred[np.argmax(pred)] * 100
    true_emotion = emotion_labels[np.argmax(y_test[i])]
    print(f"Sample {i+1}: Predicted {emotion} ({confidence:.1f}% confidence), True: {true_emotion}")
    # Show all probabilities above 10%
    for j, p in enumerate(pred):
        if p > 0.1:
            print(f"  {emotion_labels[j]}: {p*100:.1f}%")