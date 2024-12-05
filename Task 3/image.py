import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Extract only the "cat" and "dog" classes (class indices: 3 for cat, 5 for dog)
cat_dog_classes = [3, 5]
train_filter = np.isin(y_train, cat_dog_classes).flatten()
test_filter = np.isin(y_test, cat_dog_classes).flatten()

x_train = x_train[train_filter]
y_train = y_train[train_filter]
x_test = x_test[test_filter]
y_test = y_test[test_filter]

y_train = (y_train == 5).astype(int).flatten()
y_test = (y_test == 5).astype(int).flatten()

# Normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Check the shape of the dataset
print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
print(f"Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}")

# Visualize some samples
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i])
    plt.title("Dog" if y_train[i] == 1 else "Cat")
    plt.axis('off')
plt.show()
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification (0 or 1)
])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc*100:.2f}")
# Plot training and validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()
# Make predictions on the test set
predictions = model.predict(x_test)

# Convert the predicted probabilities to binary labels (0 or 1)
predicted_labels = (predictions > 0.5).astype(int)

# Function to visualize predictions
def display_predictions(images, labels, predictions, num_images=5):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        plt.title("Dog" if predictions[i] == 1 else "Cat")
        plt.axis('off')
    plt.show()

# Display a few test images with their predicted labels
display_predictions(x_test, y_test, predicted_labels, num_images=5)

# Optionally, show some of the incorrect predictions for further analysis
incorrect_indices = np.where(predicted_labels != y_test)[0]
print(f"Total incorrect predictions: {len(incorrect_indices)}")

# Display incorrect predictions
if len(incorrect_indices) > 0:
    incorrect_images = x_test[incorrect_indices[:5]]
    incorrect_true_labels = y_test[incorrect_indices[:5]]
    incorrect_pred_labels = predicted_labels[incorrect_indices[:5]]
    display_predictions(incorrect_images, incorrect_true_labels, incorrect_pred_labels, num_images=5)
