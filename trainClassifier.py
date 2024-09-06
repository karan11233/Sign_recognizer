import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data from the pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Extract data and labels
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Data Augmentation Function
def augment_landmarks(landmarks):
    noise = np.random.normal(0, 0.01, landmarks.shape)
    return landmarks + noise

# Apply augmentation to the entire dataset
augmented_data = np.array([augment_landmarks(d) for d in data])

# Reshape the data to be compatible with CNNs
data = np.array([np.expand_dims(np.array(d), axis=-1) for d in augmented_data])

# Encode the labels to integers
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

# Convert labels to one-hot encoding
labels = to_categorical(labels)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 1), activation='relu', input_shape=(42, 1, 1)),  # Adjusted kernel size to (3, 1)
    MaxPooling2D((2, 1)),

    Conv2D(64, (3, 1), activation='relu'),
    MaxPooling2D((2, 1)),

    Conv2D(128, (3, 1), activation='relu'),  # Add another layer for complexity

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), callbacks=[early_stopping])

# Evaluate the model
score = model.evaluate(x_test, y_test)
print('{}% of samples were classified correctly!'.format(score[1] * 100))

# Save the trained model
model.save('cnn_model.h5')

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Predict on the test set
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predictions to class indices
y_true_classes = np.argmax(y_test, axis=1)  # True class indices

# Confusion Matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification Report
report = classification_report(y_true_classes, y_pred_classes)
print(report)