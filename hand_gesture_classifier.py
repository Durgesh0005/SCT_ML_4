import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

DATA_DIR = "gesture_dataset"
GESTURES = sorted(os.listdir(DATA_DIR))
IMG_SIZE = 64

X, y = [], []
image_files = []

for label, gesture in enumerate(GESTURES):
    folder = os.path.join(DATA_DIR, gesture)
    for fname in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, fname), cv2.IMREAD_GRAYSCALE)
        if img is None: 
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img.flatten())
        y.append(label)
        image_files.append((gesture, fname))

X, y = np.array(X), np.array(y)

X_train, X_test, y_train, y_test, files_train, files_test = train_test_split(
    X, y, image_files, test_size=0.2, random_state=42
)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred, target_names=GESTURES))

for i in range(5):
    gesture, fname = files_test[i]
    img = X_test[i].reshape(IMG_SIZE, IMG_SIZE)
    plt.imshow(img, cmap='gray')
    plt.title(f"Actual: {gesture} | Predicted: {GESTURES[y_pred[i]]}")
    plt.axis('off')
    plt.show()
