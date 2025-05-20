import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Q1: k-NN implementation
class KNNClassifier:
    def __init__(self, k=3, distance='euclidean'): # Value of k and whether distance is euclidean or manhattan
        self.k = k
        self.distance = distance 
    
    def fit(self, x, y): # Store the training sets
        self.x_train = x
        self.y_train = y

    def _distance(self, x1, x2): # Computes the distance between 2 points, x1 and x2
        if self.distance == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance == 'manhattan':
            return np.sum(np.abs(x1 - x2))
    
    def predict(self, x_test): # Predict the class labels
        predictions = []
        for x in x_test:
            distances = [self._distance(x, x_train) for x_train in self.x_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            prediction = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(prediction)
        return np.array(predictions)

# Q2: Load a standard dataset
iris = load_iris()
x = iris.data
y = iris.target

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Q3: Evaluation
k_values = [2, 3, 5, 7]
knn_accuracies = []

for k in k_values:
    knn = KNNClassifier(k=k, distance='euclidean')
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    acc = np.mean(y_pred == y_test)
    knn_accuracies.append(acc)
    print(f"[knn] k = {k} -> Accuracy = {acc:.2f}")

plt.figure(figsize=(8, 5))
plt.plot(k_values, knn_accuracies, marker='o', label='k-NN', linestyle='--')

# Q4: Comparison with scikit-learn
sklearn_accuracies = []

for k in k_values:
    sk_model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    sk_model.fit(x_train, y_train)
    y_pred = sk_model.predict(x_test)
    acc = np.mean(y_pred == y_test)
    sklearn_accuracies.append(acc)
    print(f"[Sklearn] k = {k} -> Accuracy = {acc:.2f}")

plt.plot(k_values, sklearn_accuracies, marker='s', label='Sklearn k-NN')

plt.title("k-NN Accuracy on Iris Dataset")
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.xticks(k_values)
plt.ylim(0.9, 1.05)
plt.grid(True)
plt.legend()
plt.show()

# Observations: the accuracy and charts of both methods appear to be the same