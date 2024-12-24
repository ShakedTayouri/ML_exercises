from WeightedDecisionTreeClassifier import WeightedDecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load sample dataset
data = load_iris()
X = data.data
y = data.target

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and fit the model
clf = WeightedDecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_leaf=3)
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)
print("Predictions:", predictions)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Debugging - Print leaf classes and Gini
for i, weight in enumerate(clf.weights_):
    if weight is not None:  # Skip non-leaf nodes
        cls, gini = weight
        print(f"Leaf {i}: Class={cls}, Gini={gini:.2f}")
    else:
        print(f"Node {i}: Internal Node (No Class)")

