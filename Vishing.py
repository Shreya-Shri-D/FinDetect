import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

_ROOT = os.path.dirname(os.path.abspath(__file__))

# Custom Decision Tree Classifier from scratch
class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        self.tree = self._build_tree(dataset)

    def _build_tree(self, dataset, depth=0):
        X, y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = X.shape
        if (depth >= self.max_depth) or (num_samples < self.min_samples_split) or len(set(y)) == 1:
            leaf_value = self._most_common_label(y)
            return leaf_value

        best_split = self._find_best_split(dataset, num_features)
        if best_split['gain'] == 0:
            return self._most_common_label(y)

        left_subtree = self._build_tree(best_split['left_dataset'], depth + 1)
        right_subtree = self._build_tree(best_split['right_dataset'], depth + 1)
        return {'feature_index': best_split['feature_index'], 'threshold': best_split['threshold'],
                'left': left_subtree, 'right': right_subtree}

    def _find_best_split(self, dataset, num_features):
        best_split = {'gain': 0}
        y = dataset[:, -1]
        current_gini = self._gini(y)
        for feature_index in range(num_features):
            thresholds = np.unique(dataset[:, feature_index])
            for threshold in thresholds:
                left_dataset, right_dataset = self._split(dataset, feature_index, threshold)
                if len(left_dataset) == 0 or len(right_dataset) == 0:
                    continue
                gain = self._information_gain(y, left_dataset[:, -1], right_dataset[:, -1], current_gini)
                if gain > best_split['gain']:
                    best_split = {'feature_index': feature_index, 'threshold': threshold, 'gain': gain,
                                  'left_dataset': left_dataset, 'right_dataset': right_dataset}
        return best_split

    def _split(self, dataset, feature_index, threshold):
        left = np.array([row for row in dataset if row[feature_index] <= threshold])
        right = np.array([row for row in dataset if row[feature_index] > threshold])
        return left, right

    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        impurity = 1 - sum((count / len(y)) ** 2 for count in counts)
        return impurity

    def _information_gain(self, y, left_y, right_y, current_gini):
        weight_left = len(left_y) / len(y)
        weight_right = len(right_y) / len(y)
        gain = current_gini - (weight_left * self._gini(left_y) + weight_right * self._gini(right_y))
        return gain

    def _most_common_label(self, y):
        return np.bincount(y.astype(int)).argmax()

    def predict_row(self, row, tree):
        if not isinstance(tree, dict):
            return tree
        feature_index = tree['feature_index']
        threshold = tree['threshold']
        if row[feature_index] <= threshold:
            return self.predict_row(row, tree['left'])
        else:
            return self.predict_row(row, tree['right'])

    def predict(self, X):
        return np.array([self.predict_row(row, self.tree) for row in X])

# RandomForest class using the custom DecisionTree
class RandomForest:
    def __init__(self, n_trees=7, max_depth=7, min_samples=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples)
            dataset_sample = self.bootstrap_samples(dataset)
            X_sample, y_sample = dataset_sample[:, :-1], dataset_sample[:, -1]
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        return self

    def bootstrap_samples(self, dataset):
        n_samples = dataset.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        dataset_sample = dataset[indices]
        return dataset_sample

    def most_common_label(self, y):
        y = list(y)
        most_occuring_value = max(y, key=y.count)
        return most_occuring_value

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        preds = np.swapaxes(predictions, 0, 1)
        majority_predictions = np.array([self.most_common_label(pred) for pred in preds])
        return majority_predictions

# Function to encapsulate the Streamlit app
def Vishing():
    # Load data
    data = pd.read_csv(os.path.join(_ROOT, "vishing_data.csv"))  

    # Preprocess timedelta columns
    data['Answer Speed (AVG)'] = pd.to_timedelta(data['Answer Speed (AVG)']).dt.total_seconds()
    data['Talk Duration (AVG)'] = pd.to_timedelta(data['Talk Duration (AVG)']).dt.total_seconds()
    data['Waiting Time (AVG)'] = pd.to_timedelta(data['Waiting Time (AVG)']).dt.total_seconds()

    # Define features and target
    X = data.drop(columns=['Vishing'])
    y = data['Vishing']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize Random Forest Classifier
    rf_classifier = RandomForest(n_trees=7, max_depth=7, min_samples=2)

    # Train the model
    rf_classifier.fit(X_train_scaled, y_train.values)

    # Predict on test set
    y_pred = rf_classifier.predict(X_test_scaled)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Streamlit app
    st.title('Voice Phishing (Vishing)')

    # Display accuracy
    st.write('Model Accuracy on Test Set:', accuracy)

    # Sidebar user input
    st.sidebar.header('User Input')

    # Prediction function
    def predict_vishing(answer_rate, abandoned_calls, answer_speed_avg, talk_duration_avg, waiting_time_avg, service_level):
        answer_speed_avg = timedelta(minutes=answer_speed_avg).total_seconds()
        talk_duration_avg = timedelta(minutes=talk_duration_avg).total_seconds()
        waiting_time_avg = timedelta(minutes=waiting_time_avg).total_seconds()
        input_data = np.array([[answer_rate, abandoned_calls, answer_speed_avg, talk_duration_avg, waiting_time_avg, service_level]])
        input_data_scaled = scaler.transform(input_data)  # Scale input data
        prediction = rf_classifier.predict(input_data_scaled)
        return "Vishing High Risk" if prediction[0] == 1 else "Vishing Low Risk"

    # Input fields with compatible default values
    answer_rate = st.sidebar.number_input('Answer Rate', value=float(data['Answer Rate'].mean()))
    abandoned_calls = st.sidebar.number_input('Abandoned Calls', value=int(data['Abandoned Calls'].mean()))
    answer_speed_avg = st.sidebar.number_input('Answer Speed (AVG)', value=float(data['Answer Speed (AVG)'].mean() / 60))
    talk_duration_avg = st.sidebar.number_input('Talk Duration (AVG)', value=float(data['Talk Duration (AVG)'].mean() / 60))
    waiting_time_avg = st.sidebar.number_input('Waiting Time (AVG)', value=float(data['Waiting Time (AVG)'].mean() / 60))
    service_level = st.sidebar.number_input('Service Level (20 Seconds)', value=float(data['Service Level (20 Seconds)'].mean()))

    # Predict button
    if st.sidebar.button('Predict'):
        prediction = predict_vishing(answer_rate, abandoned_calls, answer_speed_avg, talk_duration_avg, waiting_time_avg, service_level)
        st.write('Prediction:', prediction)

# Call the vishing page function to run the app
if __name__ == "__main__":
    Vishing()
