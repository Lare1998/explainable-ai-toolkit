import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

class LimeTabularExplainer:
    """
    Explains predictions of any tabular classifier or regressor using LIME (Local Interpretable Model-agnostic Explanations).
    """
    def __init__(self, training_data: np.ndarray, feature_names: list, class_names: list, mode: str = "classification", kernel_width: float = 0.75):
        """
        Initializes the LIME Tabular Explainer.

        Args:
            training_data (np.ndarray): The training data used to fit the model.
            feature_names (list): A list of strings, names of the features.
            class_names (list): A list of strings, names of the classes (for classification).
            mode (str): "classification" or "regression".
            kernel_width (float): Width of the exponential kernel for weighting samples.
        """
        self.training_data = training_data
        self.feature_names = feature_names
        self.class_names = class_names
        self.mode = mode
        self.kernel_width = kernel_width
        self.scaler = StandardScaler()
        self.scaler.fit(training_data)

    def _kernel(self, d, kernel_width):
        """Exponential kernel for weighting samples."""
        return np.sqrt(np.exp(-(d**2) / kernel_width**2))

    def _generate_neighboring_data(self, instance: np.ndarray, num_samples: int = 5000, perturbation_std: float = 0.1):
        """
        Generates perturbed samples around the instance to be explained.
        """
        data = np.random.normal(0, 1, size=(num_samples, instance.shape[0]))
        data = self.scaler.inverse_transform(data)
        
        # Ensure the instance itself is included
        data[0] = instance
        
        return data

    def explain_instance(self, data_row: np.ndarray, predict_fn, num_features: int = 5):
        """
        Explains the prediction of a single instance.

        Args:
            data_row (np.ndarray): The instance to be explained.
            predict_fn (callable): A function that takes a numpy array of data and returns predictions.
                                   For classification, it should return probabilities for each class.
            num_features (int): The number of features to include in the explanation.

        Returns:
            Explanation: An object containing the explanation.
        """
        # Generate perturbed data
        perturbed_data = self._generate_neighboring_data(data_row)
        
        # Get predictions for perturbed data
        if self.mode == "classification":
            predictions = predict_fn(perturbed_data)
            # For classification, we explain the probability of the predicted class
            predicted_class_idx = np.argmax(predict_fn(data_row.reshape(1, -1))[0])
            labels = predictions[:, predicted_class_idx]
        else: # regression
            predictions = predict_fn(perturbed_data)
            labels = predictions

        # Calculate distances and weights
        distances = pairwise_distances(perturbed_data, data_row.reshape(1, -1), metric=\'euclidean\').ravel()
        weights = self._kernel(distances, self.kernel_width)

        # Train a local interpretable model (e.g., Linear Regression)
        # We use the original feature values for the local model
        local_model = LinearRegression()
        local_model.fit(perturbed_data, labels, sample_weight=weights)

        # Get feature importances from the local model
        feature_importances = local_model.coef_

        # Sort features by importance
        sorted_features = sorted(zip(self.feature_names, feature_importances), key=lambda x: abs(x[1]), reverse=True)

        # Return top N features
        return Explanation(data_row, self.feature_names, self.class_names, self.mode, sorted_features[:num_features])

class Explanation:
    """
    Holds the explanation for a single prediction.
    """
    def __init__(self, instance, feature_names, class_names, mode, explanation_list):
        self.instance = instance
        self.feature_names = feature_names
        self.class_names = class_names
        self.mode = mode
        self.explanation_list = explanation_list

    def as_list(self):
        """Returns the explanation as a list of (feature, weight) tuples."""
        return self.explanation_list

    def as_html(self):
        """Generates an HTML representation of the explanation (simplified)."""
        html = "<div><b>Explanation for instance:</b><br>"
        for feature, weight in self.explanation_list:
            html += f"- {feature}: {weight:.4f}<br>"
        html += "</div>"
        return html

if __name__ == "__main__":
    # Example usage with a dummy classifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    # 1. Generate synthetic data
    X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=0, random_state=42)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    class_names = ["class_0", "class_1"]

    # 2. Train a simple classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # 3. Initialize LIME explainer
    explainer = LimeTabularExplainer(
        training_data=X,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification"
    )

    # 4. Explain a prediction
    instance_to_explain = X[0]
    explanation = explainer.explain_instance(
        data_row=instance_to_explain,
        predict_fn=model.predict_proba,
        num_features=5
    )

    print("Explanation for instance:")
    print(explanation.as_list())
    print("\nHTML Explanation:")
    print(explanation.as_html())
