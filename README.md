# Explainable AI Toolkit

A comprehensive toolkit for generating transparent and interpretable explanations for predictions made by complex Artificial Intelligence (AI) models. This project aims to demystify black-box models, fostering trust and enabling better understanding of their decision-making processes.

## Features
- **LIME (Local Interpretable Model-agnostic Explanations):** Generate local explanations for individual predictions.
- **SHAP (SHapley Additive exPlanations):** Compute Shapley values to explain the output of any machine learning model.
- **Feature Importance:** Determine the global importance of features for a given model.
- **Counterfactual Explanations:** Identify the smallest changes to input features that flip a model's prediction.
- **Visualizations:** Provide intuitive graphical representations of explanations.
- **Model-agnostic:** Works with various types of machine learning models (e.g., scikit-learn, TensorFlow, PyTorch).

## Getting Started

### Installation

```bash
pip install explainable-ai-toolkit
```

### Quick Start with LIME

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from explainable_ai_toolkit.lime_explainer import LimeTabularExplainer

# 1. Train a simple classifier
X, y = make_classification(n_samples=100, n_features=10, n_informative=5, n_redundant=0, random_state=42)
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
class_names = ["class_0", "class_1"]

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 2. Initialize LIME explainer
explainer = LimeTabularExplainer(
    training_data=X,
    feature_names=feature_names,
    class_names=class_names,
    mode="classification"
)

# 3. Explain a prediction
instance_to_explain = X[0]
explanation = explainer.explain_instance(
    data_row=instance_to_explain,
    predict_fn=model.predict_proba,
    num_features=5
)

print("Explanation for instance:")
print(explanation.as_list())
```

## Project Structure

```
explainable-ai-toolkit/
├── src/
│   ├── __init__.py
│   ├── lime_explainer.py   # LIME implementation
│   ├── shap_explainer.py   # SHAP implementation
│   └── utils.py            # Utility functions
├── tests/
├── docs/
├── requirements.txt
└── README.md
```

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for details.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
