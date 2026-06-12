# General Machine Learning Training and Inference Principles

When building a machine learning system, treat the training pipeline and the inference (prediction) pipeline as two stages of the same process.

## 1. Training Data Representation

The model does not need to be trained on the raw form of the data.

Features and targets may be transformed, encoded, scaled, normalized, aggregated, or engineered in any way that improves the model's ability to learn meaningful patterns.

Examples of valid transformations include:

* Encoding categorical variables
* Scaling or normalization
* Logarithmic transformations
* Polynomial features
* Interaction features
* Ratio features
* Dimensionality reduction
* Domain-specific feature engineering

The objective of training is to provide the model with the most informative representation of the underlying problem.

---

## 2. Feature Validity Rule

A feature is valid if it can be generated using only information that would be available when making a real-world prediction.

Before using any feature, ask:

> "Will this information be available at the moment a prediction is requested?"

If the answer is yes, the feature may be considered for training.

If the answer is no, the feature should not be used.

---

## 3. Target Leakage Prevention

The target variable, and any information derived directly or indirectly from the target variable, must never be included in the feature set.

This includes:

* The target itself
* Transformations of the target
* Aggregations containing the target
* Features computed using future information
* Features that would only be known after the outcome occurs

A model should never have access to information that reveals or partially reveals the answer it is attempting to predict.

---

## 4. Feature Engineering Guidelines

Feature engineering may be performed freely on valid input features.

Engineered features may be:

* Added alongside original features
* Used to replace original features
* Evaluated experimentally to determine usefulness

Feature selection is an iterative process. There is no requirement to immediately remove original features after creating engineered ones.

The usefulness of engineered features should be determined through validation and model evaluation.

---

## 5. Consistency Between Training and Inference

Any preprocessing, encoding, scaling, transformation, or feature engineering step used during training must also be applied during inference.

The inference pipeline should reproduce the same data representation that the model saw during training.

Conceptually:

Raw Input
→ Preprocessing
→ Feature Engineering
→ Model Prediction

The same sequence of operations must be executed both during training and during deployment.

---

## 6. Target Transformations

The target variable may be transformed before training if doing so improves learning performance.

When a transformed target is used:

* The model learns to predict the transformed target.
* Evaluation may occur in either transformed space or original space depending on the objective.
* Predictions intended for end users should typically be converted back into the original target representation.

The representation used internally by the model does not need to match the representation shown to users.

---

## 7. Production Output Principle

Users should generally receive predictions in a form that is meaningful within the application domain.

Even if the model operates on transformed data internally, outputs should usually be converted back into the original business, scientific, financial, or operational representation before being returned.

The internal optimization space and the external presentation space may be different.

---

## 8. Core Principle

A machine learning model may train on transformed features and transformed targets, provided that:

1. Every feature can be generated from information available at prediction time.
2. No target-derived information is included in the feature set.
3. The same preprocessing pipeline is applied during both training and inference.
4. Predictions are converted into an appropriate user-facing representation when necessary.

The goal is not to preserve the original form of the data during training, but to preserve correctness, consistency, and real-world usability throughout the entire machine learning pipeline.
