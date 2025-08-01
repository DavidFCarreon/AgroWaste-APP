# FRAP Prediction Project

Predicting antioxidant activity from food composition data using Machine Learning and Deep Learning.

## Project Structure

```
(folders structure here)
```

## Installation

```bash
git clone https://github.com/tuusuario/frap-prediction.git
cd frap-prediction
pip install -e .
```

## Usage

### Basic Example

```python
from frap_predictor.data.data_loader import load_frap_data
from frap_predictor.models.dl_model import FRAPDeepLearningModel

# Load and prepare data
X_train, X_test, y_train, y_test = load_frap_data("data/raw/frap_data.csv")

# Initialize and train model
model = FRAPDeepLearningModel(input_dim=X_train.shape[1])
history = model.train(X_train, y_train, validation_data=(X_test, y_test))

# Evaluate
metrics = model.evaluate(X_test, y_test)
print(metrics)
```

## Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
