
import joblib
from sklearn.datasets import load_iris

def test_model_exists():
    model = joblib.load("model.joblib")
    assert model is not None

def test_model_prediction():
    model = joblib.load("model.joblib")

    iris = load_iris()

    sample = iris.data[:5]

    predictions = model.predict(sample)

    assert len(predictions) == 5
