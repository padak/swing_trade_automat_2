from joblib import load
import os

def inspect_trained_model(model_path='src/trading_model.joblib'):
    """
    Loads a trained Logistic Regression model from a joblib file and prints its attributes.

    Args:
        model_path (str): Path to the joblib file containing the trained model.
    """
    try:
        # Load the model from the file
        model = load(model_path)
        print(f"Model loaded successfully from: {model_path}")

        # Check if it's a Logistic Regression model (for basic validation)
        from sklearn.linear_model import LogisticRegression
        if not isinstance(model, LogisticRegression):
            print("Warning: Loaded model is not a Logistic Regression model.")
            return

        # Print model attributes
        print("\n--- Model Attributes ---")
        print(f"Coefficients (coef_): {model.coef_}")
        print(f"Intercept (intercept_): {model.intercept_}")
        print(f"Classes (classes_): {model.classes_}")
        print(f"Number of iterations (n_iter_): {model.n_iter_}")
        # You can add more attributes to inspect as needed

    except FileNotFoundError:
        print(f"Error: Model file not found at: {model_path}")
    except Exception as e:
        print(f"Error loading or inspecting the model: {e}")

if __name__ == "__main__":
    inspect_trained_model()