from joblib import load
import os

# Add constants for directories
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "trading_model.joblib")

def inspect_trained_model(model_path=MODEL_PATH):
    """
    Loads a trained Logistic Regression model from a joblib file and prints its attributes.

    Args:
        model_path (str): Path to the joblib file containing the trained model.
    """
    try:
        # Check if model directory exists
        if not os.path.exists(MODEL_DIR):
            print(f"Warning: Model directory not found at: {MODEL_DIR}")
            return

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
        print("Please run the trend detector first to generate the model.")
    except Exception as e:
        print(f"Error loading or inspecting the model: {e}")

if __name__ == "__main__":
    inspect_trained_model()