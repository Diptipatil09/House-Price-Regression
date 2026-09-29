import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------
# 1. Create a Real-World House Price Dataset
# ---------------------------------------------------------

data = {
    "Area": [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500,
             2800, 3000, 1100, 1400, 1700, 2100, 2400],
    
    "Bedrooms": [2, 2, 3, 3, 3, 4, 4, 4,
                 5, 5, 2, 3, 3, 4, 4],
    
    "Bathrooms": [1, 2, 2, 2, 3, 3, 3, 4,
                  4, 4, 2, 2, 3, 3, 4],
    
    "Age": [20, 15, 10, 8, 5, 4, 3, 2,
            1, 1, 18, 12, 7, 5, 3],
    
    "Price": [2500000, 3200000, 4000000, 4800000, 5800000,
              6500000, 7200000, 8500000, 9500000, 11000000,
              3500000, 4500000, 5500000, 7000000, 8200000]
}

df = pd.DataFrame(data)

print("House Price Dataset:")
print(df)


# ---------------------------------------------------------
# 2. Separate Input and Output
# ---------------------------------------------------------

X = df[["Area", "Bedrooms", "Bathrooms", "Age"]]
y = df["Price"]


# ---------------------------------------------------------
# 3. Split Dataset
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ---------------------------------------------------------
# 4. Standardize Data for Gradient Descent
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------
# 5. Gradient Descent Linear Regression
# ---------------------------------------------------------

class GradientDescentLinearRegression:

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = 0
        self.cost_history = []

    def fit(self, X, y):

        samples, features = X.shape

        self.weights = np.zeros(features)
        self.bias = 0

        for i in range(self.iterations):

            # Prediction
            y_pred = np.dot(X, self.weights) + self.bias

            # Error
            error = y_pred - y

            # Cost / Mean Squared Error
            cost = np.mean(error ** 2)

            self.cost_history.append(cost)

            # Calculate gradients
            dw = (2 / samples) * np.dot(X.T, error)
            db = (2 / samples) * np.sum(error)

            # Update weights
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, X):

        return np.dot(X, self.weights) + self.bias


# Create Gradient Descent model
gd_model = GradientDescentLinearRegression(
    learning_rate=0.01,
    iterations=1000
)

# Train model
gd_model.fit(X_train_scaled, y_train.values)

# Predict
y_pred_gd = gd_model.predict(X_test_scaled)


# ---------------------------------------------------------
# 6. Evaluation Function
# ---------------------------------------------------------

def evaluate_model(model_name, y_actual, y_pred):

    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_pred)

    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    print("MAE  :", mae)
    print("MSE  :", mse)
    print("RMSE :", rmse)
    print("R2 Score :", r2)

    return mae, mse, rmse, r2


# Evaluate Gradient Descent
gd_results = evaluate_model(
    "Linear Regression using Gradient Descent",
    y_test,
    y_pred_gd
)


# ---------------------------------------------------------
# 7. Normal Linear Regression
# ---------------------------------------------------------

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

linear_results = evaluate_model(
    "Linear Regression using Normal Equation",
    y_test,
    y_pred_linear
)


# ---------------------------------------------------------
# 8. Polynomial Regression
# ---------------------------------------------------------

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()

poly_model.fit(X_train_poly, y_train)

y_pred_poly = poly_model.predict(X_test_poly)

poly_results = evaluate_model(
    "Polynomial Regression",
    y_test,
    y_pred_poly
)


# ---------------------------------------------------------
# 9. Model Comparison
# ---------------------------------------------------------

results = pd.DataFrame({
    "Model": [
        "Gradient Descent Linear Regression",
        "Linear Regression",
        "Polynomial Regression"
    ],

    "MAE": [
        gd_results[0],
        linear_results[0],
        poly_results[0]
    ],

    "MSE": [
        gd_results[1],
        linear_results[1],
        poly_results[1]
    ],

    "RMSE": [
        gd_results[2],
        linear_results[2],
        poly_results[2]
    ],

    "R2 Score": [
        gd_results[3],
        linear_results[3],
        poly_results[3]
    ]
})

print("\n\nMODEL COMPARISON")
print("=" * 80)
print(results)


# ---------------------------------------------------------
# 10. Gradient Descent Cost Graph
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, gd_model.iterations + 1),
    gd_model.cost_history
)

plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.title("Gradient Descent - Cost vs Iterations")

plt.grid(True)
plt.show()


# ---------------------------------------------------------
# 11. Actual vs Predicted Graph
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred_gd)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")

plt.title("Actual vs Predicted House Prices")

plt.grid(True)
plt.show()


# ---------------------------------------------------------
# 12. Display Predictions
# ---------------------------------------------------------

prediction_table = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred_gd
})

print("\nActual vs Predicted Prices:")
print(prediction_table)