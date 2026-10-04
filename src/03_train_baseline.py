import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# 1. LOAD PROCESSED BATTERY DATA
# ============================================================

file_path = "data/processed/B0005_features.csv"

df = pd.read_csv(file_path)

print("\nDataset loaded successfully")
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. DEFINE INPUT FEATURES AND TARGET
# ============================================================

features = [
    "duration_min",
    "T_initial_C",
    "T_max_C",
    "delta_T_C",
    "heating_rate_C_per_min",
    "V_mean",
    "V_min"
]

target = "SOH"

X = df[features]
y = df[target]

print("\nFeatures used:")
for feature in features:
    print("-", feature)


# ============================================================
# 3. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================
#
# IMPORTANT:
# We do NOT randomly shuffle degradation cycles.
#
# First 80% of battery life = training
# Last 20% = testing
#
# This is more realistic because we use earlier battery
# behavior to predict later battery health.
# ============================================================

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\n----------------------------------------")
print("CHRONOLOGICAL SPLIT")
print("----------------------------------------")

print("Training observations:", len(X_train))
print("Testing observations:", len(X_test))

print(
    "Training cycles:",
    df["cycle"].iloc[0],
    "to",
    df["cycle"].iloc[split_index - 1]
)

print(
    "Testing cycles:",
    df["cycle"].iloc[split_index],
    "to",
    df["cycle"].iloc[-1]
)


# ============================================================
# 4. CONVERT PANDAS DATA TO NUMPY
# ============================================================

X_train_np = X_train.to_numpy(dtype=float)
X_test_np = X_test.to_numpy(dtype=float)

y_train_np = y_train.to_numpy(dtype=float)
y_test_np = y_test.to_numpy(dtype=float)


# ============================================================
# 5. ADD INTERCEPT COLUMN
# ============================================================
#
# Linear regression model:
#
# SOH_hat =
# beta_0 +
# beta_1*x_1 +
# beta_2*x_2 +
# ...
# beta_n*x_n
#
# beta_0 is the intercept.
#
# Adding a column of ones allows beta_0 to be estimated.
# ============================================================

X_train_design = np.column_stack([
    np.ones(len(X_train_np)),
    X_train_np
])

X_test_design = np.column_stack([
    np.ones(len(X_test_np)),
    X_test_np
])


# ============================================================
# 6. TRAIN LINEAR REGRESSION USING NUMPY
# ============================================================
#
# We solve the ordinary least-squares problem:
#
#       minimize ||X beta - y||^2
#
# np.linalg.lstsq() calculates the regression coefficients.
# ============================================================

beta, residuals, rank, singular_values = np.linalg.lstsq(
    X_train_design,
    y_train_np,
    rcond=None
)


# ============================================================
# 7. MAKE SOH PREDICTIONS
# ============================================================

y_pred = X_test_design @ beta


# ============================================================
# 8. CALCULATE EVALUATION METRICS
# ============================================================

# Mean Absolute Error
mae = np.mean(
    np.abs(y_test_np - y_pred)
)

# Root Mean Squared Error
rmse = np.sqrt(
    np.mean((y_test_np - y_pred) ** 2)
)

# R-squared
ss_res = np.sum(
    (y_test_np - y_pred) ** 2
)

ss_tot = np.sum(
    (y_test_np - np.mean(y_test_np)) ** 2
)

r2 = 1 - (ss_res / ss_tot)


print("\n========================================")
print("LINEAR REGRESSION BASELINE RESULTS")
print("========================================")

print(f"MAE  : {mae:.4f} SOH percentage points")
print(f"RMSE : {rmse:.4f} SOH percentage points")
print(f"R²   : {r2:.4f}")


# ============================================================
# 9. PRINT MODEL COEFFICIENTS
# ============================================================

print("\n----------------------------------------")
print("MODEL COEFFICIENTS")
print("----------------------------------------")

print(f"Intercept: {beta[0]:.6f}")

for feature, coefficient in zip(features, beta[1:]):
    print(
        f"{feature:25s}: {coefficient:.6f}"
    )


# ============================================================
# 10. SHOW SAMPLE PREDICTIONS
# ============================================================

print("\n----------------------------------------")
print("SAMPLE TEST PREDICTIONS")
print("----------------------------------------")

test_cycles = df["cycle"].iloc[split_index:].to_numpy()

for i in range(min(10, len(y_pred))):

    print(
        f"Cycle {test_cycles[i]:3d} | "
        f"True SOH = {y_test_np[i]:6.2f}% | "
        f"Predicted SOH = {y_pred[i]:6.2f}%"
    )


# ============================================================
# 11. CREATE FIGURES DIRECTORY
# ============================================================

os.makedirs(
    "figures",
    exist_ok=True
)


# ============================================================
# 12. PLOT TRUE VS PREDICTED SOH
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    test_cycles,
    y_test_np,
    marker="o",
    markersize=4,
    label="True SOH"
)

plt.plot(
    test_cycles,
    y_pred,
    linestyle="--",
    label="Predicted SOH"
)

plt.xlabel("Discharge Cycle")
plt.ylabel("SOH (%)")

plt.title(
    "NASA B0005 — SOH Prediction\n"
    "Chronological Linear Regression Baseline"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "figures/soh_linear_baseline.png",
    dpi=300
)

plt.show()


# ============================================================
# 13. PREDICTION ERROR
# ============================================================

errors = y_pred - y_test_np

plt.figure(figsize=(9, 5))

plt.plot(
    test_cycles,
    errors,
    marker="o",
    markersize=4
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Discharge Cycle")
plt.ylabel("Prediction Error (SOH percentage points)")

plt.title(
    "SOH Prediction Error — B0005"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "figures/soh_prediction_error.png",
    dpi=300
)

plt.show()


# ============================================================
# 14. SAVE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "cycle": test_cycles,
    "true_SOH": y_test_np,
    "predicted_SOH": y_pred,
    "error": errors
})

os.makedirs(
    "results",
    exist_ok=True
)

results.to_csv(
    "results/B0005_linear_predictions.csv",
    index=False
)


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n========================================")
print("BASELINE MODEL COMPLETE")
print("========================================")

print(
    "Prediction results saved to:"
    "\nresults/B0005_linear_predictions.csv"
)

print(
    "\nFigures saved to:"
    "\nfigures/soh_linear_baseline.png"
    "\nfigures/soh_prediction_error.png"
)