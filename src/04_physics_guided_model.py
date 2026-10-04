import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/processed/B0005_features.csv")

features = [
    "duration_min",
    "T_initial_C",
    "T_max_C",
    "delta_T_C",
    "heating_rate_C_per_min",
    "V_mean",
    "V_min"
]

X = df[features].to_numpy(dtype=float)
y = df["SOH"].to_numpy(dtype=float)

cycles = df["cycle"].to_numpy()


# ============================================================
# 2. CHRONOLOGICAL SPLIT
# ============================================================

split_index = int(len(df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

test_cycles = cycles[split_index:]


# ============================================================
# 3. STANDARDIZE FEATURES USING TRAINING DATA ONLY
# ============================================================
#
# This prevents information from the future test region
# entering preprocessing.
# ============================================================

mean_X = np.mean(X_train, axis=0)
std_X = np.std(X_train, axis=0)

# Avoid division by zero
std_X[std_X == 0] = 1.0

X_train_scaled = (X_train - mean_X) / std_X
X_test_scaled = (X_test - mean_X) / std_X


# ============================================================
# 4. CREATE NONLINEAR FEATURES
# ============================================================
#
# We add x^2 terms:
#
# SOH = beta0 + beta*x + gamma*x^2
#
# This allows simple nonlinear degradation relationships
# without introducing a black-box model.
# ============================================================

X_train_poly = np.column_stack([
    X_train_scaled,
    X_train_scaled ** 2
])

X_test_poly = np.column_stack([
    X_test_scaled,
    X_test_scaled ** 2
])


# Add intercept
X_train_design = np.column_stack([
    np.ones(len(X_train_poly)),
    X_train_poly
])

X_test_design = np.column_stack([
    np.ones(len(X_test_poly)),
    X_test_poly
])


# ============================================================
# 5. RIDGE REGRESSION
# ============================================================
#
# Objective:
#
# min ||X beta - y||² + lambda ||beta||²
#
# Ridge regularization reduces unstable coefficients.
# ============================================================

lambda_ridge = 1.0

n_features = X_train_design.shape[1]

I = np.eye(n_features)

# Do not penalize intercept
I[0, 0] = 0

beta = np.linalg.solve(
    X_train_design.T @ X_train_design
    + lambda_ridge * I,
    X_train_design.T @ y_train
)


# ============================================================
# 6. RAW NONLINEAR PREDICTIONS
# ============================================================

y_pred_raw = X_test_design @ beta


# ============================================================
# 7. PHYSICS-GUIDED DEGRADATION CONSTRAINT
# ============================================================
#
# Battery health generally degrades over long time scales.
#
# However, measured capacity can show short-term recovery
# and measurement noise.
#
# Therefore we do NOT force the experimental SOH labels
# themselves to be strictly monotonic.
#
# Instead, we create a slowly evolving latent health estimate.
#
# alpha controls how quickly the virtual health state follows
# new measurements.
# ============================================================

alpha = 0.35

y_pred_physics = np.zeros_like(y_pred_raw)

# Initialize using first model prediction
y_pred_physics[0] = y_pred_raw[0]

for i in range(1, len(y_pred_raw)):

    # Measurement-informed state update
    candidate = (
        alpha * y_pred_raw[i]
        + (1 - alpha) * y_pred_physics[i - 1]
    )

    # Long-term degradation constraint:
    # latent health should not increase.
    y_pred_physics[i] = min(
        candidate,
        y_pred_physics[i - 1]
    )


# ============================================================
# 8. METRIC FUNCTION
# ============================================================

def calculate_metrics(true, pred):

    mae = np.mean(
        np.abs(true - pred)
    )

    rmse = np.sqrt(
        np.mean((true - pred) ** 2)
    )

    ss_res = np.sum(
        (true - pred) ** 2
    )

    ss_tot = np.sum(
        (true - np.mean(true)) ** 2
    )

    r2 = 1 - ss_res / ss_tot

    return mae, rmse, r2


# ============================================================
# 9. EVALUATE RAW NONLINEAR MODEL
# ============================================================

raw_mae, raw_rmse, raw_r2 = calculate_metrics(
    y_test,
    y_pred_raw
)


# ============================================================
# 10. EVALUATE PHYSICS-GUIDED MODEL
# ============================================================

phy_mae, phy_rmse, phy_r2 = calculate_metrics(
    y_test,
    y_pred_physics
)


print("\n========================================")
print("NONLINEAR RIDGE MODEL")
print("========================================")

print(f"MAE  : {raw_mae:.4f}")
print(f"RMSE : {raw_rmse:.4f}")
print(f"R²   : {raw_r2:.4f}")


print("\n========================================")
print("PHYSICS-GUIDED HEALTH ESTIMATOR")
print("========================================")

print(f"MAE  : {phy_mae:.4f}")
print(f"RMSE : {phy_rmse:.4f}")
print(f"R²   : {phy_r2:.4f}")


# ============================================================
# 11. COMPARE AGAINST LINEAR BASELINE
# ============================================================

baseline_mae = 0.8341
baseline_rmse = 0.9059
baseline_r2 = 0.4267

print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

print(
    f"Linear baseline RMSE       : "
    f"{baseline_rmse:.4f}"
)

print(
    f"Nonlinear ridge RMSE       : "
    f"{raw_rmse:.4f}"
)

print(
    f"Physics-guided RMSE        : "
    f"{phy_rmse:.4f}"
)


# ============================================================
# 12. PLOT RESULTS
# ============================================================

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(9, 5))

plt.plot(
    test_cycles,
    y_test,
    label="Measured SOH"
)

plt.plot(
    test_cycles,
    y_pred_raw,
    "--",
    label="Nonlinear Ridge"
)

plt.plot(
    test_cycles,
    y_pred_physics,
    label="Physics-Guided Estimate"
)

plt.xlabel("Discharge Cycle")
plt.ylabel("SOH (%)")

plt.title(
    "Battery SOH Estimation — "
    "Data-Driven vs Physics-Guided"
)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "figures/physics_guided_soh.png",
    dpi=300
)

plt.show()


# ============================================================
# 13. SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "cycle": test_cycles,
    "measured_SOH": y_test,
    "nonlinear_prediction": y_pred_raw,
    "physics_guided_prediction": y_pred_physics
})

results.to_csv(
    "results/physics_guided_predictions.csv",
    index=False
)

print("\nResults saved successfully.")