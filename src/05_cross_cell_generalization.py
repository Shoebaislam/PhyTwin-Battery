from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_CELLS = ["B0005", "B0006", "B0007"]
TEST_CELL = "B0018"

DATA_DIR = "data/raw"

FEATURES = [
    "duration_min",
    "T_initial_C",
    "T_max_C",
    "delta_T_C",
    "heating_rate_C_per_min",
    "V_mean",
    "V_min"
]


# ============================================================
# 1. EXTRACT FEATURES FROM ONE BATTERY
# ============================================================

def extract_cell_features(cell_name):

    file_path = os.path.join(
        DATA_DIR,
        f"{cell_name}.mat"
    )

    print(f"\nLoading {cell_name}...")

    mat = loadmat(file_path)

    battery = mat[cell_name]

    cycles = battery["cycle"][0, 0]

    rows = []

    discharge_number = 0

    for i in range(cycles.shape[1]):

        cycle_type = cycles[0, i]["type"][0]

        if cycle_type != "discharge":
            continue

        discharge_number += 1

        data = cycles[0, i]["data"][0, 0]

        voltage = data["Voltage_measured"][0]
        temperature = data["Temperature_measured"][0]
        time = data["Time"][0]

        capacity = float(
            data["Capacity"][0, 0]
        )

        # ----------------------------------------------------
        # Discharge duration
        # ----------------------------------------------------

        duration_min = (
            time[-1] - time[0]
        ) / 60


        # ----------------------------------------------------
        # Thermal features
        # ----------------------------------------------------

        T_initial = temperature[0]

        T_max = np.max(
            temperature
        )

        delta_T = (
            T_max - T_initial
        )

        peak_index = np.argmax(
            temperature
        )

        time_to_peak_min = (
            time[peak_index] - time[0]
        ) / 60

        if time_to_peak_min > 0:

            heating_rate = (
                delta_T /
                time_to_peak_min
            )

        else:

            heating_rate = np.nan


        # ----------------------------------------------------
        # Voltage features
        # ----------------------------------------------------

        V_mean = np.mean(
            voltage
        )

        V_min = np.min(
            voltage
        )


        # ----------------------------------------------------
        # Store cycle
        # ----------------------------------------------------

        rows.append({

            "cell": cell_name,

            "cycle": discharge_number,

            "capacity_Ah": capacity,

            "duration_min": duration_min,

            "T_initial_C": T_initial,

            "T_max_C": T_max,

            "delta_T_C": delta_T,

            "heating_rate_C_per_min":
                heating_rate,

            "V_mean": V_mean,

            "V_min": V_min

        })


    df = pd.DataFrame(rows)


    # ========================================================
    # CELL-SPECIFIC SOH
    # ========================================================

    Q_ref = df[
        "capacity_Ah"
    ].iloc[0]

    df["SOH"] = (
        df["capacity_Ah"]
        / Q_ref
    ) * 100


    print(
        f"{cell_name}: "
        f"{len(df)} discharge cycles"
    )

    print(
        f"Initial capacity: "
        f"{Q_ref:.4f} Ah"
    )

    print(
        f"Final SOH: "
        f"{df['SOH'].iloc[-1]:.2f}%"
    )

    return df


# ============================================================
# 2. LOAD ALL CELLS
# ============================================================

all_cells = {}

for cell in TRAIN_CELLS + [TEST_CELL]:

    all_cells[cell] = (
        extract_cell_features(cell)
    )


# ============================================================
# 3. COMBINE TRAINING CELLS
# ============================================================

train_df = pd.concat(
    [
        all_cells[cell]
        for cell in TRAIN_CELLS
    ],
    ignore_index=True
)

test_df = all_cells[
    TEST_CELL
].copy()


print("\n========================================")
print("DATASET SUMMARY")
print("========================================")

print(
    "Training cells:",
    TRAIN_CELLS
)

print(
    "Training observations:",
    len(train_df)
)

print(
    "Unseen test cell:",
    TEST_CELL
)

print(
    "Testing observations:",
    len(test_df)
)


# ============================================================
# 4. REMOVE INVALID VALUES
# ============================================================

train_df = train_df.dropna(
    subset=FEATURES + ["SOH"]
)

test_df = test_df.dropna(
    subset=FEATURES + ["SOH"]
)


# ============================================================
# 5. PREPARE X AND y
# ============================================================

X_train = train_df[
    FEATURES
].to_numpy(dtype=float)

y_train = train_df[
    "SOH"
].to_numpy(dtype=float)

X_test = test_df[
    FEATURES
].to_numpy(dtype=float)

y_test = test_df[
    "SOH"
].to_numpy(dtype=float)


# ============================================================
# 6. STANDARDIZATION
# ============================================================
#
# CRITICAL:
#
# Mean and standard deviation are calculated ONLY
# from the training batteries.
#
# B0018 information must not enter preprocessing.
# ============================================================

mean_X = np.mean(
    X_train,
    axis=0
)

std_X = np.std(
    X_train,
    axis=0
)

std_X[
    std_X == 0
] = 1.0


X_train_scaled = (
    X_train - mean_X
) / std_X

X_test_scaled = (
    X_test - mean_X
) / std_X


# ============================================================
# 7. POLYNOMIAL FEATURES
# ============================================================

X_train_poly = np.column_stack([
    X_train_scaled,
    X_train_scaled ** 2
])

X_test_poly = np.column_stack([
    X_test_scaled,
    X_test_scaled ** 2
])


# ============================================================
# 8. ADD INTERCEPT
# ============================================================

X_train_design = np.column_stack([
    np.ones(
        len(X_train_poly)
    ),
    X_train_poly
])

X_test_design = np.column_stack([
    np.ones(
        len(X_test_poly)
    ),
    X_test_poly
])


# ============================================================
# 9. RIDGE REGRESSION
# ============================================================

lambda_ridge = 1.0

n_features = (
    X_train_design.shape[1]
)

I = np.eye(
    n_features
)

# Do not regularize intercept
I[0, 0] = 0


beta = np.linalg.solve(

    X_train_design.T
    @ X_train_design

    + lambda_ridge * I,

    X_train_design.T
    @ y_train
)


# ============================================================
# 10. PREDICT UNSEEN CELL
# ============================================================

y_pred_raw = (
    X_test_design
    @ beta
)


# ============================================================
# 11. PHYSICS-GUIDED LATENT HEALTH STATE
# ============================================================
#
# z_k represents slowly evolving underlying health.
#
# Measurement/model prediction may fluctuate,
# but latent degradation is constrained not to improve.
# ============================================================

alpha = 0.35

y_pred_physics = np.zeros_like(
    y_pred_raw
)

y_pred_physics[0] = (
    y_pred_raw[0]
)


for i in range(
    1,
    len(y_pred_raw)
):

    candidate = (

        alpha
        * y_pred_raw[i]

        + (1 - alpha)
        * y_pred_physics[i - 1]

    )

    y_pred_physics[i] = min(

        candidate,

        y_pred_physics[i - 1]

    )


# ============================================================
# 12. METRICS
# ============================================================

def metrics(true, pred):

    mae = np.mean(
        np.abs(
            true - pred
        )
    )

    rmse = np.sqrt(
        np.mean(
            (true - pred) ** 2
        )
    )

    ss_res = np.sum(
        (true - pred) ** 2
    )

    ss_tot = np.sum(
        (
            true
            - np.mean(true)
        ) ** 2
    )

    r2 = (
        1
        - ss_res / ss_tot
    )

    return (
        mae,
        rmse,
        r2
    )


raw_mae, raw_rmse, raw_r2 = metrics(
    y_test,
    y_pred_raw
)

phy_mae, phy_rmse, phy_r2 = metrics(
    y_test,
    y_pred_physics
)


# ============================================================
# 13. PRINT RESULTS
# ============================================================

print("\n========================================")
print("UNSEEN-CELL GENERALIZATION")
print("========================================")

print(
    f"Train: "
    f"{', '.join(TRAIN_CELLS)}"
)

print(
    f"Test : {TEST_CELL}"
)


print("\nNONLINEAR RIDGE")

print(
    f"MAE  : {raw_mae:.4f}"
)

print(
    f"RMSE : {raw_rmse:.4f}"
)

print(
    f"R²   : {raw_r2:.4f}"
)


print(
    "\nPHYSICS-GUIDED "
    "LATENT HEALTH"
)

print(
    f"MAE  : {phy_mae:.4f}"
)

print(
    f"RMSE : {phy_rmse:.4f}"
)

print(
    f"R²   : {phy_r2:.4f}"
)


# ============================================================
# 14. SAVE RESULTS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

os.makedirs(
    "figures",
    exist_ok=True
)


results = pd.DataFrame({

    "cell":
        TEST_CELL,

    "cycle":
        test_df[
            "cycle"
        ].to_numpy(),

    "measured_SOH":
        y_test,

    "ridge_prediction":
        y_pred_raw,

    "physics_guided_prediction":
        y_pred_physics

})


results.to_csv(

    "results/"
    "B0018_cross_cell_predictions.csv",

    index=False
)


# ============================================================
# 15. PROFESSOR-FACING FIGURE
# ============================================================

test_cycles = test_df[
    "cycle"
].to_numpy()


plt.figure(
    figsize=(10, 6)
)


plt.plot(

    test_cycles,

    y_test,

    linewidth=2,

    label="Measured SOH"
)


plt.plot(

    test_cycles,

    y_pred_raw,

    linestyle="--",

    label="Cross-Cell Nonlinear Ridge"
)


plt.plot(

    test_cycles,

    y_pred_physics,

    linewidth=2,

    label="Physics-Guided Latent Health"
)


plt.xlabel(
    "Discharge Cycle"
)

plt.ylabel(
    "SOH (%)"
)


plt.title(
    "Unseen-Cell Battery Health Estimation\n"
    "Train: B0005 + B0006 + B0007 | "
    "Test: B0018"
)


plt.legend()

plt.grid(True)

plt.tight_layout()


plt.savefig(

    "figures/"
    "cross_cell_B0018_soh.png",

    dpi=300
)


plt.show()


print(
    "\nCross-cell experiment complete."
)

print(
    "Figure saved:"
    "\nfigures/cross_cell_B0018_soh.png"
)

print(
    "\nResults saved:"
    "\nresults/B0018_cross_cell_predictions.csv"
)