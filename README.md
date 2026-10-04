# PhyTwin-Battery

## Physics-Guided Battery Health Estimation Under Cell-to-Cell Variation

PhyTwin-Battery is a research prototype for lithium-ion battery State-of-Health (SOH) estimation using electrothermal degradation signals and physics-guided machine learning.

The project investigates a central question:

> **Can battery-health models trained on multiple cells generalize to an unseen battery while maintaining physically meaningful degradation behavior?**

The current version uses the NASA lithium-ion battery aging dataset and focuses on SOH estimation, cross-cell generalization, and latent health-state modeling as a foundation for adaptive battery digital twins.

---

## Research Motivation

Battery degradation depends on interacting electrical, thermal, and usage conditions. Machine-learning models can achieve strong performance within known operating conditions, but reliable deployment requires them to generalize across cells and degradation trajectories.

This project focuses on four challenges:

- Cell-to-cell generalization
- Physically meaningful health estimation
- Electrothermal degradation representation
- Robust prediction under domain shift

The long-term objective is to develop an adaptive battery digital twin capable of updating its internal health state from operational measurements and estimating SOH and Remaining Useful Life (RUL) with uncertainty.

---

## Dataset

The current experiments use NASA lithium-ion battery aging cells:

- B0005
- B0006
- B0007
- B0018

Voltage, current, temperature, time, and measured discharge capacity are extracted from each discharge cycle.

Raw NASA `.mat` files are not included in this repository.

Place the downloaded battery files in:

```text
data/raw/
├── B0005.mat
├── B0006.mat
├── B0007.mat
└── B0018.mat
```

---

## Capacity Validation

Discharge capacity was independently estimated from measured current using Coulomb counting:

\[
Q = \frac{1}{3600}\int |I(t)|dt
\]

For the first B0005 discharge cycle:

- NASA reported capacity: **1.8565 Ah**
- Independently integrated capacity: **1.8622 Ah**
- Difference: approximately **0.31%**

This provides an independent consistency check between the raw current signal and the reported capacity.

---

## State of Health

SOH is defined relative to each cell's initial measured discharge capacity:

\[
SOH_k =
\frac{Q_k}{Q_{ref}}
\times 100
\]

For B0005:

- Initial capacity: approximately **1.8565 Ah**
- Final capacity: approximately **1.3251 Ah**
- Final SOH: approximately **71.4%**
- First observed 80% SOH crossing: **cycle 101**

The 80% threshold is treated as an application-level End-of-Life reference rather than physical battery failure.

---

## Electrothermal Features

One feature vector is constructed for each discharge cycle:

\[
x_k =
[
t_{discharge},
T_0,
T_{max},
\Delta T,
r_T,
\bar{V},
V_{min}
]
\]

where:

- \(t_{discharge}\): discharge duration
- \(T_0\): initial temperature
- \(T_{max}\): maximum temperature
- \(\Delta T = T_{max}-T_0\)
- \(r_T\): average heating rate
- \(\bar{V}\): mean discharge voltage
- \(V_{min}\): minimum voltage

The thermal features are motivated by the simplified battery heat balance:

\[
mC_p\frac{dT}{dt}
=
\dot{q}_{gen}
-
hA(T-T_{amb})
\]

with irreversible Joule heating approximately related to:

\[
\dot{q}_{irr}\approx I^2R
\]

These features therefore provide observable signatures associated with evolving electrothermal battery behavior.

---

## Modeling

### 1. Linear Baseline

An ordinary least-squares regression model provides an interpretable baseline.

A chronological 80/20 split is used instead of random shuffling so that later degradation cycles are predicted from earlier-cycle information.

### 2. Nonlinear Electrothermal Model

Standardized electrothermal features and their second-order terms are used with ridge regularization:

\[
\min_{\beta}
\|X\beta-y\|_2^2
+
\lambda\|\beta\|_2^2
\]

This captures nonlinear relationships between measurable electrothermal behavior and battery SOH.

### 3. Physics-Guided Latent Health State

A latent health state is introduced to represent slowly evolving degradation.

The model combines the current data-driven prediction with the previous health state and imposes a simplified non-increasing degradation constraint.

This is a physics-guided prototype rather than a full Physics-Informed Neural Network.

---

## Experiment 1 — Chronological SOH Prediction

The first experiment uses B0005 and evaluates prediction on later degradation cycles.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Baseline | 0.8341 | 0.9059 | 0.4267 |
| Nonlinear Ridge | 0.6679 | **0.8363** | **0.5114** |
| Physics-Guided Latent State | **0.5911** | 0.8981 | 0.4365 |

The nonlinear model improves RMSE compared with the linear baseline.

The latent-state constraint reduces average absolute error but increases RMSE, indicating a trade-off between smooth physical degradation behavior and tracking short-term measured capacity variation.

---

## Experiment 2 — Unseen-Cell Generalization

To investigate cell-to-cell generalization, the model is trained on:

```text
B0005 + B0006 + B0007
```

and evaluated on the completely held-out cell:

```text
B0018
```

B0018 is not used for model fitting or preprocessing-statistic estimation.

### Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Nonlinear Ridge | **2.5611** | **3.1020** | **0.8609** |
| Physics-Guided Latent State | 2.9480 | 3.6915 | 0.8030 |

The nonlinear electrothermal model achieves:

\[
\boxed{R^2 = 0.8609}
\]

on the unseen B0018 cell.

The result provides evidence that electrothermal degradation features can transfer useful health information across cells within this dataset.

However, the simplified monotonic latent-state constraint reduces cross-cell prediction accuracy.

---

## Key Research Finding

The experiments reveal an important modeling trade-off.

A nonlinear electrothermal model provides promising cross-cell SOH estimation, while imposing a universal monotonic health-state constraint does not automatically improve generalization.

This motivates a more physically meaningful degradation transition model:

\[
z_{k+1}
=
z_k-\Delta z_k
\]

with:

\[
\Delta z_k =
f_\theta(
T_k,
I_k,
V_k,
\text{operating history}
)
\]

and:

\[
\Delta z_k \geq 0
\]

Rather than simply forcing predicted SOH to decrease, future models should learn degradation increments from battery stressors and operating history.

---

## Limitations

The current prototype has several important limitations.

The dataset contains a limited number of cells, and the current cross-cell experiment is evaluated on one held-out battery.

Several features, particularly full-discharge duration and heating rate, contain information strongly related to discharge capacity. Therefore, the current experiment should be interpreted as full-discharge retrospective SOH estimation rather than an online early-cycle estimator.

The current physics-guided model uses a simplified monotonic latent-state constraint and does not yet explicitly represent electrochemical degradation mechanisms.

Uncertainty quantification and domain adaptation have not yet been incorporated.

---

## Next Research Direction

The next stage of PhyTwin-Battery will focus on:

1. Leave-one-cell-out validation across multiple batteries
2. Partial-discharge and early-cycle health indicators
3. Stress-aware degradation-state transition modeling
4. Physics-guided neural state-space models
5. Domain adaptation across cells and operating conditions
6. Prediction uncertainty and confidence intervals
7. Remaining Useful Life prognostics
8. Sequential measurement-based digital-twin state updating

The intended digital-twin formulation is:

\[
z_{k+1}
=
f_\theta(z_k,u_k)
\]

\[
y_k =
g_\theta(z_k,u_k)
\]

where \(z_k\) represents latent battery health and \(u_k\) represents measurable operating conditions.

Incoming battery measurements can then continuously update the virtual health state.

---

## Repository Structure

```text
PhyTwin-Battery/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── 01_inspect_data.py
│   ├── 02_build_features.py
│   ├── 03_train_baseline.py
│   ├── 04_physics_guided_model.py
│   └── 05_cross_cell_generalization.py
│
├── figures/
│   ├── soh_linear_baseline.png
│   ├── soh_prediction_error.png
│   ├── physics_guided_soh.png
│   └── cross_cell_B0018_soh.png
│
├── results/
│   ├── B0005_linear_predictions.csv
│   ├── physics_guided_predictions.csv
│   └── B0018_cross_cell_predictions.csv
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Running the Project

Create and activate a Python environment, install the required packages, place the NASA battery files in `data/raw/`, and run:

```bash
python src/01_inspect_data.py
python src/02_build_features.py
python src/03_train_baseline.py
python src/04_physics_guided_model.py
python src/05_cross_cell_generalization.py
```

---

## Requirements

```text
numpy
pandas
scipy
matplotlib
```

---

## Current Status

**Research prototype / ongoing work**

The current implementation establishes the data-processing, electrothermal feature-engineering, SOH-estimation, physics-guided state-modeling, and cross-cell validation pipeline.

The next phase will extend this foundation toward adaptive physics-guided battery digital twins for robust SOH and RUL estimation under cell and operating-condition domain shift.
