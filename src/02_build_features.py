from scipy.io import loadmat
import numpy as np
import pandas as pd

file_path = "data/raw/B0005.mat"

mat = loadmat(file_path)
battery = mat["B0005"]
cycles = battery["cycle"][0, 0]

rows = []

discharge_number = 0

for i in range(cycles.shape[1]):

    if cycles[0, i]["type"][0] != "discharge":
        continue

    discharge_number += 1

    data = cycles[0, i]["data"][0, 0]

    voltage = data["Voltage_measured"][0]
    current = data["Current_measured"][0]
    temperature = data["Temperature_measured"][0]
    time = data["Time"][0]

    capacity = float(data["Capacity"][0, 0])
    duration_min = (time[-1] - time[0]) / 60

    T_initial = temperature[0]
    T_max = np.max(temperature)
    delta_T = T_max - T_initial

    peak_index = np.argmax(temperature)
    time_to_peak_min = (time[peak_index] - time[0]) / 60

    if time_to_peak_min > 0:
        heating_rate = delta_T / time_to_peak_min
    else:
        heating_rate = np.nan

    V_mean = np.mean(voltage)
    V_min = np.min(voltage)

    rows.append({
    "cycle": discharge_number,
    "capacity_Ah": capacity,
    "duration_min": duration_min,
    "T_initial_C": T_initial,
    "T_max_C": T_max,
    "delta_T_C": delta_T,
    "heating_rate_C_per_min": heating_rate,
    "V_mean": V_mean,
    "V_min": V_min
})

df = pd.DataFrame(rows)

print(df.head())
print("\nShape:", df.shape)

Q_ref = df["capacity_Ah"].iloc[0]

df["SOH"] = (
    df["capacity_Ah"] / Q_ref
) * 100

eol_candidates = df.loc[df["SOH"] <= 80, "cycle"]

if len(eol_candidates) > 0:
    eol_cycle = int(eol_candidates.iloc[0])
else:
    eol_cycle = int(df["cycle"].iloc[-1])

df["RUL"] = np.maximum(
    eol_cycle - df["cycle"],
    0
)

print("\nEOL cycle:", eol_cycle)
print(df.head())
print(df.tail())

output_path = "data/processed/B0005_features.csv"

df.to_csv(output_path, index=False)

print("\nSaved:", output_path)

