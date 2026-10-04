from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

file_path = 'data/raw/B0005.mat'
mat = loadmat(file_path)

print(mat.keys())

battery = mat['B0005']

"""print ("Shape:", battery.shape)
print ("Data type:", battery.dtype)
print ("Fields:", battery.dtype.names)"""

cycles = battery['cycle'] [0,0]

"""print ("Cycles shape", cycles.shape)
print ("Cycles fields", cycles.dtype.names)"""

from collections import Counter

cycle_types = []

for i in range (cycles.shape[1]):
    cycle_type = cycles[0,i]['type'][0]
    cycle_types.append(cycle_type)

#print(Counter(cycle_types))

for i in range(cycles.shape[1]):
    if cycles[0, i]["type"][0] == "discharge":
        first_discharge = cycles[0, i]
        print("First discharge index:", i)
        break

discharge_data = first_discharge["data"][0, 0]

#print("Discharge fields:", discharge_data.dtype.names)

capacity = discharge_data["Capacity"][0, 0]

#print("First discharge capacity:", capacity)
#print("Capacity type:", type(capacity))

current = discharge_data["Current_measured"][0]
time = discharge_data["Time"][0]

#print("Current shape:", current.shape)
#print("Time shape:", time.shape)
#print("First 5 current values:", current[:5])
#print("First 5 time values:", time[:5])

capacity_calculated = np.trapezoid(-current, time) / 3600

#print("NASA capacity:", capacity)
#print("Calculated capacity:", capacity_calculated)
#print("Difference:", capacity_calculated - capacity)


capacities = []

for i in range(cycles.shape[1]):

    if cycles[0, i]["type"][0] == "discharge":

        data = cycles[0, i]["data"][0, 0]
        capacity_i = data["Capacity"][0, 0]

        capacities.append(capacity_i)

capacities = np.array(capacities)

#print("Number of discharge cycles:", len(capacities))
#print("First 5 capacities:", capacities[:5])
#print("Last 5 capacities:", capacities[-5:])



discharge_cycles = np.arange(1, len(capacities) + 1)

"""plt.figure(figsize=(8, 5))
plt.plot(discharge_cycles, capacities)

plt.xlabel("Discharge Cycle")
plt.ylabel("Capacity (Ah)")
plt.title("B0005 Capacity Degradation")

plt.grid(True)
plt.tight_layout()
plt.show()"""


Q_ref = capacities[0]

soh = (capacities / Q_ref) * 100

"""print("Reference capacity:", Q_ref)
print("First SOH:", soh[0])
print("Last SOH:", soh[-1])"""

eol_indices = np.where(soh <= 80)[0]

first_eol_index = eol_indices[0]
first_eol_cycle = discharge_cycles[first_eol_index]

"""print("First cycle at or below 80% SOH:", first_eol_cycle)
print("SOH at that cycle:", soh[first_eol_index])
print("Capacity at that cycle:", capacities[first_eol_index])"""

eol_cycle = first_eol_cycle

rul = np.maximum(eol_cycle - discharge_cycles, 0)

"""print("RUL at cycle 1:", rul[0])
print("RUL at cycle 50:", rul[49])
print("RUL at cycle 80:", rul[79])
print("RUL at cycle 100:", rul[99])
print("RUL at cycle 101:", rul[100])"""

discharge_records = []

for i in range(cycles.shape[1]):
    if cycles[0, i]["type"][0] == "discharge":
        discharge_records.append(cycles[0, i])

selected_cycles = [1, 80, 150]



plt.figure(figsize=(8, 5))

for k in selected_cycles:

    record = discharge_records[k - 1]
    data = record["data"][0, 0]

    voltage = data["Voltage_measured"][0]
    time = data["Time"][0]

    plt.plot(time / 60, voltage, label=f"Cycle {k}")

"""plt.xlabel("Time (min)")
plt.ylabel("Terminal Voltage (V)")
plt.title("Voltage Evolution with Battery Aging")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()"""

plt.figure(figsize=(8, 5))

for k in selected_cycles:

    record = discharge_records[k - 1]
    data = record["data"][0, 0]

    temperature = data["Temperature_measured"][0]
    time = data["Time"][0]

    plt.plot(time / 60, temperature, label=f"Cycle {k}")

plt.xlabel("Time (min)")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Evolution with Battery Aging")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

for k in selected_cycles:

    record = discharge_records[k - 1]
    data = record["data"][0, 0]

    temperature = data["Temperature_measured"][0]
    time = data["Time"][0]

    T_initial = temperature[0]
    T_max = np.max(temperature)
    delta_T = T_max - T_initial

    peak_index = np.argmax(temperature)
    time_to_peak = time[peak_index] / 60

    heating_rate = delta_T / time_to_peak

    print(f"\nCycle {k}")
    print("Initial temperature:", T_initial)
    print("Maximum temperature:", T_max)
    print("Temperature rise:", delta_T)
    print("Time to peak (min):", time_to_peak)
    print("Average heating rate (°C/min):", heating_rate)

delta_temps = []
heating_rates = []
time_to_peaks = []

for record in discharge_records:

    data = record["data"][0, 0]

    temperature = data["Temperature_measured"][0]
    time = data["Time"][0]

    T_initial = temperature[0]
    T_max = np.max(temperature)

    delta_T = T_max - T_initial

    peak_index = np.argmax(temperature)
    time_to_peak = time[peak_index] / 60

    heating_rate = delta_T / time_to_peak

    delta_temps.append(delta_T)
    heating_rates.append(heating_rate)
    time_to_peaks.append(time_to_peak)

print(len(delta_temps))
print(len(heating_rates))
print(len(soh))

corr_heating_soh = np.corrcoef(heating_rates, soh)[0, 1]

print("Correlation between heating rate and SOH:",
      corr_heating_soh)

plt.figure(figsize=(7, 5))

plt.scatter(soh, heating_rates)

plt.xlabel("SOH (%)")
plt.ylabel("Average Heating Rate (°C/min)")
plt.title("Heating Rate vs Battery SOH")

plt.grid(True)
plt.tight_layout()
plt.show()