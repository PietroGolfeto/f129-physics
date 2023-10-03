import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

from math import floor, log10, sqrt

def sig_figs(x: float, precision: int):
    x = float(x)
    precision = int(precision)

    return round(x, -int(floor(log10(abs(x)))) + (precision - 1))

V1 = [0.56, 1.15, 1.68, 2.13, 2.62, 3.11, 3.72, 4.16, 4.81, 5.27, 5.70, 6.18,
      6.69, 7.29, 7.64, 8.25, 8.82, 9.25, 9.74, 10.21, 10.76, 11.31, 11.81, 12.23, 12.75]

mA1 = [2.5, 5.2, 7.6, 9.7, 12.0, 14.2, 17.1, 19.1, 22.1, 24.3, 26.3,
       28.6, 31.0, 33.9, 35.7, 38.5, 41.4, 43.5, 46.0, 48.4, 51.2, 53.9, 56.6, 58.8, 61.5]

V2 = [-0.56, -1.08, -1.59, -2.17, -2.7, -3.13, -3.66, -4.24, -4.76, -5.24, -5.7, -
      6.22, -6.73, -7.19, -7.7, -8.3, -8.77, -9.28, -9.81, -10.46, -10.76, -11.43, -11.71, -12.22, -12.78]

mA2 = [-2.5, -4.9, -7.2, -9.9, -12.3, -14.3, -16.7, -19.4, -21.8, -24.1, -26.3, -
       28.7, -31.1, -33.3, -35.7, -38.8, -41.0, -43.6, -46.2, -49.5, -51.0, -54.5, -56.0, -58.5, -61.4]

dadosV1 = np.array(V1).reshape(-1, 1)
dadosmA1 = np.array(mA1)
dadosV2 = np.array(V2).reshape(-1, 1)
dadosmA2 = np.array(mA2)

DC_voltage_scale = "20 V"
voltage_resolution = 0.01 # 10mV = 0.01 V
voltage_percentile = 0.008
voltage_resolution_multiple = 5
voltage_precision = "+- (voltage_percentile * data + voltage_resolution_multiple * voltage_resolution)" # 0.8% + 5 * resolution

DC_current_scale = "200 mA"
current_resolution = 0.1 # 100 uA = 100 * 10^-6 = 10^-4 in mA = 0.1 mA
current_percentile = 0.02
current_resolution_multiple = 2
current_precision = "+- (current_percentile * data + current_resolution_multiple * current_resolution)" # 2% + 2 * resolution

V1_uncertainty, V2_uncertainty, mA1_uncertainty, mA2_uncertainty = [], [], [], []
V1_final, V2_final, mA1_final, mA2_final = [], [], [], []

voltage_medition_uncertainty = (voltage_resolution / 2) / sqrt(3)
current_medition_uncertainty = (current_resolution / 2) / sqrt(3)

# Calculate precision uncertainty and combined uncertainty with 1 significant digit
for v in V1:
      precision_uncertainty = (voltage_percentile * v + voltage_resolution_multiple * voltage_resolution) / sqrt(3)
      uncertainty = (sqrt(voltage_medition_uncertainty * voltage_medition_uncertainty + precision_uncertainty * precision_uncertainty))
      sigfig_uncertainty = sig_figs(uncertainty, 1)
      V1_uncertainty.append(sigfig_uncertainty)
      v = "{:.2f}".format(v)
      V1_final.append(f"  {v} += {sigfig_uncertainty}  ")
for v in V2:
      precision_uncertainty = (voltage_percentile * abs(v) + voltage_resolution_multiple * voltage_resolution) / sqrt(3)
      uncertainty = (sqrt(voltage_medition_uncertainty * voltage_medition_uncertainty + precision_uncertainty * precision_uncertainty))
      sigfig_uncertainty = sig_figs(uncertainty, 1)
      V2_uncertainty.append(sigfig_uncertainty)
      v = "{:.2f}".format(v)
      V2_final.append(f"  {v} += {sigfig_uncertainty}  ")      
for data in mA1:
      precision_uncertainty = (current_percentile * data + current_resolution_multiple * current_resolution) / sqrt(3)
      uncertainty = (sqrt(current_medition_uncertainty * current_medition_uncertainty + precision_uncertainty * precision_uncertainty))
      sigfig_uncertainty = sig_figs(uncertainty, 1)      
      mA1_uncertainty.append(sigfig_uncertainty)
      mA1_final.append(f"  {data} += {sigfig_uncertainty}  ")
for data in mA2:
      precision_uncertainty = (current_percentile * abs(data) + current_resolution_multiple * current_resolution) / sqrt(3)
      uncertainty = (sqrt(current_medition_uncertainty * current_medition_uncertainty + precision_uncertainty * precision_uncertainty))
      sigfig_uncertainty = sig_figs(uncertainty, 1)      
      mA2_uncertainty.append(sigfig_uncertainty)
      mA2_final.append(f"  {data} += {sigfig_uncertainty}  ")

model1 = LinearRegression().fit(dadosV1, dadosmA1)
model2 = LinearRegression().fit(dadosV2, dadosmA2)

# Predicted values
pred_mA1 = model1.predict(dadosV1.reshape(-1, 1))
pred_mA2 = model2.predict(dadosV2.reshape(-1, 1))

# Calculate linear coefficients (slopes)
slope_model1 = model1.coef_[0]
slope_model2 = model2.coef_[0]

# Calculate resistances (Ohms)
resistance_model1 = 1000 / slope_model1  # Convert mA to A
resistance_model2 = 1000 / slope_model2  # Convert mA to A

# Plotting the data points and regression lines
plt.figure(figsize=(10, 5))

# Model 1
plt.subplot(1, 2, 1)
plt.scatter(dadosV1, dadosmA1, color='blue', label='Actual')
plt.plot(dadosV1, pred_mA1, color='red', linewidth=2, label='Predicted')
plt.title('Model 1')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (mA)')
plt.legend()

# Model 2
plt.subplot(1, 2, 2)
plt.scatter(dadosV2, dadosmA2, color='blue', label='Actual')
plt.plot(dadosV2, pred_mA2, color='red', linewidth=2, label='Predicted')
plt.title('Model 2')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (mA)')
plt.legend()

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()

print(f"Model 1 - Linear Coefficient (Slope): {slope_model1}")
print(f"Model 1 - Resistance (Ohms): {resistance_model1} Ohms")

print(f"Model 2 - Linear Coefficient (Slope): {slope_model2}")
print(f"Model 2 - Resistance (Ohms): {resistance_model2} Ohms")

print(f"Voltage medition uncertainty (V): {voltage_medition_uncertainty}")
print(f"Current medition uncertainty (mA): {current_medition_uncertainty}")

# Create a DataFrame of positive values
positive = pd.DataFrame({
    'Tensao (V)': V1_final,
    'Corrente (mA)': mA1_final,
})

pd.set_option('display.max_colwidth', 50)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.width', 1000)
print("\n             Medicao 1")
print(positive.to_string(index=False))

# Create a DataFrame of negative values
negative = pd.DataFrame({
    'Tensao (V)': V2_final,
    'Corrente (mA)': mA2_final,
})

pd.set_option('display.max_colwidth', 50)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.width', 1000)
print("\n\n             Medicao 2")
print(negative.to_string(index=False))
print()