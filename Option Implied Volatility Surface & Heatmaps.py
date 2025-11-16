import yfinance as yf
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------
# 1. Input Stock
# ----------------------------
symbol = input("Enter stock ticker (example: AAPL): ").upper()
tk = yf.Ticker(symbol)

# ----------------------------
# 2. Get All Expiries
# ----------------------------
expiries = tk.options
print("\nAvailable Expiries:", expiries)

# ----------------------------
# 3. Download Option Chains (Calls + Puts)
# ----------------------------
surface_data = []

for exp in expiries:
    try:
        chain = tk.option_chain(exp)

        # Calls
        calls = chain.calls.copy()
        if "srike" in calls.columns: calls.rename(columns={"srike":"strike"}, inplace=True)
        if "impliedVolatility" in calls.columns:
            calls["expiry"] = exp
            calls["type"] = "call"
            surface_data.append(calls)

        # Puts
        puts = chain.puts.copy()
        if "srike" in puts.columns: puts.rename(columns={"srike":"strike"}, inplace=True)
        if "impliedVolatility" in puts.columns:
            puts["expiry"] = exp
            puts["type"] = "put"
            surface_data.append(puts)

    except:
        continue

# ----------------------------
# 4. Build Master DataFrame
# ----------------------------
data = pd.concat(surface_data, ignore_index=True)
data["T"] = (pd.to_datetime(data["expiry"]) - pd.Timestamp.today()).dt.days / 365  # in years
data["IV"] = data["impliedVolatility"]
data = data.dropna(subset=["IV", "strike", "T"])
data = data[data["T"] > 0]

# ----------------------------
# 5. Function to Build Surface
# ----------------------------
def build_surface(df, grid_points=5):
    points = df[["strike", "T"]].values
    values = df["IV"].values
    strike_grid = np.linspace(df["strike"].min(), df["strike"].max(), grid_points)
    T_grid = np.linspace(df["T"].min(), df["T"].max(), grid_points)
    Strike, T = np.meshgrid(strike_grid, T_grid)
    IV_surface = griddata(points, values, (Strike, T), method="linear")
    IV_surface = np.nan_to_num(IV_surface, nan=np.nanmean(values))
    return Strike, T, IV_surface

# Separate calls and puts
call_data = data[data["type"]=="call"]
put_data = data[data["type"]=="put"]

Strike_call, T_call, IV_call = build_surface(call_data, grid_points=5)
Strike_put, T_put, IV_put = build_surface(put_data, grid_points=5)

# Convert time to expiry to months for axis labels
T_call_months = T_call * 12
T_put_months = T_put * 12

# ----------------------------
# 6. Plot 3D Surfaces Side by Side
# ----------------------------
fig = plt.figure(figsize=(16,7))

# Calls 3D
ax1 = fig.add_subplot(121, projection="3d")
ax1.plot_surface(Strike_call, T_call_months, IV_call, cmap="viridis", linewidth=0, antialiased=True)
ax1.set_title(f"{symbol} Calls IV Surface")
ax1.set_xlabel("Strike"); ax1.set_ylabel("Time to Expiry (Months)"); ax1.set_zlabel("IV")

# Puts 3D
ax2 = fig.add_subplot(122, projection="3d")
ax2.plot_surface(Strike_put, T_put_months, IV_put, cmap="plasma", linewidth=0, antialiased=True)
ax2.set_title(f"{symbol} Puts IV Surface")
ax2.set_xlabel("Strike"); ax2.set_ylabel("Time to Expiry (Months)"); ax2.set_zlabel("IV")

plt.tight_layout()
plt.show()

# ----------------------------
# 7. Plot 2D Heatmaps (Super Chunky Squares)
# ----------------------------
fig, axs = plt.subplots(1, 2, figsize=(16,6))

# Calls heatmap
im1 = axs[0].imshow(IV_call,
                    origin='lower',
                    aspect='auto',
                    extent=[Strike_call.min(), Strike_call.max(), T_call_months.min(), T_call_months.max()],
                    cmap='viridis')
axs[0].set_title(f"{symbol} Calls IV Heatmap")
axs[0].set_xlabel("Strike"); axs[0].set_ylabel("Time to Expiry (Months)")
fig.colorbar(im1, ax=axs[0], label="IV")

# Puts heatmap
im2 = axs[1].imshow(IV_put,
                    origin='lower',
                    aspect='auto',
                    extent=[Strike_put.min(), Strike_put.max(), T_put_months.min(), T_put_months.max()],
                    cmap='plasma')
axs[1].set_title(f"{symbol} Puts IV Heatmap")
axs[1].set_xlabel("Strike"); axs[1].set_ylabel("Time to Expiry (Months)")
fig.colorbar(im2, ax=axs[1], label="IV")

plt.tight_layout()
plt.show()
