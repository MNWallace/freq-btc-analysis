import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Fetch data (price and volume) from CoinGecko
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {
    "vs_currency": "usd",
    "days": "60"
}
response = requests.get(url, params=params)
data = response.json()

# Extract prices and volumes
prices = data['prices']
volumes = data['total_volumes']

# Convert to DataFrame
df_price = pd.DataFrame(prices, columns=['timestamp', 'price'])
df_volume = pd.DataFrame(volumes, columns=['timestamp', 'volume'])

# Convert timestamps to datetime
df_price['timestamp'] = pd.to_datetime(df_price['timestamp'], unit='ms')
df_volume['timestamp'] = pd.to_datetime(df_volume['timestamp'], unit='ms')

# Ensure same timestamps (they should be aligned but just in case)
df = pd.merge(df_price, df_volume, on='timestamp')

# Extract values
price_vals = df['price'].values
volume_vals = df['volume'].values
N = len(price_vals)
T = 1.0  # Sampling interval in hours

# FFT for price
fft_price = np.fft.fft(price_vals)
freqs = np.fft.fftfreq(N, d=T)
power_price = np.abs(fft_price)

# FFT for volume
fft_volume = np.fft.fft(volume_vals)
power_volume = np.abs(fft_volume)

# Positive frequencies mask
pos_mask = freqs > 0
freqs_pos = freqs[pos_mask]

# Zoom frequency range 0 to 0.2 cph
freq_zoom_mask = (freqs_pos <= 0.2)
freqs_zoom = freqs_pos[freq_zoom_mask]

power_price_zoom = power_price[pos_mask][freq_zoom_mask]
power_volume_zoom = power_volume[pos_mask][freq_zoom_mask]

# Top 5 strongest cycles helper function
def print_top_cycles(freqs_arr, power_arr, label):
    top_indices = power_arr.argsort()[-5:][::-1]
    print(f"\nTop 5 strongest {label} cycles (0 to 0.2 cph):")
    for idx in top_indices:
        freq = freqs_arr[idx]
        period = 1 / freq if freq != 0 else float('inf')
        print(f"Frequency: {freq:.5f} cph, Period: {period:.1f} h (~{period/24:.2f} days), Magnitude: {power_arr[idx]:.2f}")

print_top_cycles(freqs_zoom, power_price_zoom, "price")
print_top_cycles(freqs_zoom, power_volume_zoom, "volume")

# Plot time series and power spectra
plt.figure(figsize=(16, 10))

plt.subplot(3,1,1)
plt.plot(df['timestamp'], price_vals, color='orange')
plt.title('Bitcoin Price (Past 60 Days)')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(df['timestamp'], volume_vals, color='blue')
plt.title('Bitcoin Volume (Past 60 Days)')
plt.xlabel('Time')
plt.ylabel('Volume (USD)')
plt.grid(True)

plt.subplot(3,1,3)
plt.stem(freqs_zoom, power_price_zoom, linefmt='C1-', markerfmt='C1o', basefmt='C1-', label='Price')
plt.stem(freqs_zoom, power_volume_zoom, linefmt='C0-', markerfmt='C0o', basefmt='C0-', label='Volume')
plt.title('Fourier Power Spectrum (0 to 0.2 cycles/hour)')
plt.xlabel('Frequency (cycles/hour)')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()