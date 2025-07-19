import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Fetch data (60 days, hourly)
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {
    "vs_currency": "usd",
    "days": "60"
}
response = requests.get(url, params=params)
data = response.json()

prices = data['prices']
df = pd.DataFrame(prices, columns=['timestamp', 'price'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# 1. Extract price and ensure uniform sampling
prices = df['price'].values
N = len(prices)  # Number of data points
T = 1.0  # Sampling interval in hours (CoinGecko hourly data)

# 2. Compute FFT
fft_vals = np.fft.fft(prices)
fft_freq = np.fft.fftfreq(N, d=T)

# 3. Take magnitude (power spectrum)
power = np.abs(fft_vals)

# 4. Only plot positive frequencies (real signal symmetry)
pos_mask = fft_freq > 0
freqs = fft_freq[pos_mask]
powers = power[pos_mask]

# Zoom frequency range from 0 to 0.2 cycles/hour
freq_zoom_mask = (freqs <= 0.2)
freqs_zoom = freqs[freq_zoom_mask]
powers_zoom = powers[freq_zoom_mask]

# Find indices of top 5 peaks by magnitude within zoomed range
top_indices_zoom = powers_zoom.argsort()[-5:][::-1]

print("5 strongest price cycles (0 to 0.2 cph):")
for idx in top_indices_zoom:
    freq = freqs_zoom[idx]
    period = 1 / freq if freq != 0 else float('inf')
    magnitude = powers_zoom[idx]
    print(f"Frequency: {freq:.5f} cycles/hour, Period: {period:.1f} hours (~{period/24:.2f} days), Magnitude: {magnitude:.2f}")

plt.figure(figsize=(14, 6))

plt.subplot(2,1,1)
plt.plot(df['timestamp'], prices, color='orange')
plt.title('Bitcoin Price (Past 60 Days)')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.grid(True)

plt.subplot(2,1,2)
plt.stem(freqs_zoom, powers_zoom, linefmt='C0-', markerfmt='C0o', basefmt='C0-')
plt.title('Fourier Power Spectrum (0 to 0.2 cycles/hour)')
plt.xlabel('Frequency (cycles/hour)')
plt.ylabel('Magnitude')
plt.grid(True)

plt.tight_layout()
plt.show()