import numpy as np

# Assuming fft_freq, power computed as before:

# Mask positive frequencies only
pos_mask = fft_freq > 0
freqs = fft_freq[pos_mask]
powers = power[pos_mask]

# Find indices of top 5 peaks by magnitude
top_indices = powers.argsort()[-5:][::-1]

print("Top 5 cycles in your BTC price data:")
for idx in top_indices:
    freq = freqs[idx]
    period = 1 / freq if freq != 0 else float('inf')
    magnitude = powers[idx]
    print(f"Frequency: {freq:.5f} cycles/hour, Period: {period:.1f} hours, Magnitude: {magnitude:.2f}")
