import torch
import numpy as np

'''
tensor = torch.load('results/bandpower_20260127_1514/data/sub-040_bandpower.pt')
print("Shape:", tensor.shape)           # [1081, 19, 5]
print("Window 0, Channel 0:", tensor[0,0])  # [delta, theta, alpha, beta, gamma]
print("Means across windows:", tensor.mean(0).mean(0))  # Avg power per band
'''

'''
tensor = torch.load('results/stft_20260129_2155/data/sub-001_stft.pt')
print("Shape:", tensor.shape)           # [1081, 19, 129, 7]
print("Window 0 shape:", tensor[0].shape)  # [19, 129, 7]
print("Freq range example:", tensor[0,0, :5, 0])  # First 5 freq bins, time 0
print("Power range:", tensor.min().item(), "→", tensor.max().item())
print("Window 0, Channel 0 mean power:", tensor[0,0].mean().item())

# Plot first spectrogram (Window 0, Channel 0)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.imshow(tensor[0, 0].numpy(), aspect='auto', cmap='viridis', origin='lower')
plt.title('STFT Spectrogram: Sub 001, Window 0, Channel 0')
plt.xlabel('Time bins (7 total)')
plt.ylabel('Frequency (129 bins, 0-64Hz)')
plt.colorbar(label='Log Power')
plt.savefig('stft_sample.png', dpi=150)
plt.show()
print("✅ Saved stft_sample.png")
'''

'''
tensor = torch.load('results/connectivity_20260129_2214/data/sub-001_connectivity.pt')
print(f"Shape: {tensor.shape}")  # [1081, 171]
print(f"Range: {tensor.min():.3f} → {tensor.max():.3f}")
print(f"Mean: {tensor.mean():.3f}")
print(f"Window 0 top 5 PLVs: {tensor[0].topk(5)[0]}")  # Strongest connections

# AD vs HC difference (if you have both)
ad_plv = torch.load('results/connectivity_20260129_2214/data/sub-001_connectivity.pt').mean()
hc_plv = torch.load('results/connectivity_20260129_2214/data/sub-040_connectivity.pt').mean()
print(f"AD mean PLV: {ad_plv:.3f}, HC mean PLV: {hc_plv:.3f}")
'''
