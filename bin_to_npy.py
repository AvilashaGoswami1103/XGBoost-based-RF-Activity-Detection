import numpy as np

bin_file = r"C:\Users\RF-LAB\Desktop\Pipeline\New_Test_Data_Valid\DJI_phantom_4_pro_plus_5G_2of2.bin"
out_file = r"C:\Users\RF-LAB\Desktop\Pipeline\New_Test_Data_Valid_processed\DJI_phantom_4_pro_plus_5G_2of2.npy"

data = np.fromfile(bin_file, dtype=np.int16)
I = data[0::2].astype(np.float32) / 32768.0
Q = data[1::2].astype(np.float32) / 32768.0

# DC removal
I -= I.mean()
Q -= Q.mean()

# RMS normalization
rms = np.sqrt(np.mean(I**2 + Q**2))
if rms > 1e-10:
    I /= rms
    Q /= rms

# Save as complex64
z = (I + 1j * Q).astype(np.complex64)
np.save(out_file, z)

print(f"IQ pairs : {len(z):,}")
print(f"Saved → {out_file}")


