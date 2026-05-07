import scipy.io
import numpy as np
import os
import glob

MAT_DIR    = r"C:\Users\RF-LAB\Desktop\Pipeline\New_Test_Data_Noise"
OUTPUT_DIR = r"C:\Users\RF-LAB\Desktop\Pipeline\New_Test_Data_Noise_processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

mat_files = sorted(glob.glob(os.path.join(MAT_DIR, "*.mat")))
print(f"Found {len(mat_files)} .mat file(s)")

for mat_path in mat_files:
    fname = os.path.splitext(os.path.basename(mat_path))[0]
    mat   = scipy.io.loadmat(mat_path)

    iq = mat['IQ_samples'].squeeze()  # shape: (8192000,) complex

    # Extract I and Q
    I = iq.real.astype(np.float32)
    Q = iq.imag.astype(np.float32)

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
    out_path = os.path.join(OUTPUT_DIR, f"{fname}_processed.npy")
    np.save(out_path, z)
    print(f"  [✓] {fname}  →  shape={z.shape}")

print(f"\nDone. {len(mat_files)} file(s) converted → {OUTPUT_DIR}")