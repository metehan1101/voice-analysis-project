from tkinter import Tk, filedialog
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Dosya seçme penceresi
root = Tk()
root.withdraw()

audio_path = filedialog.askopenfilename(
    title="Select a WAV file",
    filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
)

print("Selected file:", audio_path)

if not audio_path:
    print("No file selected.")
    exit()

# Ses dosyasını yükle
y, sr = librosa.load(audio_path, sr=None)

# Waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("Voice Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# Frequency spectrum
fft = np.fft.fft(y)
freq = np.fft.fftfreq(len(fft), d=1/sr)

plt.figure(figsize=(10, 4))
plt.plot(freq[:len(freq)//2], np.abs(fft[:len(fft)//2]))
plt.title("Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.tight_layout()
plt.show()

# Spectrogram oluşturma
D = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
plt.colorbar(label='Intensity (dB)')
plt.title('Spectrogram')
plt.tight_layout()
plt.show()

# Pitch detection
f0, voiced_flag, voiced_probs = librosa.pyin(
    y,
    fmin=70,
    fmax=300
)

valid_f0 = f0[~np.isnan(f0)]

plt.figure(figsize=(10, 4))
plt.plot(f0, label='Estimated Pitch (Hz)')
plt.title('Pitch Contour')
plt.xlabel('Frame')
plt.ylabel('Frequency (Hz)')
plt.ylim(0, 320)
plt.legend()
plt.tight_layout()
plt.show()

print("\n--- Voice Analysis Summary ---")
if len(valid_f0) > 0:
    avg_pitch = np.mean(valid_f0)
    pitch_std = np.std(valid_f0)

    print(f"Average pitch: {avg_pitch:.2f} Hz")
    print(f"Pitch variation: {pitch_std:.2f} Hz")

    if avg_pitch < 120:
        print("Voice tendency: relatively deep")
    elif avg_pitch < 180:
        print("Voice tendency: medium")
    else:
        print("Voice tendency: relatively high")

    if pitch_std < 20:
        print("Pitch dynamics: stable")
    elif pitch_std < 50:
        print("Pitch dynamics: moderate")
    else:
        print("Pitch dynamics: expressive")
else:
    print("Pitch could not be estimated clearly.")

# ---------------------------
# Speech Energy Analysis
# ---------------------------
frame_length = 2048
hop_length = 512

rms = librosa.feature.rms(
    y=y,
    frame_length=frame_length,
    hop_length=hop_length
)[0]

plt.figure(figsize=(10, 4))
plt.plot(rms)
plt.title("Speech Energy (RMS)")
plt.xlabel("Frame")
plt.ylabel("Energy")
plt.tight_layout()
plt.show()

avg_energy = np.mean(rms)

print("\n--- Energy Analysis ---")
print(f"Average energy: {avg_energy:.5f}")

if avg_energy < 0.02:
    print("Energy level: low")
elif avg_energy < 0.05:
    print("Energy level: medium")
else:
    print("Energy level: high")

    # MFCC extraction
mfcc = librosa.feature.mfcc(
    y=y,
    sr=sr,
    n_mfcc=13
)

plt.figure(figsize=(10, 4))
librosa.display.specshow(
    mfcc,
    x_axis='time',
    sr=sr,
    cmap='viridis'
)
plt.colorbar()
plt.title("MFCC Features")
plt.tight_layout()
plt.show()


print("\n--- Artistic Voice Notes ---")

# Pitch yorumu
if len(valid_f0) > 0:
    if avg_pitch < 90:
        print("Your voice has a notably deep tonal profile.")
    elif avg_pitch < 140:
        print("Your voice falls in a balanced deep-medium range.")
    else:
        print("Your voice has a relatively brighter pitch profile.")

    if pitch_std < 10:
        print("Your pitch movement sounds controlled and stable.")
    elif pitch_std < 30:
        print("Your pitch movement shows natural variation.")
    else:
        print("Your pitch movement sounds expressive and dynamic.")

# Energy yorumu
if avg_energy < 0.02:
    print("Your delivery in this sample sounds soft and restrained.")
elif avg_energy < 0.05:
    print("Your delivery in this sample is balanced and clear.")
else:
    print("Your delivery in this sample sounds energetic and emphatic.")

# Kayıt süresi
duration = len(y) / sr
print(f"Estimated sample duration: {duration:.2f} seconds")

# Basit genel yorum
if len(valid_f0) > 0 and avg_pitch < 90 and avg_energy >= 0.02:
    print("This sample gives the impression of a deep and controlled speaking style.")
elif len(valid_f0) > 0 and avg_pitch < 140:
    print("This sample gives the impression of a balanced and natural speaking style.")
else:
    print("This sample shows a distinctive vocal profile worth exploring further.")

# ---------------------------
# Duration, Pause, and Voice Presence Analysis
# ---------------------------

duration = len(y) / sr

# Pitch üzerinden voiced/unvoiced hesaplama
voiced_frames = np.sum(~np.isnan(f0))
total_frames = len(f0)

if total_frames > 0:
    unvoiced_frames = total_frames - voiced_frames
    voiced_ratio = voiced_frames / total_frames
    unvoiced_ratio = unvoiced_frames / total_frames
else:
    unvoiced_frames = 0
    voiced_ratio = 0
    unvoiced_ratio = 0

# RMS energy üzerinden pause analizi
pause_threshold = 0.005
pause_frames = np.sum(rms < pause_threshold)

if len(rms) > 0:
    pause_ratio = pause_frames / len(rms)
else:
    pause_ratio = 0

print("\n--- Timing & Voice Presence Analysis ---")
print(f"Sample duration: {duration:.2f} seconds")
print(f"Voiced frames: {voiced_frames}")
print(f"Unvoiced frames: {unvoiced_frames}")
print(f"Voiced ratio: {voiced_ratio:.2%}")
print(f"Unvoiced ratio: {unvoiced_ratio:.2%}")
print(f"Pause ratio: {pause_ratio:.2%}")

if pause_ratio < 0.15:
    print("Pause usage: very low")
elif pause_ratio < 0.35:
    print("Pause usage: balanced")
else:
    print("Pause usage: frequent")

if voiced_ratio > 0.60:
    print("Voice presence: strong")
elif voiced_ratio > 0.35:
    print("Voice presence: moderate")
else:
    print("Voice presence: limited")

print("TIMING BLOCK RAN")





