import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import sounddevice as sd
from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt

sound, sample_rate = librosa.load('guitar.wav', sr=None)
#sd.play(sound, sample_rate)
#sd.wait()

start_time = 1
end_time = 1.1

#plot the waveform
plt.figure(figsize=(14, 5))
plt.plot(np.linspace(0, len(sound) / sample_rate, num=len(sound)), sound)
plt.xlim(start_time, end_time)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (dB)')
plt.title('Waveform for original signal')
plt.show()

#LFO
def lfo(frequency, length, sample_rate, depth=1.0, phase=0):
    # Calculate the number of samples
    num_samples = int(length * sample_rate)

    # Create a time array
    t = np.linspace(0, length, num_samples, endpoint=False)

    # Generate LFO signal using a sine wave
    lfo_signal = depth * np.sin(2 * np.pi * frequency * t + phase)

    return lfo_signal

#Delay
def dynamic_delay(audio_signal, delay_ms, lfo_signal, sample_rate, feedback=0.0):
    # Initialize the delay buffer
    max_delay_ms = np.max(delay_ms + lfo_signal)
    max_delay_samples = int(sample_rate * max_delay_ms / 1000)
    delay_buffer = np.zeros(max_delay_samples)
    output_signal = np.zeros_like(audio_signal)

    for i in range(len(audio_signal)):
        # Calculate the current delay in samples
        current_delay_samples = int((delay_ms + lfo_signal[i]) * sample_rate / 1000)

        # Make sure the delay is within the range of the delay buffer
        current_delay_samples = np.clip(current_delay_samples, 1, max_delay_samples - 1)

        # Read the delayed sample from the delay buffer using linear interpolation
        fraction = current_delay_samples % 1
        delayed_sample = (1 - fraction) * delay_buffer[int(current_delay_samples - 1)] + fraction * delay_buffer[
            int(current_delay_samples)]

        # Write the current input sample to the delay buffer
        delay_buffer = np.roll(delay_buffer, -1)
        delay_buffer[-1] = audio_signal[i] + delayed_sample * feedback

        # Store the output sample·
        output_signal[i] = delayed_sample

    return output_signal


#VCA Mix
def vca_mix(signal_a, signal_b, gain_a=1.0, gain_b=1.0):
    len_a = len(signal_a)
    len_b = len(signal_b)
    if len_a > len_b:
        signal_b = np.pad(signal_b, (0, len_a - len_b), 'constant')
    elif len_b > len_a:
        signal_a = np.pad(signal_a, (0, len_b - len_a), 'constant')

    # Apply the gains to each signal
    signal_a = signal_a * gain_a
    signal_b = signal_b * gain_b

    # Mix the signals
    mixed_signal = signal_a + signal_b

    return mixed_signal

#Chorus effector
def chorus(audio_signal, sample_rate, lfo_freq, lfo_depth, base_delay_ms, feedback, mix):
    # Generate the LFO signal
    lfo_length = len(audio_signal) / sample_rate
    lfo_signal = lfo(lfo_freq, lfo_length, sample_rate, lfo_depth)

    # Apply the dynamic delay to the input signal
    delayed_signal = dynamic_delay(audio_signal, base_delay_ms, lfo_signal, sample_rate, feedback)

    # Mix the original (dry) and the delayed (wet) signals
    wet_signal = delayed_signal * mix
    dry_signal = audio_signal * (1 - mix)
    chorus_signal = dry_signal + wet_signal

    return chorus_signal

#Example
lfo_freq = 0.6  # LFO frequency in Hz
lfo_depth = 5  # LFO depth in milliseconds
base_delay_ms = 20  # Base delay time in milliseconds
feedback = 0.3  # Feedback amount
mix = 0.9  # Wet/dry mix ratio

chorus_signal = chorus(sound, sample_rate, lfo_freq, lfo_depth, base_delay_ms, feedback, mix)
mix_signal = vca_mix(sound,chorus_signal)
#sd.play(mix_signal,sample_rate)
#sd.wait()
#write("mix_signal_g_0.9.wav", sample_rate, mix_signal)

#plot the waveform for chorus signal
plt.figure(figsize=(14, 5))
plt.plot(np.linspace(0, len(chorus_signal) / sample_rate, num=len(chorus_signal)), chorus_signal)
plt.xlim(start_time, end_time)  # 设置x轴的范围为0到0.1秒
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Waveform for chorus signal')
plt.show()

#plot the waveform for mix signal
plt.figure(figsize=(14, 5))
plt.plot(np.linspace(0, len(mix_signal) / sample_rate, num=len(mix_signal)), mix_signal)
plt.xlim(start_time, end_time)  # 设置x轴的范围为0到0.1秒
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Waveform for mixed signal')
plt.show()