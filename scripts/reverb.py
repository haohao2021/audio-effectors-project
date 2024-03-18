import numpy as np
import soundfile as sf

def early_reflections(input_sample, buffer, fs, n):

    # Convert delay times from milliseconds to samples
    delay_times = np.array([0, 0.015, 0.016, 0.017, 0.018, 0.020, 0.025, 0.030, 0.035, 0.040,
                        0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080, 0.085]) * fs
    delay_times = np.round(delay_times).astype(int) 

    gains = np.array([1, 0.85, -0.75, 0.65, 0.55, 0.45, -0.35, -0.25, -0.15, 0.15,
                  -0.25, 0.35, 0.45, 0.55, 0.65, -0.75, 0.85, 0.95, 1])


    # Update circular buffer with current sample
    buffer[n % len(buffer)] = input_sample

    # Initialize output sample
    output_sample = 0
    
    # Convert delay times from milliseconds to samples and ensure they are integers
    delay_times = np.fix(fs * np.array([0, 0.01277, 0.01283, 0.01293, 0.01333,
                                    0.01566, 0.02404, 0.02679, 0.02731, 0.02737,
                                    0.02914, 0.02920, 0.02981, 0.03389, 0.04518,
                                    0.04522, 0.04527, 0.05452, 0.06958])).astype(int)

    # Loop through all taps
    for tap in range(len(delay_times)):
        # Calculate the circular buffer index for the current tap
        index_tdl = (n - delay_times[tap] - 1) % len(buffer)
        # Add current tap with output
        output_sample += gains[tap] * buffer[index_tdl]

    return output_sample, buffer


def lpcf(input_sample, buffer, fs, n, delay, fb_gain, amp, rate, fb_lpf):

    # Calculate time in seconds for the current sample
    t = (n - 1) / fs
    # Apply LFO modulation to the delay
    frac_delay = amp * np.sin(2 * np.pi * rate * t)
    int_delay = int(np.floor(frac_delay))
    frac = frac_delay - int_delay
    
    # Determine indexes for the circular buffer
    len_buffer = len(buffer)
    index_c = (n - 1) % len_buffer  # Current index
    index_d = (n - delay - 1 + int_delay) % len_buffer  # Delay index with LFO modulation
    index_f = (n - delay - 1 + int_delay + 1) % len_buffer  # Fractional index for interpolation
    
    # Interpolate between the delayed samples for fractional delay
    out = (1 - frac) * buffer[index_d] + frac * buffer[index_f]
    
    # Store the current output in the buffer and apply LPF in the feedback path
    buffer[index_c] = input_sample + fb_gain * (0.5 * out + 0.5 * fb_lpf)
    
    # Update feedback LPF value for the next sample
    fb_lpf = out
    
    return out, buffer, fb_lpf



def apf(input_sample, buffer, fs, n, delay, gain, amp, rate):

    # Calculate time in seconds for the current sample for LFO modulation
    t = (n - 1) / fs
    frac_delay = amp * np.sin(2 * np.pi * rate * t)
    int_delay = int(np.floor(frac_delay))
    frac = frac_delay - int_delay
    
    # Determine indices for the circular buffer
    len_buffer = len(buffer)
    index_c = (n - 1) % len_buffer  # Current index
    index_d = (n - delay - 1 + int_delay) % len_buffer  # Delay index
    index_f = (n - delay - 1 + int_delay + 1) % len_buffer  # Fractional index for interpolation
    
    # Interpolation for fractional delay adjustment
    w = (1 - frac) * buffer[index_d] + frac * buffer[index_f]
    
    # Compute the output of the APF
    v = input_sample - gain * w
    out = gain * v + w
    
    # Update the buffer with the current processed input
    buffer[index_c] = v
    
    return out, buffer


# Define the main reverb function
def moorer_reverb(input_signal, fs):
    
    input_signal = input_signal.astype(np.float64)

    # Initialize buffers for early reflections, LPCF, and APF
    max_delay_early_reflections = int(fs * 0.07)  # Max delay for early reflections
    early_reflections_buffer = np.zeros(max_delay_early_reflections)

    max_delay_lpcf = int(fs * 0.05)  # Max delay for LPCF
    lpcf_buffer = np.zeros(max_delay_lpcf)
    max_delay_apf = int(fs * 0.05)  # Max delay for APF
    apf_buffer = np.zeros(max_delay_apf)
    output_signal = np.zeros_like(input_signal)

    # Parameters for early reflections, LPCF, and APF
    fb_lpf_initial = 0
    delay_lpcf = int(fs * 0.042)
    fb_gain_lpcf = 0.8
    amp_lpcf = 6
    rate_lpcf = 0.9

    delay_apf = int(fs * 0.032)
    gain_apf = 0.8
    amp_apf = 6
    rate_apf = 0.9

    for n, input_sample in enumerate(input_signal, 1):
        # Apply early reflections
        early_reflections_out, early_reflections_buffer = early_reflections(input_sample, early_reflections_buffer, fs, n)

        # Apply LPCF
        lpcf_out, lpcf_buffer, fb_lpf_initial = lpcf(early_reflections_out, lpcf_buffer, fs, n, delay_lpcf, fb_gain_lpcf, amp_lpcf, rate_lpcf, fb_lpf_initial)

        # Apply APF
        apf_out, apf_buffer = apf(lpcf_out, apf_buffer, fs, n, delay_apf, gain_apf, amp_apf, rate_apf)

        output_signal[n - 1] = apf_out

    return output_signal

input_signal, fs = sf.read('effect-off.wav') 

# If the audio is stereo, convert it to mono
if input_signal.ndim == 2:
    input_signal = input_signal.mean(axis=1)

reverberated_signal = moorer_reverb(input_signal, fs)
sf.write('reverberated_signal.wav', reverberated_signal, fs)
