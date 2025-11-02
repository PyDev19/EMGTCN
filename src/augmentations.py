from scipy.interpolate import CubicSpline
import numpy as np
import scipy
import torch


def low_pass_filter(signal, cutoff_freq=1.0, sampling_freq=100):
    """
    Applies a low-pass filter to the input signal.

    Parameters:
        signal (array-like): The input signal to be filtered. It is expected to be a 1D or 2D array.
        cutoff_freq (float, optional): The cutoff frequency of the low-pass filter in Hz. Default is 1.0 Hz.
        sampling_freq (float, optional): The sampling frequency of the input signal in Hz. Default is 100 Hz.

    Returns:
        numpy.ndarray: The filtered signal with the same shape as the input.

    Notes:
        - The function uses a first-order Butterworth filter for low-pass filtering.
        - The input signal is first converted to its absolute value before filtering.
        - The `scipy.signal.filtfilt` function is used to apply the filter, which ensures zero-phase distortion.
    """
    normalized_cutoff = cutoff_freq / (sampling_freq / 2)
    signal = np.abs(signal)

    b_coefficients, a_coefficients = scipy.signal.butter(1, normalized_cutoff, "low")

    filtered_signal = scipy.signal.filtfilt(
        b_coefficients,
        a_coefficients,
        signal,
        axis=0,
        padtype="odd",
        padlen=3 * (max(len(b_coefficients), len(a_coefficients)) - 1),
    )

    return filtered_signal


def jitter(x, snr_db=25):
    if isinstance(snr_db, list):
        snr_db_low = snr_db[0]
        snr_db_up = snr_db[1]
    else:
        snr_db_low = snr_db
        snr_db_up = 45
    snr_db = np.random.randint(snr_db_low, snr_db_up, (1,))[0]
    snr = 10 ** (snr_db / 10)
    Xp = np.sum(x**2, axis=0, keepdims=True) / x.shape[0]
    Np = Xp / snr
    n = np.random.normal(size=x.shape, scale=np.sqrt(Np), loc=0.0)
    xn = x + n
    return xn


def scale(x, sigma=0.2):
    """Multiply signal with random scalar from normal distribution N(1,sigma)."""
    a = np.random.normal(size=x.shape[1], scale=sigma, loc=1.0)
    output = a * x
    return output


def rotate(x, rotation=2, mask=None):
    """Rotate signal channels randomly between [0,2] positions. Use mask to disable rotation of specific channels"""
    output = np.zeros(x.shape)
    r = np.random.randint(-rotation, rotation + 1, size=1)[0]
    if mask is None:
        mask = np.ones(x.shape[1])
    channels = np.arange(x.shape[1])
    rolled = np.roll(channels[np.where(mask)], r)
    channels[np.where(mask)] = rolled
    output = x[:, channels]
    return output


def _generate_random_curve(x, sigma=0.2, knot=4):
    xx = ((np.arange(0, x.shape[0], (x.shape[0] - 1) / (knot + 1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2,))
    
    x_range = np.arange(x.shape[0])
    cs = CubicSpline(xx[:], yy[:])
    
    return np.array(cs(x_range)).transpose()


def _distort_timesteps(x, sigma=0.2):
    # Regard these samples aroun 1 as time intervals
    tt = _generate_random_curve(x, sigma)
    
    # Add intervals to make a cumulative graph
    tt_cum = np.cumsum(tt, axis=0)
    
    # Make the last value to have X.shape[0]
    t_scale = (x.shape[0] - 1) / tt_cum[-1]
    tt_cum = tt_cum * t_scale
    
    return tt_cum


def mag_warp(x, sigma=0.2):
    output = np.zeros(x.shape)

    for i in range(x.shape[1]):
        rc = _generate_random_curve(x[:, i], sigma)
        output[:, i] = x[:, i] * rc
    
    return output


def time_warp(x, sigma=0.1):
    output = np.zeros(x.shape)

    for i in range(x.shape[1]):
        tt_new = _distort_timesteps(x[:, i], sigma)
        tt_new = np.clip(tt_new, 0, x.shape[0] - 1)
        output[:, i] = x[tt_new.astype(int), i]
    
    return output


def permute(x, nPerm=4, minSegLength=10):
    def max_seg(segments):
        m = 0
        for i in range(len(segments) - 1):
            if (segments[i + 1] - segments[i]) >= (segments[m + 1] - segments[m]):
                m = i
        return (segments[m], segments[m + 1])

    segs = [0, x.shape[0]]

    it = 0
    while len(segs) < nPerm + 1:
        a, b = max_seg(segs)
        if b - a > 2 * minSegLength:
            p = np.random.randint(a, b, size=1)[0]
            if ((b - p) >= minSegLength) and ((p - a) >= minSegLength):
                segs.append(p)
        elif b - a == 2 * minSegLength:
            segs.append((b + a) / 2)
        else:
            break
        segs.sort()
        it += 1
    
    segs = np.array(segs, dtype=int)
    idx = np.arange(len(segs) - 1)
    np.random.shuffle(idx)
    output = np.zeros(x.shape)
    pp = 0
    
    for ii in range(len(idx)):
        x_temp = x[segs[idx[ii]] : segs[idx[ii] + 1], :]
        output[pp : pp + len(x_temp), :] = x_temp
        pp += len(x_temp)
    
    return output
