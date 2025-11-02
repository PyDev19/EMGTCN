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


def jitter(signal, snr_db=25):
    """
    Adds random noise to the input signal to simulate jitter.

    Parameters:
        signal (numpy.ndarray): The input signal to which noise will be added. Expected to be a 2D array.
        snr_db (int or list, optional): The desired Signal-to-Noise Ratio (SNR) in decibels.
            - If an integer is provided, it is used as the lower bound, and the upper bound defaults to 45 dB.
            - If a list is provided, it should contain two values [lower_bound, upper_bound] for the SNR range.
            Default is 25 dB.

    Returns:
        numpy.ndarray: The signal with added noise, having the same shape as the input.

    Notes:
        - The function generates random noise based on the specified SNR range.
        - The noise is added to the signal to simulate jitter while maintaining the desired SNR.
    """
    if isinstance(snr_db, list):
        snr_db_lower_bound = snr_db[0]
        snr_db_upper_bound = snr_db[1]
    else:
        snr_db_lower_bound = snr_db
        snr_db_upper_bound = 45

    # Randomly select an SNR value within the specified range
    selected_snr_db = np.random.randint(snr_db_lower_bound, snr_db_upper_bound, (1,))[0]
    snr_linear = 10 ** (selected_snr_db / 10)

    # Calculate the power of the signal
    signal_power = np.sum(signal**2, axis=0, keepdims=True) / signal.shape[0]

    # Calculate the power of the noise based on the SNR
    noise_power = signal_power / snr_linear

    # Generate random noise with the calculated power
    noise = np.random.normal(size=signal.shape, scale=np.sqrt(noise_power), loc=0.0)

    # Add the noise to the signal
    noisy_signal = signal + noise

    return noisy_signal


def scale(signal, std_dev=0.2):
    """
    Scales the input signal by multiplying it with random scalars drawn from a normal distribution.

    Parameters:
        signal (numpy.ndarray): The input signal to be scaled. Expected to be a 2D array (time steps x channels).
        scale_std_dev (float, optional): The standard deviation of the normal distribution used to generate
                                         the scaling factors. Default is 0.2.

    Returns:
        numpy.ndarray: The scaled signal with the same shape as the input.

    Notes:
        - The scaling factors are drawn from a normal distribution N(1, scale_std_dev).
        - Each channel of the signal is scaled independently.
    """
    # Generate random scaling factors for each channel
    scaling_factors = np.random.normal(
        loc=1.0, scale=std_dev, size=signal.shape[1]
    )

    # Apply the scaling factors to the signal
    scaled_signal = scaling_factors * signal

    return scaled_signal


def rotate(signal, max_rotation=2, channel_mask=None):
    """
    Rotates the signal channels randomly within a specified range.

    Parameters:
        signal (numpy.ndarray): The input signal to be rotated. Expected to be a 2D array (time steps x channels).
        max_rotation (int, optional): The maximum number of positions to rotate the channels.
                                      Rotation is randomly chosen between [-max_rotation, max_rotation]. Default is 2.
        channel_mask (numpy.ndarray, optional): A binary mask indicating which channels are allowed to rotate.
                                                If None, all channels are rotated. Default is None.

    Returns:
        numpy.ndarray: The rotated signal with the same shape as the input.

    Notes:
        - The function rotates the channels of the signal along the second axis (channels).
        - Channels not included in the mask remain in their original positions.
        - The rotation is circular, meaning channels wrap around when rotated beyond the last position.
    """
    rotated_signal = np.zeros(signal.shape)

    # Randomly determine the number of positions to rotate
    rotation_amount = np.random.randint(-max_rotation, max_rotation + 1, size=1)[0]

    # If no mask is provided, allow rotation for all channels
    if channel_mask is None:
        channel_mask = np.ones(signal.shape[1], dtype=bool)

    # Get the indices of channels to rotate
    channel_indices = np.arange(signal.shape[1])
    channels_to_rotate = channel_indices[channel_mask]

    # Perform the rotation on the selected channels
    rotated_channels = np.roll(channels_to_rotate, rotation_amount)
    channel_indices[channel_mask] = rotated_channels

    # Apply the rotation to the signal
    rotated_signal = signal[:, channel_indices]

    return rotated_signal


def _generate_random_curve(signal, std_dev=0.2, num_knots=4):
    """
    Generates a random smooth curve using cubic spline interpolation.

    Parameters:
        signal (numpy.ndarray): The input signal for which the random curve is generated.
                                Expected to be a 1D array.
        std_dev (float, optional): The standard deviation of the normal distribution used to
                                   generate random values for the curve. Default is 0.2.
        num_knots (int, optional): The number of knots (control points) for the cubic spline.
                                   Default is 4.

    Returns:
        numpy.ndarray: A smooth random curve with the same length as the input signal.

    Notes:
        - The function generates random values at evenly spaced knots along the signal length.
        - A cubic spline is fitted to these random values to create a smooth curve.
        - The curve is used for augmentations such as time warping or magnitude warping.
    """
    # Generate evenly spaced knot positions along the signal length
    knot_positions = np.linspace(0, signal.shape[0] - 1, num_knots + 2)

    # Generate random values for the knots from a normal distribution
    knot_values = np.random.normal(loc=1.0, scale=std_dev, size=(num_knots + 2,))

    # Create a cubic spline interpolation based on the knots
    cubic_spline = CubicSpline(knot_positions, knot_values)

    # Generate the smooth curve by evaluating the spline at each point in the signal
    smooth_curve = cubic_spline(np.arange(signal.shape[0]))

    return smooth_curve


def _distort_timesteps(x, sigma=0.2):
    # Regard these samples aroun 1 as time intervals
    tt = _generate_random_curve(x, sigma)

    # Add intervals to make a cumulative graph
    tt_cum = np.cumsum(tt, axis=0)

    # Make the last value to have X.shape[0]
    t_scale = (x.shape[0] - 1) / tt_cum[-1]
    tt_cum = tt_cum * t_scale

    return tt_cum


def magnitude_warp(signal, std_dev=0.2):
    """
    Applies magnitude warping to the input signal by scaling it with a smooth random curve.

    Parameters:
        signal (numpy.ndarray): The input signal to be warped. Expected to be a 2D array (time steps x channels).
        std_dev (float, optional): The standard deviation of the normal distribution used to generate
                                   the random curve. Default is 0.2.

    Returns:
        numpy.ndarray: The magnitude-warped signal with the same shape as the input.

    Notes:
        - A smooth random curve is generated for each channel using cubic spline interpolation.
        - The signal is scaled element-wise by the generated curve for each channel.
    """
    # Initialize the output array with the same shape as the input signal
    warped_signal = np.zeros(signal.shape)

    # Apply magnitude warping to each channel independently
    for channel_idx in range(signal.shape[1]):
        random_curve = _generate_random_curve(signal[:, channel_idx], std_dev)
        warped_signal[:, channel_idx] = signal[:, channel_idx] * random_curve

    return warped_signal


def time_warp(signal, std_dev=0.1):
    """
    Applies time warping to the input signal by distorting its time steps using a smooth random curve.

    Parameters:
        signal (numpy.ndarray): The input signal to be time-warped. Expected to be a 2D array (time steps x channels).
        std_dev (float, optional): The standard deviation of the normal distribution used to generate
                                   the random curve for time step distortion. Default is 0.1.

    Returns:
        numpy.ndarray: The time-warped signal with the same shape as the input.

    Notes:
        - A smooth random curve is generated for each channel to distort the time steps.
        - The time steps are clipped to ensure they remain within valid bounds.
        - The signal is resampled at the distorted time steps for each channel.
    """
    # Initialize the output array with the same shape as the input signal
    warped_signal = np.zeros(signal.shape)

    # Apply time warping to each channel independently
    for channel_idx in range(signal.shape[1]):
        # Generate distorted time steps for the current channel
        distorted_time_steps = _distort_timesteps(signal[:, channel_idx], std_dev)

        # Clip the time steps to ensure they are within valid bounds
        distorted_time_steps = np.clip(distorted_time_steps, 0, signal.shape[0] - 1)

        # Resample the signal at the distorted time steps
        warped_signal[:, channel_idx] = signal[
            distorted_time_steps.astype(int), channel_idx
        ]

    return warped_signal


def find_largest_segment(segments):
    """
    Finds the largest segment (in terms of length) from the list of segments.

    Parameters:
        segments (list): A list of segment boundaries.

    Returns:
        tuple: The start and end indices of the largest segment.
    """
    largest_idx = 0
    for i in range(len(segments) - 1):
        if (segments[i + 1] - segments[i]) >= (
            segments[largest_idx + 1] - segments[largest_idx]
        ):
            largest_idx = i
    return segments[largest_idx], segments[largest_idx + 1]


def permute(signal, num_permutations=4, min_segment_length=10):
    """
    Randomly permutes segments of the input signal.

    Parameters:
        signal (numpy.ndarray): The input signal to be permuted. Expected to be a 2D array (time steps x channels).
        num_permutations (int, optional): The number of segments to create for permutation. Default is 4.
        min_segment_length (int, optional): The minimum length of each segment. Default is 10.

    Returns:
        numpy.ndarray: The permuted signal with the same shape as the input.

    Notes:
        - The function divides the signal into random segments and shuffles their order.
        - Segments are created such that their lengths are at least `min_segment_length`.
        - The permutation is applied along the time axis (first dimension).
    """
    # Initialize segment boundaries with the start and end of the signal
    segment_boundaries = [0, signal.shape[0]]

    # Iteratively add new segment boundaries
    iterations = 0
    while len(segment_boundaries) < num_permutations + 1:
        start, end = find_largest_segment(segment_boundaries)
        if end - start > 2 * min_segment_length:
            new_boundary = np.random.randint(start, end, size=1)[0]
            if ((end - new_boundary) >= min_segment_length) and (
                (new_boundary - start) >= min_segment_length
            ):
                segment_boundaries.append(new_boundary)
        elif end - start == 2 * min_segment_length:
            segment_boundaries.append((end + start) // 2)
        else:
            break
        segment_boundaries.sort()
        iterations += 1

    # Convert segment boundaries to a numpy array
    segment_boundaries = np.array(segment_boundaries, dtype=int)

    # Create an array of segment indices and shuffle them
    segment_indices = np.arange(len(segment_boundaries) - 1)
    np.random.shuffle(segment_indices)

    # Initialize the output signal
    permuted_signal = np.zeros(signal.shape)

    # Permute the segments and construct the output signal
    current_position = 0
    for idx in segment_indices:
        segment = signal[segment_boundaries[idx] : segment_boundaries[idx + 1], :]
        permuted_signal[current_position : current_position + len(segment), :] = segment
        current_position += len(segment)

    return permuted_signal
