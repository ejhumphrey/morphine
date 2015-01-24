import numpy as np
import scipy.io.wavfile as WF
import subprocess
import tempfile
import os


def sox(args):
    """Pass an argument list to SoX.

    Parameters
    ----------
    args : list
        Argument list for SoX. The first item can, but does not need to,
        be 'sox'.

    Returns
    -------
    status : bool
        True on success.
    """
    if args[0].lower() != "sox":
        args.insert(0, "sox")
    else:
        args[0] = "sox"

    process_handle = subprocess.Popen(args, stderr=subprocess.PIPE)
    return process_handle.wait() == 0


def convert(input_file, output_file,
            samplerate=None, channels=None, bytedepth=None):
    """Converts one audio file to another on disk.

    Parameters
    ----------
    input_file : str
        Input file to convert.
    output_file : str
        Output file to writer.
    samplerate : float, default=None
        Desired samplerate. If None, defaults to the same as input.
    channels : int, default=None
        Desired channels. If None, defaults to the same as input.
    bytedepth : int, default=None
        Desired bytedepth. If None, defaults to the same as input.

    Returns
    -------
    status : bool
        True on success.
    """
    args = ['sox', '--no-dither', input_file]

    if bytedepth:
        assert bytedepth in [1, 2, 3, 4, 8]
        args += ['-b{0}'.format(bytedepth * 8)]
    args += ['-c {0}'.format(channels)] if channels else []

    args += [output_file]
    if samplerate:
        args += ['rate', '-I', '{0}'.format(samplerate)]

    return sox(args)


def read(filename, samplerate=None, channels=None):
    """Read a soundfile into memory.

    Note: samplerate and channels are converted if the file does not match the
    requested parameters.

    Parameters
    ----------
    filename : str
        Path to a sound file to read.
    samplerate : scalar, default=None (as-is)
        Desired samplerate for the sound file.
    channels : int, default=None (as-is)
        Desired number of channels for the sound file.

    Returns
    -------
    x : np.ndarray, shape=(num_samples, num_channels)
        Sound signal.
    fs : scalar
        Samplerate of the returned signal.
    """
    try:
        fs, x = WF.read(filename)
        x = x.reshape(-1, 1) if x.ndim == 1 else x
        assert fs == samplerate or samplerate is None
        assert x.shape[1] == channels or channels is None
        assert x.dtype == 'int16'
    except:
        fid, tmpfile = tempfile.mkstemp(suffix=".wav")
        convert(
            filename, tmpfile, samplerate=samplerate,
            channels=channels, bytedepth=2)
        fs, x = WF.read(tmpfile)
        x = x.reshape(-1, 1) if x.ndim == 1 else x
        os.remove(tmpfile)

    return x.astype(np.float) / np.power(2.0, 15.0), samplerate
