import numpy as np
import random
import time

from sklearn.neighbors import kneighbors_graph

from audiotools import read


def randwalk(num_points, ndim, step_scale=0.25, init_points=None):
    """Create a line using a random walk algorithm.

    Parameters
    ----------
    num_points : int
        Number of points for the n-dimensional line.
    ndim : int
        Number of dimensions.
    step_scale : float, default=0.25
        Scale of the average step size.
    init_points : array_length, len=ndim, default=None
        Starting point for the random walk.
    """
    path = np.empty([num_points, ndim])
    init_points = np.zeros(ndim) if init_points is None else init_points
    path[0, :] = np.asarray(init_points)
    for index in range(1, num_points):
        steps = np.random.normal(0, step_scale, size=(ndim))
        path[index, :] = path[index - 1, :] + steps

    return path


def overlap_and_add(grains, window=None, overlap=0.5):
    """Overlap and add a collection of signals.

    Parameters
    ----------
    grains : list of np.ndarrays, each w/shape=(num_samples, num_channels)
        Equally-shaped audio signals.
    window : np.ndarray, shape=(num_samples), default=np.hanning(num_samples)
        Window to apply to each grain.
    overlap : scalar, in [0.0, 1), default=0.5
        Ratio of the overlap between adjacent grains.

    Returns
    -------
    y_out : np.ndarray
        Resulting signal.
    """
    assert len(np.unique([_.shape for _ in grains])) == 1
    framesize, channels = grains[0].shape

    window = np.hanning(framesize).reshape(-1, 1) if window is None else window
    y_out = np.zeros([int(framesize * ((len(grains) - 1) * overlap + 1)),
                      channels])
    for idx, x_n in enumerate(grains):
        i0 = int(idx * framesize * (1.0 - overlap))
        i1 = i0 + framesize
        y_out[i0:i1, :] += window * x_n
    return y_out


def sample_feature_files(npz_files, num_samples):
    counts = {f: 0 for f in npz_files}
    # Arbitrarily distribute samples among each file.
    for n in xrange(num_samples):
        counts[random.choice(npz_files)] += 1

    files, time_points, features = [], [], []
    for f, n in counts.iteritems():
        data = np.load(f)
        idx = np.random.permutation(len(data['time_points']))
        for i in idx[:n]:
            files.append(f)
            time_points.append(data['time_points'][i])
            features.append(data['features'][i])
    return files, np.array(time_points), np.array(features)


def build_kneighbors_table(features, k_neighbors):
    sparse_connections = kneighbors_graph(features, k_neighbors + 1)
    # Iterate to unpack neighbors from sparse connections
    connections = list()
    for ridx in xrange(len(features)):
        connections.append(sparse_connections[ridx].nonzero()[1].tolist()[1:])

    return connections


def rand_acyclic_walk(connections, num_points, start_idx=None):
    connects = list(connections)
    path = []

    if start_idx is None:
        start_idx = np.random.randint(len(connections))

    while len(path) < num_points:
        path.append(start_idx)
        if not connects[start_idx]:
            break
        row_idx = np.random.randint(len(connects[start_idx]))
        start_idx = connects[start_idx].pop(row_idx)

    return path


def extract_grains(files, time_points, samplerate=44100, duration=0.5):
    grains = []
    framesize = int(samplerate * duration)
    for idx, (f, t) in enumerate(zip(files, time_points)):
        print "[{0}] {1:4d} {2}".format(time.asctime(), idx, f)
        x = read(f, samplerate=samplerate)[0]
        idx = int((t - duration / 2) * samplerate)
        idx = 0 if idx < 0 else idx
        grains.append(x[idx:idx + framesize, :])
    return grains
