# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import argparse
import base64
from pathlib import Path
import logging
import io
from math import ceil
import traceback

import numpy as np
from scipy.signal import butter, sosfilt
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import png
from flask import Flask, render_template, send_file, session, request
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO, Namespace, emit

from datoviz import Requester, Renderer
from mtscomp.lossy import decompress_lossy


# -------------------------------------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------------------------------------

logger = logging.getLogger('datoviz')
mpl.style.use('seaborn')


# -------------------------------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / 'data/rep_site'
PORT = 4321
SAMPLE_RATE = 3e4  # HACK NOTE: we assume the sample rate to be fixed here!
HP_FILTER_FREQ = 300  # 300 Hz high-pass filter

MAX_USERS = 8
N_USERS = 0

TIME_HALF_WINDOW = 0.1  # in seconds, needs to match HALF_WINDOW in scripts.js


# -------------------------------------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------------------------------------

class Bunch(dict):
    def __init__(self, *args, **kwargs):
        self.__dict__ = self
        super().__init__(*args, **kwargs)


def normalize(x, target='float'):
    m = x.min()
    M = x.max()
    if m == M:
        # logger.warning("degenerate values")
        m = M - 1
    if target == 'float':  # normalize in [-1, +1]
        return -1 + 2 * (x - m) / (M - m)
    elif target == 'uint8':  # normalize in [0, 255]
        return np.round(255 * (x - m) / (M - m)).astype(np.uint8)
    raise ValueError("unknow normalization target")


def to_png(arr):
    p = png.from_array(arr, mode="L")
    b = io.BytesIO()
    p.write(b)
    b.seek(0)
    return b


def send_image(img):
    return send_file(to_png(img), mimetype='image/png')


def send_figure(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


def hpfilter(arr, freq=HP_FILTER_FREQ):
    sos = butter(3, freq, fs=SAMPLE_RATE,  btype='highpass', output='sos')
    return sosfilt(sos, arr, axis=1)


# -------------------------------------------------------------------------------------------------
# Data access
# -------------------------------------------------------------------------------------------------

def get_array(data):
    if data.mode == 'base64':
        r = base64.decodebytes(data.buffer.encode('ascii'))
        return np.frombuffer(r, dtype=np.uint8)
    elif data.mode == 'ibl_ephys':
        # Retrieve the requested session eid.
        data.session = Bunch(data.session)
        eid = data.session.eid
        session_dir = DATA_DIR / eid

        # Load the data.
        spike_times = np.load(session_dir / 'spikes.times.npy')
        spike_depths = np.load(session_dir / 'spikes.depths.npy')
        spike_clusters = np.load(session_dir / 'spikes.clusters.npy')
        spike_amps = np.load(session_dir / 'spikes.amps.npy')
        n = len(spike_times)

        # Cluster metrics
        metrics = pd.read_parquet(session_dir / 'clusters.metrics.pqt')
        metrics = metrics.reindex(metrics['cluster_id'])
        quality = metrics.label == 1
        spike_quality = quality[spike_clusters].values.astype(np.uint8)

        spike_depths[np.isnan(spike_depths)] = 0

        logger.debug(f"downloaded {len(spike_times)} spikes")

        # Prepare the vertex buffer for the raster graphics.
        arr = np.zeros(n, dtype=[
            ('pos', np.float32, 2),
            ('depth', np.uint8),
            ('cmap_val', np.uint8),
            ('alpha', np.uint8),
            ('size', np.uint8)
        ])

        # Available features.
        features = {
            'time': spike_times,
            'cluster': spike_clusters,
            'depth': spike_depths,
            'amplitude': spike_amps,
            'quality': spike_quality,
            None: np.ones(n),
        }

        # Generate the position data.
        x = normalize(spike_times)
        y = normalize(spike_depths)
        arr["pos"][:, 0] = x
        arr["pos"][:, 1] = y

        # Color feature.
        color = data.session.get('color', None) or 'cluster'
        arr["cmap_val"][:] = normalize(features[color], target='uint8')

        # Alpha feature.
        alpha = data.session.get('alpha', None) or None
        arr["alpha"][:] = normalize(features[alpha], target='uint8')

        # Size feature.
        size = data.session.get('size', None) or None
        arr["size"][:] = normalize(features[size], target='uint8')

        return arr

    raise Exception(f"Data upload mode '{data.mode}' unsupported")


def get_vertex_count(data):
    if not isinstance(data, dict):
        return data
    data = Bunch(data)
    if data.mode == 'ibl_ephys':
        # Retrieve the requested session eid.
        data.session = Bunch(data.session)
        eid = data.session.eid
        session_dir = DATA_DIR / eid
        spike_times = np.load(session_dir / 'spikes.times.npy', mmap_mode='r')
        return spike_times.size

    raise Exception(f"Data upload mode '{data.mode}' unsupported")


# -------------------------------------------------------------------------------------------------
# Renderer
# -------------------------------------------------------------------------------------------------

ROUTER = {
    ('create', 'board'): lambda r, req: r.create_board(int(req.content.width), int(req.content.height), id=int(req.id), background=req.content.background, flags=int(req.flags)),
    ('create', 'graphics'): lambda r, req: r.create_graphics(int(req.content.board), int(req.content.type), id=int(req.id), flags=int(req.flags)),
    ('create', 'dat'): lambda r, req: r.create_dat(int(req.content.type), int(req.content.size), id=int(req.id), flags=int(req.flags)),
    ('bind', 'dat'): lambda r, req: r.bind_dat(int(req.id), int(req.content.slot_idx), int(req.content.dat)),
    ('set', 'vertex'): lambda r, req: r.set_vertex(int(req.id), int(req.content.dat)),
    ('upload', 'dat'): lambda r, req: r.upload_dat(int(req.id), int(req.content.offset), get_array(Bunch(req.content.data))),
    ('record', 'begin'): lambda r, req: r.record_begin(int(req.id)),
    ('record', 'viewport'): lambda r, req: r.record_viewport(int(req.id), int(req.content.offset[0]), int(req.content.offset[1]), int(req.content.shape[0]), int(req.content.shape[0])),
    ('record', 'draw'): lambda r, req: r.record_draw(int(req.id), int(req.content.graphics), int(req.content.first_vertex), get_vertex_count(req.content.vertex_count)),
    ('record', 'end'): lambda r, req: r.record_end(int(req.id)),
    ('update', 'board'): lambda r, req: r.update_board(int(req.id)),
}


def process(rnd, requests):
    requester = Requester()
    with requester.requests():
        # Process all requests.
        for req in requests:
            req = Bunch(req)
            if 'flags' not in req:
                req.flags = 0
            if 'content' in req:
                req.content = Bunch(req.content)
            ROUTER[req.action, req.type](requester, req)
    requester.submit(rnd)


def render(rnd, board_id=1):
    # Trigger a redraw.
    requester = Requester()
    with requester.requests():
        requester.update_board(board_id)
    requester.submit(rnd)

    # Get the image.
    img = rnd.get_png(board_id)

    # Return as PNG
    output = io.BytesIO(img)
    return output


# -------------------------------------------------------------------------------------------------
# Server
# -------------------------------------------------------------------------------------------------

app = Flask(__name__)
CORS(app, support_credentials=True)
socketio = SocketIO(app)


# -------------------------------------------------------------------------------------------------
# Serving the HTML page
# -------------------------------------------------------------------------------------------------

def get_session_info(session):
    session_dir = DATA_DIR / session

    st = np.load(session_dir / 'spikes.times.npy', mmap_mode='r')
    metrics = pd.read_parquet(session_dir / 'clusters.metrics.pqt')

    cluster_ids = metrics['cluster_id'].tolist()
    spike_count = st.size
    duration = int(ceil(st[-1] + 1))
    return {
        'spike_count': spike_count,
        'duration': duration,
        'cluster_ids': cluster_ids,
    }


def get_context():
    return {
        'sessions': {
            session.name: get_session_info(session) for session in sorted(DATA_DIR.iterdir())
        }
    }


@app.route('/')
def main():
    global N_USERS
    if N_USERS >= MAX_USERS:
        return f"Sorry, maximum number of users reached ({MAX_USERS}), please try again later"
    ctx = get_context()
    return render_template('index.html', sessions=ctx['sessions'], js_context=ctx)


# -------------------------------------------------------------------------------------------------
# WebSocket server
# -------------------------------------------------------------------------------------------------

class RendererNamespace(Namespace):
    def on_connect(self, e):
        global N_USERS
        N_USERS += 1
        logger.info(f"Client connected {N_USERS}")
        session['renderer'] = Renderer()

    def on_disconnect(self):
        logger.info('Client disconnected')
        session.pop('renderer', None)
        global N_USERS
        N_USERS -= 1

    def on_request(self, msg):
        rnd = session.get('renderer', None)
        assert rnd
        try:
            process(rnd, msg['requests'])
            # TODO: board id
            img = render(rnd, 1)  # msg['render'])
            emit('image', {"image": img.getvalue()})
        except Exception as e:
            logger.error(traceback.format_exc())
            return


# -------------------------------------------------------------------------------------------------
# Raw ephys data server
# -------------------------------------------------------------------------------------------------

def get_img(eid, time=0, dt=TIME_HALF_WINDOW):
    time = float(time)
    assert 0 <= time <= 1e6

    session_dir = DATA_DIR / eid
    path_lossy = list(session_dir.glob("*.lossy.npy"))
    path_lossy = path_lossy[0] if path_lossy else None
    if not path_lossy:
        logger.error(f"lossy raw data file does not exist at {path_lossy}")
        return np.zeros((1, 1, 3), dtype=np.uint8)
    assert path_lossy.exists()
    lossy = decompress_lossy(path_lossy)
    duration = lossy.duration
    assert duration > 0

    sr = lossy.sample_rate
    t = float(time)
    t = np.clip(t, dt, duration - dt)

    t0, t1 = t-dt, t+dt
    arr = lossy.get(t0, t1, filter=hpfilter, cast_to_uint8=True).T
    arr = arr[::-1, ::+1].copy()
    ns, nc = arr.shape

    # if (session_dir / 'spikes.samples.npy').exists():
    #     #
    #     # NOTE: TODO, used spikes.samples.npy, NOT times (not syn)
    #     # samples = np.load(session_dir / 'spikes.samples.npy', mmap_mode='r')
    #     times = np.load(session_dir / 'spikes.times.npy', mmap_mode='r')
    #     samples = lossy.t2s(times)
    #     #

    #     # Find the spikes in the selected region.
    #     s0 = lossy.t2s(t0)
    #     s1 = lossy.t2s(t1)
    #     i0, i1 = np.searchsorted(samples, np.array([s0, s1]))

    #     # Load and normalize the spike depths.
    #     depths = np.load(session_dir / 'spikes.depths.npy')
    #     depths[np.isnan(depths)] = 0
    #     m, M = depths.min(), depths.max()
    #     depths_n = (depths[i0:i1] - m) / (M - m)

    #     # Find the spike indices in the image array.
    #     cols = np.round(depths_n * nc).astype(np.int64)
    #     rows = (samples[i0:i1] - s0).astype(np.int64)

    #     cols = np.clip(cols, 0, nc - 1)
    #     rows = np.clip(rows, 0, ns - 1)

    #     for i, j in zip(rows, cols):
    #         arr[i-3:i+3, j-3:j+3] = 255

    return arr


def t2s(t):
    return np.round(t * SAMPLE_RATE).astype(np.uint64)


def s2t(s):
    return s / float(SAMPLE_RATE)


def get_spikes(eid, time=0, dt=TIME_HALF_WINDOW):
    time = float(time)
    assert 0 <= time <= 1e6

    session_dir = DATA_DIR / eid

    fn_samples = session_dir / 'spikes.samples.npy'
    if not fn_samples.exists():
        times = np.load(session_dir / 'spikes.times.npy', mmap_mode='r')
        samples = t2s(times)
    else:
        samples = np.load(fn_samples, mmap_mode='r')
        times = s2t(samples)

    # Estimate the duration.
    duration = times[-1] + 1

    t = float(time)
    t = np.clip(t, dt, duration - dt)
    t0, t1 = t-dt, t+dt

    # Find the spikes in the selected region.
    s0 = t2s(t0)
    s1 = t2s(t1)
    i0, i1 = np.searchsorted(samples, np.array([s0, s1]))

    # Prepare the output arrays.
    spike_times = times[i0:i1]
    spike_samples = samples[i0:i1]

    depths = np.load(session_dir / 'spikes.depths.npy')
    depths[np.isnan(depths)] = 0
    spike_depths = depths[i0:i1]

    sc = np.load(session_dir / 'spikes.clusters.npy', mmap_mode='r')
    spike_clusters = sc[i0:i1]

    cc = np.load(session_dir / 'clusters.channels.npy', mmap_mode='r')
    spike_channels = cc[spike_clusters]

    # Coordinates on the plot.
    x = spike_times - t0
    y = spike_channels

    d = dict(
        spike_times=np.around(spike_times, decimals=3).tolist(),
        spike_samples=np.around(spike_samples).tolist(),
        spike_clusters=np.around(spike_clusters).tolist(),
        spike_depths=np.around(spike_depths, decimals=3).tolist(),
        spike_channels=np.around(spike_channels).tolist(),
        x=np.around(x, decimals=3).tolist(),
        y=np.around(y, decimals=3).tolist(),
    )

    # return json.dumps(d, indent=1, sort_keys=True)
    return d


@app.route('/<eid>/raw/<time>')
@cross_origin(supports_credentials=True)
def serve_time_float(eid, time=None):
    if time == 'null':
        return ''
    args = request.args
    # Half of the window size (in seconds).
    dt = args.get('dt', default=TIME_HALF_WINDOW, type=float)
    time = float(time)
    img = get_img(eid, time=time, dt=dt)
    if img is not None:
        return send_image(img)


@app.route('/<eid>/spikes/<time>')
@cross_origin(supports_credentials=True)
def get_spikes_in_window(eid, time=None):
    if time == 'null':
        return ''
    time = float(time)
    return get_spikes(eid, time=time)


@app.route('/<eid>/cluster/<cluster_id>')
@cross_origin(supports_credentials=True)
def cluster_plot(eid, cluster_id):
    if cluster_id == 'null':
        return ''
    cluster_id = int(cluster_id)

    assert eid
    if cluster_id is None or not cluster_id >= 0:
        logger.warning(f"invalid cluster number {cluster_id}")
        return ''

    session_dir = DATA_DIR / eid
    spike_clusters = np.load(
        session_dir / 'spikes.clusters.npy', mmap_mode='r')
    spike_times = np.load(session_dir / 'spikes.times.npy', mmap_mode='r')
    spike_depths = np.load(session_dir / 'spikes.depths.npy', mmap_mode='r')
    spike_amps = np.load(session_dir / 'spikes.amps.npy', mmap_mode='r')

    duration = spike_times[-1] + 1

    ind = np.isin(spike_clusters, [cluster_id])
    st = spike_times[ind]
    sd = spike_depths[ind]
    sa = spike_amps[ind] * 1e3

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(st, sa, ',')
    ax.set_xlim(0, duration)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('amplitude (mV)')
    ax.set_ylim(0)
    ax.set_title(f'spike amplitudes (cluster {cluster_id:04d})')
    fig.tight_layout()

    out = send_figure(fig)
    plt.close(fig)
    return out


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Launch the Flask server.')
    parser.add_argument('--port', help='the TCP port')
    args = parser.parse_args()

    socketio.on_namespace(RendererNamespace('/'))
    port = args.port or PORT
    logger.info(f"Serving the Flask application on port {port}")
    socketio.run(app, '0.0.0.0', port=port)
