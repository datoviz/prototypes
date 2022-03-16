# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import base64
from pathlib import Path
import logging
import io
import traceback

import numpy as np
import png
from flask import Flask, render_template, send_file, session
from flask_cors import CORS, cross_origin
from flask_socketio import SocketIO, Namespace, emit

from datoviz import Requester, Renderer
from mtscomp.lossy import decompress_lossy


# -------------------------------------------------------------------------------------------------
# Logger
# -------------------------------------------------------------------------------------------------

logger = logging.getLogger('datoviz')


# -------------------------------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / 'data/rep_site'

WIDTH = 800
HEIGHT = 600

TIME_HALF_WINDOW = 0.1  # in seconds

PORT = 4321


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

        # Generate the position data.
        x = normalize(spike_times)
        y = normalize(spike_depths)
        arr["pos"][:, 0] = x
        arr["pos"][:, 1] = y

        features = {
            'time': spike_times,
            'cluster': spike_clusters,
            'depth': spike_depths,
            'amplitude': spike_amps,
            None: np.ones(n),
        }

        # Color feature.
        arr["cmap_val"][:] = normalize(
            features[data.session.color], target='uint8')

        # Alpha feature.
        arr["alpha"][:] = normalize(
            features[data.session.alpha], target='uint8')

        # Size feature.
        arr["size"][:] = normalize(features[data.session.size], target='uint8')

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

def get_duration(session):
    return np.load(DATA_DIR / session / 'spikes.times.npy', mmap_mode='r')[-1] + 1


def get_context():
    sessions = [session.name for session in sorted(DATA_DIR.iterdir())]
    return {
        'sessions': sessions,
        'durations': {
            session: get_duration(session)
            for session in sessions
        }
    }


@app.route('/')
def main():
    ctx = get_context()
    return render_template('index.html', sessions=ctx['sessions'], js_context=ctx)


# -------------------------------------------------------------------------------------------------
# WebSocket server
# -------------------------------------------------------------------------------------------------

class RendererNamespace(Namespace):
    def on_connect(self, e):
        logger.info("Client connected")
        session['renderer'] = Renderer()

    def on_disconnect(self):
        logger.info('Client disconnected')
        session.pop('renderer', None)

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

def get_img(eid, time=0):
    time = float(time)
    assert 0 <= time <= 1e6
    path_lossy = Path(__file__).parent / 'raw.lossy.npy'
    lossy = decompress_lossy(path_lossy)
    duration = lossy.duration
    assert duration > 0

    dt = TIME_HALF_WINDOW
    t = float(time)
    t = np.clip(t, dt, duration - dt)

    arr = lossy.get(t - dt, t + dt, cast_to_uint8=True).T
    return arr


@app.route('/raw/<eid>/')
@cross_origin(supports_credentials=True)
def serve_default(eid):
    img = get_img(eid)
    return send_image(img)


@app.route('/raw/<eid>/<time>')
@cross_origin(supports_credentials=True)
def serve_time_float(eid, time=None):
    if not time or time == 'null':
        time = 0
    time = float(time)
    img = get_img(eid, time=time)
    return send_image(img)


if __name__ == '__main__':
    socketio.on_namespace(RendererNamespace('/'))
    socketio.run(app, '0.0.0.0', port=PORT)
