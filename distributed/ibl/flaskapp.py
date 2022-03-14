# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import base64
from pathlib import Path
import logging
import io
import traceback

import numpy as np

from flask import Flask, render_template
from flask_socketio import SocketIO, emit

from datoviz import Requester, Renderer


# -------------------------------------------------------------------------------------------------
# Logger
# -------------------------------------------------------------------------------------------------

logger = logging.getLogger('datoviz')


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


# -------------------------------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent.resolve()
DATA_DIR = ROOT_DIR / 'data/rep_site'

WIDTH = 800
HEIGHT = 600


# -------------------------------------------------------------------------------------------------
# Data access
# -------------------------------------------------------------------------------------------------

def get_array(data):
    if data.mode == 'base64':
        r = base64.decodebytes(data.buffer.encode('ascii'))
        return np.frombuffer(r, dtype=np.uint8)
    elif data.mode == 'ibl_ephys':

        # Retrieve the requested session eid.
        eid = data.session['eid']
        session_dir = DATA_DIR / eid

        # Load the data.
        spike_times = np.load(session_dir / 'spikes.times.npy')
        spike_depths = np.load(session_dir / 'spikes.depths.npy')
        spike_clusters = np.load(session_dir / 'spikes.clusters.npy')
        spike_amps = np.load(session_dir / 'spikes.amps.npy')

        spike_depths[np.isnan(spike_depths)] = 0

        logger.debug(f"downloaded {len(spike_times)} spikes")

        # HACK: maximum data size
        n = data.count
        assert n > 0
        assert n <= spike_times.size

        # Prepare the vertex buffer for the raster graphics.
        arr = np.zeros(n, dtype=[
            ('pos', np.float32, 2),
            ('depth', np.uint8),
            ('cmap_val', np.uint8),
            ('alpha', np.uint8),
            ('size', np.uint8)
        ])

        # Generate the position data.
        x = normalize(spike_times[:n])
        y = normalize(spike_depths[:n])
        arr["pos"][:, 0] = x
        arr["pos"][:, 1] = y

        features = {
            'time': spike_times[:n],
            'cluster': spike_clusters[:n],
            'depth': spike_depths[:n],
            'amplitude': spike_amps[:n],
            None: np.ones(n),
        }

        reqfet = data.get('features', {})

        # Color feature.
        fet_color = reqfet.get('color', None)
        arr["cmap_val"][:] = normalize(features[fet_color], target='uint8')

        # Alpha feature.
        fet_alpha = reqfet.get('alpha', None)
        arr["alpha"][:] = normalize(features[fet_alpha], target='uint8')

        # Size feature.
        fet_size = reqfet.get('size', None)
        arr["size"][:] = normalize(features[fet_size], target='uint8')

        print(arr)

        return arr

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
    ('record', 'draw'): lambda r, req: r.record_draw(int(req.id), int(req.content.graphics), int(req.content.first_vertex), int(req.content.vertex_count)),
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
socketio = SocketIO(app)


def get_sessions():
    for session in sorted(DATA_DIR.iterdir()):
        yield session.name


@app.route('/')
def main():
    return render_template('index.html', sessions=get_sessions())


rnd = Renderer()


@socketio.on('connect')
def test_connect():
    logger.debug("Client connected")


@socketio.on('request')
def on_request(msg):
    try:
        process(rnd, msg['requests'])
        # TODO: board id
        img = render(rnd, 1)  # msg['render'])
        emit('image', {"image": img.getvalue()})
    except Exception as e:
        logger.error(traceback.format_exc())
        return


@socketio.on('disconnect')
def test_disconnect():
    logger.debug('Client disconnected')


if __name__ == '__main__':
    socketio.run(app, port=5000)
