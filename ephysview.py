"""
Python example of an interactive raw ephys data viewer.

TODO:
- top panel with another file
- sort by different words
- apply different filters

"""

import logging
import math
from pathlib import Path

from joblib import Memory
import numpy as np
from oneibl.one import ONE

from datoviz import canvas, run


logger = logging.getLogger(__name__)


DESC = '''
Keyboard shortcuts:
- right/left: go to the next/previous 1s chunk
- +/-: change the scaling
- Home/End keys: go to start/end of the recording
- G: enter a time in seconds as a floating point in the terminal and press Enter to jump to that time

'''


# def create_image(shape):
#     image = np.zeros((shape[1], shape[0], 4), dtype=np.uint8)
#     image[..., 3] = 255
#     return image


def _memmap_flat(path, dtype=None, n_channels=None, offset=0):
    path = Path(path)
    # Find the number of samples.
    assert n_channels > 0
    fsize = path.stat().st_size
    item_size = np.dtype(dtype).itemsize
    n_samples = (fsize - offset) // (item_size * n_channels)
    if item_size * n_channels * n_samples != (fsize - offset):
        raise IOError("n_channels incorrect or binary file truncated")
    shape = (n_samples, n_channels)
    return np.memmap(path, dtype=dtype, offset=offset, shape=shape)


def get_scale(x):
    return np.median(x, axis=0), x.std()


def normalize(x, scale):
    m, s = scale
    out = np.empty_like(x, dtype=np.float32)
    out[...] = x
    out -= m
    out *= (1.0 / s)
    out += 1
    out *= 255 * .5
    out[out < 0] = 0
    out[out > 255] = 255
    return out.astype(np.uint8)


def get_data_urls(eid, probe_idx=0):
    # Find URL to .cbin file
    dsets = one.alyx.rest(
        'datasets', 'list', session=eid,
        django='name__icontains,ap.cbin,collection__endswith,probe%02d' % probe_idx)
    for fr in dsets[0]['file_records']:
        if fr['data_url']:
            url_cbin = fr['data_url']

    # Find URL to .ch file
    dsets = one.alyx.rest(
        'datasets', 'list', session=eid,
        django='name__icontains,ap.ch,collection__endswith,probe%02d' % probe_idx)
    for fr in dsets[0]['file_records']:
        if fr['data_url']:
            url_ch = fr['data_url']

    return url_cbin, url_ch


one = ONE()

# Disk cache of the downloading
location = Path('~/.one_cache/').expanduser()
memory = Memory(location, verbose=0)
@memory.cache
def _dl(url_cbin, url_ch, chunk_idx):
    reader = one.download_raw_partial(url_cbin, url_ch, chunk_idx, chunk_idx)
    return reader.cmeta, reader[:]


class RawDataModel:
    def __init__(self, eid, probe_idx=0):
        self.eid = eid
        self.probe_idx = probe_idx
        self.url_cbin, self.url_ch = get_data_urls(eid, probe_idx=probe_idx)
        assert self.url_cbin
        assert self.url_ch
        # Read the first chunk to get the total number of samples.
        info, _ = _dl(self.url_cbin, self.url_ch, 0)
        # print(info)
        self.n_samples = info.chopped_total_samples
        self.n_channels = _.shape[1]
        self.sample_rate = float(info.sample_rate)
        self.duration = self.n_samples / self.sample_rate

        assert self.n_samples > 0
        assert self.n_channels > 0
        assert self.sample_rate > 0
        assert self.duration > 0

    def _download_chunk(self, chunk_idx):
        _, arr = _dl(self.url_cbin, self.url_ch, chunk_idx)
        return arr

    def get_raw_data(self, t0, t1):
        t0 = np.clip(t0, 0, self.duration)
        t1 = np.clip(t1, 0, self.duration)
        assert t0 < t1
        expected_samples = int(round((t1 - t0) * self.sample_rate))

        # Chunk idxs, assuming 1 second chunk
        i0 = int(t0)  # in seconds
        i1 = int(t1)  # in seconds
        assert i0 >= 0
        assert i0 <= t0
        assert i1 <= t1
        assert i1 < self.n_samples

        # Download the chunks.
        arr = np.vstack([self._download_chunk(i) for i in range(i0, i1 + 1)])
        assert arr.ndim == 2
        assert arr.shape[1] == self.n_channels, (arr.shape, self.n_channels)

        # Offset within the array.
        s0 = int(round((t0 - i0) * self.sample_rate))
        assert 0 <= s0 < self.n_samples
        s1 = int(round((t1 - i0) * self.sample_rate))
        assert s0 < s1, (s0, s1)
        assert 0 < s1 <= self.n_samples, (s1, self.n_samples)
        assert s1 - s0 == expected_samples, (s0, s1, expected_samples)
        out = arr[s0:s1, :]
        assert out.shape == (expected_samples, self.n_channels)
        return out


class RawDataView:
    def __init__(self, n_channels):
        assert n_channels > 0
        self.n_channels = n_channels

        # Create the Visky view.
        self.canvas = canvas()
        self.panel = self.canvas.panel(controller='axes')

        # Image cmap visual
        self.v_image = self.panel.visual('image_cmap')

        # Initialize the POS prop.
        self.set_image_range(0, 1)

        # Top left, top right, bottom right, bottom left
        self.v_image.data('texcoords', np.atleast_2d([0, 0]), idx=0)
        self.v_image.data('texcoords', np.atleast_2d([1, 0]), idx=1)
        self.v_image.data('texcoords', np.atleast_2d([1, 1]), idx=2)
        self.v_image.data('texcoords', np.atleast_2d([0, 1]), idx=3)

    def set_image_range(self, t0, t1):
        # Top left, top right, bottom right, bottom left
        self.v_image.data('pos', np.array([[t0, t1, 0]]), idx=0)
        self.v_image.data('pos', np.array([[t1, t1, 0]]), idx=1)
        self.v_image.data('pos', np.array([[t1, t0, 0]]), idx=2)
        self.v_image.data('pos', np.array([[t0, t0, 0]]), idx=3)

    def set_vrange(self, vmin, vmax):
        self.v_image.data('range', np.array([vmin, vmax]))

    def set_image(self, img):
        assert img.ndim == 2
        assert img.shape[0] > 0
        assert img.shape[1] == self.n_channels
        self.v_image.image(img)
        self.set_vrange(0, 300)#img.min(), img.max())


class RawDataController:
    def __init__(self, model, view):
        self.m = model
        self.v = view

    def set_range(self, t0, t1):
        arr = self.m.get_raw_data(t0, t1)
        self.v.set_image_range(t0, t1)
        self.v.set_image(arr)


if __name__ == '__main__':

    eid = 'd33baf74-263c-4b37-a0d0-b79dcb80a764'

    m = RawDataModel(eid)

    # arr = m.get_raw_data(0, .1)
    # import matplotlib.pyplot as plt
    # plt.imshow(arr)
    # plt.show()

    v = RawDataView(m.n_channels)
    c = RawDataController(m, v)
    c.set_range(0, .1)

    run()

#         # Load the data and put it on the GPU.
#         self.load_data()
#         self.update_view()

#         # # Interactivity bindings.
#         # self.canvas.on_key(self.on_key)
#         # self.canvas.on_mouse(self.on_mouse)
#         # self.canvas.on_prompt(self.on_prompt)

#     def update_view(self):
#         self.scale = scale = self.scale or get_scale(self.arr_buf)
#         self.image[..., :3] = normalize(
#             self.arr_buf, scale).T[:, :, np.newaxis]
#         self.v_image.image(self.image)
#         # self.panel.axes_range(
#         #     self.sample / self.sample_rate,
#         #     0,
#         #     (self.sample + self.buffer_size) / self.sample_rate,
#         #     self.n_channels)

#     @property
#     def duration(self):
#         return self.n_samples / self.sample_rate

#     @property
#     def time(self):
#         return self.sample / self.sample_rate

#     def goto(self, time):
#         self.sample = int(round(time * self.sample_rate))
#         self.load_data()
#         self.update_view()

#     def on_key(self, key=None, modifiers=None):
#         pass
#         # delta = .1 * self.buffer_size / self.sample_rate
#         # if key == 'left':
#         #     self.goto(self.time - delta)
#         # elif key == 'right':
#         #     self.goto(self.time + delta)
#         # elif key == '+':
#         #     self.scale = (self.scale[0], self.scale[1] / 1.1)
#         #     self.update_view()
#         # elif key == '-':
#         #     self.scale = (self.scale[0], self.scale[1] * 1.1)
#         #     self.update_view()
#         # elif key == 'home':
#         #     self.goto(0)
#         # elif key == 'end':
#         #     self.goto(self.duration)
#         # else:
#         #     self.canvas.prompt()

#     def on_mouse(self, button, pos, state=None):
#         pass
#         # if state == 'click' and button == 'left':
#         #     # TODO: check 'click' mouse state instead
#         #     x, y = pos
#         #     x, y = self.canvas.pick(x, y)
#         #     # print(x, y)
#         #     i = math.floor(
#         #         (x - self.sample / self.sample_rate) /
#         #         (self.buffer_size / self.sample_rate) *
#         #         self.buffer_size)
#         #     j = math.floor(y)
#         #     j = self.n_channels - 1 - j
#         #     i = np.clip(i, 0, self.n_samples - 1)
#         #     j = np.clip(j, 0, self.n_channels - 1)
#         #     print(
#         #         f"Picked {x}, {y} : {self.arr_buf[i, j]}")

#     def on_prompt(self, t):
#         pass
#         # if not t:
#         #     return
#         # try:
#         #     t = float(t)
#         # except Exception:
#         #     logger.error("Invalid time %s" % str(t))
#         #     return
#         # if t:
#         #     self.goto(t)

#     def show(self):
#         run()


# if __name__ == '__main__':
#     n_channels = 385
#     dtype = np.int16
#     sample_rate = 30_000

#     viewer = RawEphysViewer(n_channels, sample_rate, dtype)

#     if 1:
#         # Load from HTTP.
#         viewer.load_session('d33baf74-263c-4b37-a0d0-b79dcb80a764')
#         viewer.create()
#         viewer.show()
#     else:
#         # Load from disk.
#         path = Path(__file__).parent / "raw_ephys.bin"
#         viewer.memmap_file(path)
#         viewer.create()
#         viewer.show()
