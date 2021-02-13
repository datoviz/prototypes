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


def create_image(shape):
    image = np.zeros((shape[1], shape[0], 4), dtype=np.uint8)
    image[..., 3] = 255
    return image


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


DESC = '''
Keyboard shortcuts:
- right/left: go to the next/previous 1s chunk
- +/-: change the scaling
- Home/End keys: go to start/end of the recording
- G: enter a time in seconds as a floating point in the terminal and press Enter to jump to that time

'''


def _dl(url_cbin, url_ch, chunk):
    one = ONE()
    reader = one.download_raw_partial(url_cbin, url_ch, chunk, chunk)
    return reader.cmeta.chopped_total_samples, reader[:]


class RawEphysViewer:
    def __init__(self, n_channels, sample_rate, dtype, buffer_size=7_500):
        self.n_channels = n_channels
        self.sample_rate = float(sample_rate)
        self.dtype = dtype
        self.buffer_size = buffer_size
        self.sample = 0  # current sample
        self.arr_buf = None
        self.scale = None
        print(DESC)
        # self._download = lru_cache(maxsize=10)(self._download)

    def memmap_file(self, path):
        self.mmap_array = _memmap_flat(
            path, dtype=self.dtype, n_channels=self.n_channels)
        assert self.mmap_array.ndim == 2
        assert self.mmap_array.shape[1] == n_channels
        self.n_samples = self.mmap_array.shape[0]

    def load_session(self, eid, probe_idx=0):
        from oneibl.one import ONE
        self.one = ONE()

        # Disk cache of the downloading
        location = Path('~/.one_cache/').expanduser()
        self._memory = Memory(location, verbose=0)
        self._dl = self._memory.cache(_dl)

        dsets = self.one.alyx.rest(
            'datasets', 'list', session=eid,
            django='name__icontains,ap.cbin,collection__endswith,probe%02d' % probe_idx)
        for fr in dsets[0]['file_records']:
            if fr['data_url']:
                self.url_cbin = fr['data_url']

        dsets = self.one.alyx.rest(
            'datasets', 'list', session=eid,
            django='name__icontains,ap.ch,collection__endswith,probe%02d' % probe_idx)
        for fr in dsets[0]['file_records']:
            if fr['data_url']:
                self.url_ch = fr['data_url']

        self.n_samples = 0  # HACK: will be set by _load_from_web()

    def _clip_sample(self):
        if self.n_samples == 0:
            self.sample = max(0, self.sample)
        else:
            self.sample = int(round(np.clip(
                self.sample, 0, self.n_samples - self.buffer_size)))

    def _download(self, chunk):
        # print("download %d %d" % (i0, i1))

        # Call the cached function.
        ns, arr = self._dl(self.url_cbin, self.url_ch, chunk)

        # NOTE: set n_samples after the first download has been done
        if self.n_samples == 0:
            self.n_samples = ns
        assert arr.shape[1] == self.n_channels
        # assert arr.shape[0] <= int(round((i1 + 1 - i0) * self.sample_rate))
        return arr

    def _load_from_file(self):
        return self.mmap_array[self.sample:self.sample + self.buffer_size, :]

    def _load_from_web(self):
        if self.n_samples == 0:
            i0 = i1 = 0
        else:
            t0 = self.time
            t1 = t0 + (self.buffer_size - 1) / self.sample_rate
            assert t0 >= 0
            assert t1 <= self.duration
            assert t0 < t1
            i0 = int(t0)
            i1 = int(t1)
            assert i0 >= 0
            assert i1 < self.n_samples

        arr = self._download(i0)
        if i1 != i0:
            arr2 = self._download(i1)
            arr = np.vstack((arr, arr2))

        assert self.n_samples > 0
        s0 = self.sample - int(round(i0 * self.sample_rate))
        assert s0 >= 0
        s1 = s0 + self.buffer_size
        assert s1 - s0 == self.buffer_size
        return arr[s0:s1, :]

    def load_data(self):
        self._clip_sample()
        if hasattr(self, 'mmap_array'):
            self.arr_buf = self._load_from_file()
        elif hasattr(self, 'one'):
            self.arr_buf = self._load_from_web()
        assert self.arr_buf.shape == (self.buffer_size, self.n_channels)

    def create(self):
        # Create the Visky view.
        self.canvas = canvas()

        # Create the image and visual
        self.image = create_image((self.buffer_size, self.n_channels))
        self.panel = self.canvas.panel(row=0, col=0, controller='axes')

        self.v_image = self.panel.visual('image')

        # Top left, top right, bottom right, bottom left
        self.v_image.data('pos', np.array([[-1, +1, 0]]), idx=0)
        self.v_image.data('pos', np.array([[+1, +1, 0]]), idx=1)
        self.v_image.data('pos', np.array([[+1, -1, 0]]), idx=2)
        self.v_image.data('pos', np.array([[-1, -1, 0]]), idx=3)

        self.v_image.data('texcoords', np.atleast_2d([0, 0]), idx=0)
        self.v_image.data('texcoords', np.atleast_2d([0, 1]), idx=1)
        self.v_image.data('texcoords', np.atleast_2d([1, 1]), idx=2)
        self.v_image.data('texcoords', np.atleast_2d([1, 0]), idx=3)

        self.v_image.image(self.image)

        # Load the data and put it on the GPU.
        self.load_data()
        self.update_view()

        # # Interactivity bindings.
        # self.canvas.on_key(self.on_key)
        # self.canvas.on_mouse(self.on_mouse)
        # self.canvas.on_prompt(self.on_prompt)

    def update_view(self):
        self.scale = scale = self.scale or get_scale(self.arr_buf)
        self.image[..., :3] = normalize(
            self.arr_buf, scale).T[:, :, np.newaxis]
        self.v_image.image(self.image)
        # self.panel.axes_range(
        #     self.sample / self.sample_rate,
        #     0,
        #     (self.sample + self.buffer_size) / self.sample_rate,
        #     self.n_channels)

    @property
    def duration(self):
        return self.n_samples / self.sample_rate

    @property
    def time(self):
        return self.sample / self.sample_rate

    def goto(self, time):
        self.sample = int(round(time * self.sample_rate))
        self.load_data()
        self.update_view()

    def on_key(self, key=None, modifiers=None):
        pass
        # delta = .1 * self.buffer_size / self.sample_rate
        # if key == 'left':
        #     self.goto(self.time - delta)
        # elif key == 'right':
        #     self.goto(self.time + delta)
        # elif key == '+':
        #     self.scale = (self.scale[0], self.scale[1] / 1.1)
        #     self.update_view()
        # elif key == '-':
        #     self.scale = (self.scale[0], self.scale[1] * 1.1)
        #     self.update_view()
        # elif key == 'home':
        #     self.goto(0)
        # elif key == 'end':
        #     self.goto(self.duration)
        # else:
        #     self.canvas.prompt()

    def on_mouse(self, button, pos, state=None):
        pass
        # if state == 'click' and button == 'left':
        #     # TODO: check 'click' mouse state instead
        #     x, y = pos
        #     x, y = self.canvas.pick(x, y)
        #     # print(x, y)
        #     i = math.floor(
        #         (x - self.sample / self.sample_rate) /
        #         (self.buffer_size / self.sample_rate) *
        #         self.buffer_size)
        #     j = math.floor(y)
        #     j = self.n_channels - 1 - j
        #     i = np.clip(i, 0, self.n_samples - 1)
        #     j = np.clip(j, 0, self.n_channels - 1)
        #     print(
        #         f"Picked {x}, {y} : {self.arr_buf[i, j]}")

    def on_prompt(self, t):
        pass
        # if not t:
        #     return
        # try:
        #     t = float(t)
        # except Exception:
        #     logger.error("Invalid time %s" % str(t))
        #     return
        # if t:
        #     self.goto(t)

    def show(self):
        run()


if __name__ == '__main__':
    n_channels = 385
    dtype = np.int16
    sample_rate = 30_000

    viewer = RawEphysViewer(n_channels, sample_rate, dtype)

    if 1:
        # Load from HTTP.
        viewer.load_session('d33baf74-263c-4b37-a0d0-b79dcb80a764')
        viewer.create()
        viewer.show()
    else:
        # Load from disk.
        path = Path(__file__).parent / "raw_ephys.bin"
        viewer.memmap_file(path)
        viewer.create()
        viewer.show()
