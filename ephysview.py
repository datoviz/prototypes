"""
Python example of an interactive raw ephys data viewer.
"""

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import logging
import math
from pathlib import Path

from joblib import Memory
import numpy as np

from ibllib.atlas import AllenAtlas
from ibllib.io.spikeglx import download_raw_partial
from one.api import ONE

from datoviz import canvas, run, colormap


logger = logging.getLogger(__name__)

one = ONE()



# -------------------------------------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------------------------------------

def _index_of(arr, lookup):
    lookup = np.asarray(lookup, dtype=np.int32)
    m = (lookup.max() if len(lookup) else 0) + 1
    tmp = np.zeros(m + 1, dtype=np.int)
    # Ensure that -1 values are kept.
    tmp[-1] = -1
    if len(lookup):
        tmp[lookup] = np.arange(len(lookup))
    return tmp[arr]


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

    # Find URL to .meta file
    dsets = one.alyx.rest(
        'datasets', 'list', session=eid,
        django='name__icontains,ap.meta,collection__endswith,probe%02d' % probe_idx)
    for fr in dsets[0]['file_records']:
        if fr['data_url']:
            url_meta = fr['data_url']

    return url_cbin, url_ch, url_meta


one = ONE()

# Disk cache of the downloading
location = Path('~/.one_cache/').expanduser()
memory = Memory(location, verbose=0)

@memory.cache
def _dl(url_cbin, url_ch, url_meta, chunk_idx):
    reader = download_raw_partial(url_cbin, url_ch, url_meta, chunk_idx, chunk_idx)
    return reader._raw.cmeta, reader[:]


def _is_cached(f, args, kwargs):
    func_id, args_id = f._get_output_identifiers(*args, **kwargs)
    return (f._check_previous_func_code(stacklevel=4)
            and f.store_backend.contains_item([func_id, args_id]))


@memory.cache
def _load_spikes(probe_id):
    dtypes = [
        'spikes.times', 'spikes.amps', 'spikes.clusters', 'spikes.depths',
        'clusters.brainLocationIds_ccf_2017']
    dsets = one.alyx.rest('datasets', 'list', probe_insertion=probe_id)
    dsets_int = [[d for d in dsets if d['dataset_type'] in _][0] for _ in dtypes]
    return [np.load(_) for _ in one._download_datasets(dsets_int) if str(_).endswith('.npy')]



# -------------------------------------------------------------------------------------------------
# Raster viewer
# -------------------------------------------------------------------------------------------------

class RasterModel:
    def __init__(self, probe_id):
        self.st, self.sa, self.sc, self.sd, self.cr = _load_spikes(probe_id)
        n = 10000
        self.st = self.st[:n]
        self.sa = self.sa[:n]
        self.sc = self.sc[:n]
        self.sd = self.sd[:n]
        self.cr = self.cr[:n]
        print(f"Loaded {len(self.st)} spikes")


class RasterView:
    def __init__(self, canvas, panel):
        self.canvas = canvas
        self.panel = panel
        self.v_point = self.panel.visual('point')

        self.atlas = AllenAtlas(25)

    def set_spikes(self, spike_times, spike_clusters, spike_depths, brain_regions):
        N = len(spike_times)
        assert spike_times.shape == spike_depths.shape == spike_clusters.shape
        assert 100 < len(brain_regions) < 1000

        pos = np.c_[spike_times, spike_depths, np.zeros(N)]

        # Brain region colors
        n = len(self.atlas.regions.rgb)
        alpha = 255 * np.ones((n, 1))
        rgb = np.hstack((self.atlas.regions.rgb, alpha)).astype(np.uint8)
        spike_regions = brain_regions[spike_clusters]
        # HACK: spurious values
        spike_regions[spike_regions > 2000] = 0
        color = rgb[spike_regions]

        self.v_point.data('pos', pos)
        self.v_point.data('color', color)
        self.v_point.data('ms', np.array([3.]))


class RasterController:
    _time_select_cb = None

    def __init__(self, model, view):
        self.m = model
        self.v = view
        self.canvas = view.canvas

        # Callbacks
        self.scene = self.canvas.scene()
        self.canvas.connect(self.on_mouse_click)

    def set_data(self):
        self.v.set_spikes(self.m.st, self.m.sc, self.m.sd, self.m.cr)

    def on_mouse_click(self, x, y, button=None, modifiers=()):
        if not modifiers:
            return
        p = self.scene.panel_at(x, y)
        if p != self.v.panel:
            return
        xd, yd = p.pick(x, y)
        if self._time_select_cb is not None:
            self._time_select_cb(xd)

    def on_time_select(self, f):
        self._time_select_cb = f



# -------------------------------------------------------------------------------------------------
# Raw data viewer
# -------------------------------------------------------------------------------------------------

class RawDataModel:
    def __init__(self, eid, probe_idx=0):
        self.eid = eid
        self.probe_idx = probe_idx
        self.url_cbin, self.url_ch, self.url_meta = get_data_urls(eid, probe_idx=probe_idx)
        assert self.url_cbin
        assert self.url_ch
        assert self.url_meta
        # Read the first chunk to get the total number of samples.
        info, _ = _dl(self.url_cbin, self.url_ch, self.url_meta, 0)
        self.n_samples = info.chopped_total_samples
        self.n_channels = _.shape[1]
        self.sample_rate = float(info.sample_rate)
        self.duration = self.n_samples / self.sample_rate
        print(f"Load raw data {self.n_samples=}, {self.n_channels=}, {self.duration=}")

        assert self.n_samples > 0
        assert self.n_channels > 0
        assert self.sample_rate > 0
        assert self.duration > 0

    def _download_chunk(self, chunk_idx):
        _, arr = _dl(self.url_cbin, self.url_ch, self.url_meta, chunk_idx)
        return arr

    def _get_range_chunks(self, t0, t1):
        # Chunk idxs, assuming 1 second chunk
        i0 = int(t0)  # in seconds
        i1 = int(t1)  # in seconds

        assert i0 >= 0
        assert i0 <= t0
        assert i1 <= t1
        assert i1 < self.n_samples

        return i0, i1

    def is_chunk_cached(self, chunk_idx):
        return _is_cached(_dl, (self.url_cbin, self.url_ch, self.url_meta, chunk_idx), {})

    def get_raw_data(self, t0, t1):
        t0 = np.clip(t0, 0, self.duration)
        t1 = np.clip(t1, 0, self.duration)
        assert t0 < t1
        expected_samples = int(round((t1 - t0) * self.sample_rate))

        # Find the chunks.
        i0, i1 = self._get_range_chunks(t0, t1)

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
    def __init__(self, canvas, panel, n_channels, vrange=()):
        self.canvas = canvas
        self.panel = panel

        assert n_channels > 0
        self.n_channels = n_channels

        # Place holder for the data.
        self.arr = np.zeros((3_000, self.n_channels), dtype=np.uint16)
        self.tex = canvas.gpu().context().texture(
            *self.arr.shape, dtype=self.arr.dtype, ndim=2, ncomp=1)
        self.tex.upload(self.arr)

        # Image cmap visual
        self.v_image = self.panel.visual('image_cmap')
        self.v_image.texture(self.tex)

        self.v_image.data('colormap', np.array([26]))

        # Initialize the POS prop.
        self.set_xrange(0, 1)
        if vrange:
            self.set_vrange(*vrange)
        self._init_vrange = vrange

        self._set_tex_coords(1)

    def _set_tex_coords(self, x=1):
        # Top left, top right, bottom right, bottom left
        self.v_image.data('texcoords', np.atleast_2d([0, 0]), idx=0)
        self.v_image.data('texcoords', np.atleast_2d([0, 1]), idx=1)
        self.v_image.data('texcoords', np.atleast_2d([x, 1]), idx=2)
        self.v_image.data('texcoords', np.atleast_2d([x, 0]), idx=3)

    def set_xrange(self, t0, t1):
        # Top left, top right, bottom right, bottom left
        self.v_image.data('pos', np.array([[t0, self.n_channels, 0]]), idx=0)
        self.v_image.data('pos', np.array([[t1, self.n_channels, 0]]), idx=1)
        self.v_image.data('pos', np.array([[t1, 0, 0]]), idx=2)
        self.v_image.data('pos', np.array([[t0, 0, 0]]), idx=3)

    def set_vrange(self, vmin, vmax):
        pass
        # self.v_image.data('range', np.array([vmin, vmax]))

    def set_image(self, img):
        assert img.ndim == 2
        assert img.shape[0] > 0
        assert img.shape[1] == self.n_channels
        k = .001
        m = np.quantile(img, k)
        M = np.quantile(img, 1-k)
        img_n = (img - m) / (M - m)
        # img_n = (img - np.median(img)) / np.std(img)
        img_n *= 40000
        print(img_n)

        self.tex.upload(img_n.astype(np.uint16))
        self.v_image.data('range', np.array([0, 1]))

        # n = min(img.shape[0], self.arr.shape[0])
        # self.arr[:n, :] = img[:n, :] * 1e7
        # print(self.arr[:n, :])
        # self.tex.upload(self.arr[:n, :])
        # self._set_tex_coords(n / float(self.arr.shape[0]))


class RawDataController:
    _is_fetching = False
    _do_update_control = True
    filters = (None, 'filter_1', 'filter_2')
    _cur_filter_idx = 0

    def __init__(self, model, view):
        self.m = model
        self.v = view
        self.t0 = 0
        self.t1 = 10

        # Callbacks
        self.canvas = view.canvas
        self.canvas.connect(self.on_key_press)

        # GUI
        self.gui = self.canvas.gui("GUI")
        self.input = self.gui.control('input_float', 'time', step=.1, step_fast=1, mode='async')
        self.input.connect(self.on_slider)

        # self.slider = self.gui.control(
        #     'slider_float2', 'vrange', vmin=-.01, vmax=+.01, value=self.v._init_vrange)

        # @self.slider.connect
        # def on_vrange(i, j):
        #     self.v.set_vrange(i, j)

    def set_range(self, t0, t1):
        if self._is_fetching:
            return
        assert t0 < t1
        d = t1 - t0
        assert d > 0
        if t0 < 0:
            t0 = 0
            t1 = d
        if t1 > self.m.duration:
            t1 = self.m.duration
            t0 = t1 - d
        assert abs(t1 - t0 - d) < 1e-6
        assert t0 < t1
        self.t0, self.t1 = t0, t1
        print("Set time range %.3f %.3f" % (t0, t1))

        # Update slider only when changing the time by using another method than the slider.
        if self._do_update_control:
            self._update_control()

        # Update the positions.
        self.v.set_xrange(t0, t1)

        # If fetching from the Internet, clear the array until we have the data.
        i0, i1 = self.m._get_range_chunks(t0, t1)
        arr = np.zeros((int((t1 - t0) * self.m.sample_rate), self.m.n_channels), dtype=np.int16)
        if not self.m.is_chunk_cached(i0) or not self.m.is_chunk_cached(i1):
            self.v.set_image(arr)

        # Fetch the raw data, SLOW when fetching from the Internet.
        self._is_fetching = True
        arr = self.m.get_raw_data(t0, t1)
        self._is_fetching = False

        # CAR
        # TODO: as an independent filter?
        arr -= arr.mean(axis=0).astype(arr.dtype)

        # Filter.
        arr = self.filter(arr)

        # Update the image.
        self.v.set_image(arr)

    def filter_1(self, arr):
        return arr * 2

    def filter_2(self, arr):
        return arr * .5

    def filter(self, arr):
        f = self.filters[self._cur_filter_idx]
        if not f:
            return arr
        print(f"Apply filter {f}")
        return getattr(self, f)(arr)

    def next_filter(self):
        self._cur_filter_idx = (self._cur_filter_idx + 1) % len(self.filters)
        self.set_range(self.t0, self.t1)

    def go_left(self, shift):
        d = self.t1 - self.t0
        t0 = self.t0 - shift
        t1 = self.t1 - shift
        if t0 < 0:
            t0 = 0
            t1 = d
        assert abs(t1 - t0 - d) < 1e-6
        self.set_range(t0, t1)

    def go_right(self, shift):
        d = self.t1 - self.t0
        t0 = self.t0 + shift
        t1 = self.t1 + shift
        if t1 > self.m.duration:
            t0 = self.m.duration - shift
            t1 = self.m.duration
        self.set_range(t0, t1)

    def go_to(self, t):
        d = self.t1 - self.t0
        t0 = t - d / 2
        t1 = t + d / 2
        self.set_range(t0, t1)

    def on_key_press(self, key, modifiers=()):
        if key == 'left':
            self.go_left(.25 * (self.t1 - self.t0))
        if key == 'right':
            self.go_right(.25 * (self.t1 - self.t0))
        if key == 'home':
            self.set_range(0, self.t1 - self.t0)
        if key == 'end':
            self.set_range(self.m.duration - (self.t1 - self.t0), self.m.duration)
        if key == 'f':
            self.next_filter()

    def _update_control(self,):
        # Update the input float value.
        self.input.set(float((self.t0 + self.t1) / 2))

    def on_slider(self, value):
        # HACK: do not update the control programmatically when it's being used with the slider.
        self._do_update_control = False
        self.go_to(value)
        self._do_update_control = True



# -------------------------------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    insertions = one.alyx.rest('insertions', 'list', dataset_type='channels.mlapdv')
    insertion_id = insertions[0]['id']
    session_id = insertions[0]['session_info']['id']

    m_raster = RasterModel(insertion_id)
    m_raw = RawDataModel(session_id)

    # Create the Datoviz view.
    c = canvas(width=1600, height=800, show_fps=True)
    scene = c.scene(rows=1, cols=2)

    # Panels.
    p0 = scene.panel(col=0, controller='axes', hide_grid=False)
    p1 = scene.panel(col=1, controller='axes', hide_grid=True)

    # Raster view.
    v_raster = RasterView(c, p0)
    c_raster = RasterController(m_raster, v_raster)
    c_raster.set_data()

    # Raw data view.
    v_raw = RawDataView(c, p1, m_raw.n_channels, vrange=(+.00231, -.00169))
    c_raw = RawDataController(m_raw, v_raw)
    c_raw.set_range(0, .1)

    # Link between the panels.
    @c_raster.on_time_select
    def on_time_select(t):
        c_raw._update_control()
        c_raw.go_to(t)

    run()
