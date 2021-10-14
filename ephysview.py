"""
Python example of an interactive raw ephys data viewer.
"""

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import logging
from pathlib import Path

from joblib import Memory
import numpy as np

from ibllib.atlas import AllenAtlas
from ibllib.io.spikeglx import download_raw_partial
from one.api import ONE

from datoviz import canvas, run, colormap, colorpal


logger = logging.getLogger('datoviz')
logger.setLevel('DEBUG')
logger.propagate = False


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


def get_data_urls(eid, probe_idx=0, one=None):
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


# -------------------------------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------------------------------

class Bunch(dict):
    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self


class SpikeData(Bunch):
    def __init__(self, spike_times, spike_clusters, spike_depths, spike_colors):
        self.spike_times = spike_times
        self.spike_clusters = spike_clusters
        self.spike_depths = spike_depths
        self.spike_colors = spike_colors


location = Path('~/.one_cache/').expanduser()
memory = Memory(location, verbose=0)


@memory.cache
def _load_spikes(probe_id):
    one = ONE()
    dtypes = [
        'spikes.times', 'spikes.amps', 'spikes.clusters', 'spikes.depths',
        'clusters.brainLocationIds_ccf_2017']
    dsets = one.alyx.rest('datasets', 'list', probe_insertion=probe_id)
    dsets_int = [[d for d in dsets if d['dataset_type'] in _][0]
                 for _ in dtypes]
    st, sa, sc, sd, cr = (
        np.load(_) for _ in one._download_datasets(dsets_int) if str(_).endswith('.npy'))

    sd[np.isnan(sd)] = sd[~np.isnan(sd)].min()

    color = colorpal(sc.astype(np.int32), cpal='glasbey')

    # assert 100 < len(cr) < 1000
    # # Brain region colors
    # atlas = AllenAtlas(25)
    # n = len(atlas.regions.rgb)
    # alpha = 255 * np.ones((n, 1))
    # rgb = np.hstack((atlas.regions.rgb, alpha)).astype(np.uint8)
    # spike_regions = cr[sc]
    # # HACK: spurious values
    # spike_regions[spike_regions > 2000] = 0
    # color = rgb[spike_regions]

    return SpikeData(st, sc, sd, color)


class Model:
    def __init__(self, eid, probe_id, probe_idx=0, one=None):
        self.eid = eid
        self.probe_id = probe_id
        self.probe_idx = probe_idx
        self.one = one

        self._download_chunk = memory.cache(self._download_chunk)

        # Ephys data
        logger.info(
            f"Downloading first chunk of ephys data {eid}, probe #{probe_idx}")
        info, arr = self._download_chunk(eid, probe_idx=probe_idx, chunk_idx=0)
        assert info
        assert arr.size

        self.n_samples = info.chopped_total_samples
        assert self.n_samples > 0

        self.n_channels = arr.shape[1]
        assert self.n_channels > 0

        self.sample_rate = float(info.sample_rate)
        assert self.sample_rate > 1000

        self.duration = self.n_samples / self.sample_rate
        assert self.duration > 0
        logger.info(
            f"Downloaded first chunk of ephys data "
            f"{self.n_samples=}, {self.n_channels=}, {self.duration=}")

        # Spike data.
        self.d = _load_spikes(probe_id)
        self.depth_min = self.d.spike_depths.min()
        self.depth_max = self.d.spike_depths.max()
        assert self.depth_min < self.depth_max
        logger.info(f"Loaded {len(self.d.spike_times)} spikes")

    # return tuple (info, array)
    def _download_chunk(self, eid, probe_idx=0, chunk_idx=0):
        one = ONE()
        url_cbin, url_ch, url_meta = get_data_urls(
            eid, probe_idx=probe_idx, one=one)
        reader = download_raw_partial(
            url_cbin, url_ch, url_meta, chunk_idx, chunk_idx)
        return reader._raw.cmeta, reader[:]

    def get_chunk(self, chunk_idx):
        return self._download_chunk(self.eid, self.probe_idx, chunk_idx=chunk_idx)[1]

    def _get_range_chunks(self, t0, t1):
        # Chunk idxs, assuming 1 second chunk
        i0 = int(t0)  # in seconds
        i1 = int(t1)  # in seconds

        assert i0 >= 0
        assert i0 <= t0
        assert i1 <= t1
        assert i1 < self.n_samples

        return i0, i1

    def get_data(self, t0, t1, filter=None):  # float32
        t0 = np.clip(t0, 0, self.duration)
        t1 = np.clip(t1, 0, self.duration)
        assert t0 < t1
        expected_samples = int(round((t1 - t0) * self.sample_rate))

        # Find the chunks.
        i0, i1 = self._get_range_chunks(t0, t1)

        # Download the chunks.
        arr = np.vstack([self.get_chunk(i) for i in range(i0, i1 + 1)])
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

        # HACK: the last column seems corrupted
        out[:, -1] = out[:, -2]
        return out

    def spikes_in_range(self, t0, t1):
        imin = np.searchsorted(self.d.spike_times, t0)
        imax = np.searchsorted(self.d.spike_times, t1)
        return imin, imax

    def get_cluster_spikes(self, cl, t0_t1=None):
        # Select spikes in the given time range, or all spikes.
        if t0_t1 is not None:
            i0, i1 = self.spikes_in_range(*t0_t1)
            s = slice(i0, i1, 1)
        else:
            s = slice(None, None, None)

        # Select the spikes from the requested cluster within the time range.
        sc = self.d.spike_clusters[s]
        idx = sc == cl
        if np.sum(idx) == 0:
            return

        # x and y coordinates of the spikes.
        x = self.d.spike_times[s][idx]
        y = self.d.spike_depths[s][idx]

        # Color of the first spike.
        i = np.nonzero(idx)[0][0]
        color = self.d.spike_colors[s][i]

        return x, y, color


# -------------------------------------------------------------------------------------------------
# Views
# -------------------------------------------------------------------------------------------------

class RasterView:
    def __init__(self, canvas, panel):
        self.canvas = canvas
        self.panel = panel
        self.v_point = self.panel.visual('point')

        # Cluster line.
        self.v_line = self.panel.visual('line_strip')
        self.v_line.data('pos', np.zeros((2, 3)))

        # Vertical lines.
        self.v_vert = self.panel.visual('path')
        self.v_vert.data('length', np.array([2, 2]))

    def show_spikes(self, spike_times, spike_clusters, spike_depths, spike_colors, ms=2):
        self.ymin = spike_depths.min()
        self.ymax = spike_depths.max()

        N = len(spike_times)
        assert spike_times.shape == spike_depths.shape == spike_clusters.shape

        self.cluster_ids = np.unique(spike_clusters)

        pos = np.c_[spike_times, spike_depths, np.zeros(N)]

        self.v_point.data('pos', pos)
        self.v_point.data('color', spike_colors)
        self.v_point.data('ms', np.array([ms]))

        self.set_vert(0, 0.1)

    def show_line(self, x, y, color):
        p = np.c_[x, y, np.zeros(len(x))]
        self.v_line.data('pos', p)
        self.v_line.data('color', color)

    def set_vert(self, x0, x1):
        self.v_vert.data('pos', np.array([
            [x0, self.ymin, 0], [x0, self.ymax, 0],
            [x1, self.ymin, 0], [x1, self.ymax, 0],
        ]))

    def change_marker_size(self, x):
        assert 0 <= x and x <= 30
        self.v_point.data('ms', np.array([x]))


class EphysView:
    def __init__(self, canvas, panel, n_channels):
        self.canvas = canvas
        self.panel = panel

        assert n_channels > 0
        self.n_channels = n_channels

        self.n_samples_tex = 3000
        self.tex = canvas.gpu().context().texture(
            self.n_samples_tex, n_channels, dtype=np.dtype(np.uint8), ndim=2, ncomp=4)
        # Placeholder for the data so as to keep the data to upload in memory.
        self._arr = np.empty(
            (self.n_samples_tex, n_channels, 4), dtype=np.uint8)

        # Image visual
        self.v_image = self.panel.visual('image')
        self.v_image.texture(self.tex)

        self.set_xrange(0, 1)
        self._set_tex_coords(1)

        # Cluster line.
        self.v_line = self.panel.visual('path')
        self.v_line.data('pos', np.zeros((2, 3)))

    def _set_tex_coords(self, x=1):
        # Top left, top right, bottom right, bottom left
        self.v_image.data('texcoords', np.atleast_2d([0, 0]), idx=0)
        self.v_image.data('texcoords', np.atleast_2d([0, 1]), idx=1)
        self.v_image.data('texcoords', np.atleast_2d([x, 1]), idx=2)
        self.v_image.data('texcoords', np.atleast_2d([x, 0]), idx=3)

    def set_xrange(self, t0, t1):
        # Top left, top right, bottom right, bottom left
        self.v_image.data('pos', np.array([[t0, 0, 0]]), idx=0)
        self.v_image.data('pos', np.array([[t1, 0, 0]]), idx=1)
        self.v_image.data('pos', np.array([[t1, self.n_channels, 0]]), idx=2)
        self.v_image.data('pos', np.array([[t0, self.n_channels, 0]]), idx=3)

    def set_image(self, img):
        assert img.ndim == 3
        assert img.shape[2] == 4
        assert img.dtype == np.uint8
        # Resize the texture if needed.
        if self.tex.shape != img.shape:
            self.tex.resize(img.shape[0], img.shape[1])
            self._arr.resize(img.shape)
        assert self.tex.shape == img.shape
        assert self._arr.shape == img.shape
        assert self._arr.dtype == img.dtype
        self._arr[:] = img[:]
        self.tex.upload(self._arr)

    def show_line(self, x, y, color):
        p = np.c_[x, y, np.zeros(len(x))]
        self.v_line.data('pos', p)
        self.v_line.data('color', color)


# -------------------------------------------------------------------------------------------------
# Controller
# -------------------------------------------------------------------------------------------------

class Controller:
    _is_fetching = False
    _cur_filter_idx = 0
    vmin = None
    vmax = None
    data = None
    data_f = None
    img = None
    t0 = 0
    t1 = 1

    def __init__(self, model, raster_view, ephys_view):
        assert model
        assert raster_view
        assert ephys_view
        assert model.n_channels > 0
        assert model.depth_min < model.depth_max

        self.canvas = raster_view.canvas
        self.m = model
        self.rv = raster_view
        self.ev = ephys_view

        # Raster.
        self.show_spikes()

        # Raw data filters.
        self.filters = [None]

        # Raw data.
        self.set_range(0, .1)
        assert self.vmin is not None
        assert self.vmax is not None

        # Callbacks
        self.scene = self.canvas.scene()
        self.canvas.connect(self.on_mouse_click)
        self.canvas.connect(self.on_key_press)

        @self.add_filter
        def my_filter(data):
            return data - np.median(data, axis=1).reshape((-1, 1))

    def show_spikes(self):
        self.rv.show_spikes(
            self.m.d.spike_times,
            self.m.d.spike_clusters,
            self.m.d.spike_depths,
            self.m.d.spike_colors)

    def highlight_area(self, img, it0, it1, ic0, ic1, color):
        it0 = np.clip(it0, 0, self.m.n_samples - 1)
        it1 = np.clip(it1, 0, self.m.n_samples - 1)
        ic0 = np.clip(ic0, 0, self.m.n_channels - 1)
        ic1 = np.clip(ic1, 0, self.m.n_channels - 1)
        img[it0:it1, ic0:ic1, :3] = (
            img[it0:it1, ic0:ic1, :3] * color).astype(img.dtype)
        return img

    def highlight_spike(self, img, t, depth, color):
        if t < self.t0 or t > self.t1:
            logger.debug(
                "Spike to be highlighted is beyond the bounds of the current data area")
            return

        dm, dM = self.m.depth_min, self.m.depth_max
        x = (depth - dm) / (dM - dm)
        assert 0 <= x <= 1
        ic = int(round(x * (self.m.n_channels - 1)))

        t = (t - self.t0) / float(self.t1 - self.t0)
        assert 0 <= t <= 1
        it = int(round(t * (self.ev.n_samples_tex - 1)))

        nc = 5
        nt = 15
        return self.highlight_area(img, it - nt, it + nt, ic - nc, ic + nc, color)

    def to_image(self, data):
        # CAR
        data -= data.mean(axis=0)

        # Vrange
        self.vmin = data.min() if self.vmin is None else self.vmin
        self.vmax = data.max() if self.vmax is None else self.vmax

        # Colormap
        img = colormap(data.ravel().astype(np.double),
                       vmin=self.vmin, vmax=self.vmax, cmap='gray')
        img = img.reshape(data.shape + (-1,))
        assert img.shape == data.shape[:2] + (4,)

        return img

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
        logger.info("Set time range %.3f %.3f" % (t0, t1))

        # # Update slider only when changing the time by using another method than the slider.
        # if self._do_update_control:
        #     self._update_control()

        # Update the positions.
        self.ev.set_xrange(t0, t1)

        # Filter.
        self.data = self.m.get_data(t0, t1)
        self.data_f = self.apply_filter(self.data)

        # Apply colormap.
        img = self.to_image(self.data_f)

        # Highlight the spikes.
        st = self.m.d.spike_times
        sd = self.m.d.spike_depths

        imin = np.searchsorted(st, self.t0)
        imax = np.searchsorted(st, self.t1)
        for i in range(imin, imax):
            t = st[i]
            d = sd[i]
            color = self.m.d.spike_colors[i]
            color = color[:3] / 255.0
            img = self.highlight_spike(img, t, d, color)

        # Update the image.
        self.img = img
        self.ev.set_image(self.img)

    def update_ephys_view(self):
        self.set_range(self.t0, self.t1)

    def set_vrange(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax
        self.update_ephys_view()

    # Time navigation
    # ---------------------------------------------------------------------------------------------

    def _update_time(self, t0, t1):
        self.set_range(t0, t1)
        self.rv.set_vert(t0, t1)

    def go_left(self, shift):
        d = self.t1 - self.t0
        t0 = self.t0 - shift
        t1 = self.t1 - shift
        if t0 < 0:
            t0 = 0
            t1 = d
        assert abs(t1 - t0 - d) < 1e-6
        self._update_time(t0, t1)

    def go_right(self, shift):
        t0 = self.t0 + shift
        t1 = self.t1 + shift
        if t1 > self.m.duration:
            t0 = self.m.duration - shift
            t1 = self.m.duration
        self._update_time(t0, t1)

    def go_to(self, t):
        d = self.t1 - self.t0
        t0 = t - d / 2
        t1 = t + d / 2
        self.set_range(t0, t1)

    # Filters
    # ---------------------------------------------------------------------------------------------

    def add_filter(self, f):
        self.filters.append(f)

    def next_filter(self):
        self._cur_filter_idx = (self._cur_filter_idx + 1) % len(self.filters)
        self.update_ephys_view()

    def apply_filter(self, arr):
        f = self.filters[self._cur_filter_idx % len(self.filters)]
        logger.info(f"Apply filter {(f.__name__ if f else 'default')}")
        if not f:
            return arr
        arr_f = f(arr)
        assert arr_f.dtype == arr.dtype
        assert arr_f.shape == arr.shape
        return arr_f

    # Event callbacks
    # ---------------------------------------------------------------------------------------------

    def on_mouse_click(self, x, y, button=None, modifiers=()):
        if not modifiers:
            return
        p = self.scene.panel_at(x, y)
        if p == self.rv.panel:
            xd, yd = p.pick(x, y)
            self.rv.set_vert(xd - .05, xd + .05)
            self.go_to(xd)

    def on_key_press(self, key, modifiers=()):
        k = .1
        if key == 'left':
            self.go_left(k * (self.t1 - self.t0))
        if key == 'right':
            self.go_right(k * (self.t1 - self.t0))
        if key == 'home':
            self.set_range(0, self.t1 - self.t0)
        if key == 'end':
            self.set_range(self.m.duration -
                           (self.t1 - self.t0), self.m.duration)
        if key == 'f':
            self.next_filter()


# -------------------------------------------------------------------------------------------------
# GUI
# -------------------------------------------------------------------------------------------------

class GUI:
    def __init__(self, ctrl):
        self.ctrl = ctrl
        self.c = ctrl.rv.canvas
        self.m = ctrl.m

        self._gui = self.c.gui("GUI")
        self._make_slider_ms(ctrl.rv)
        self._make_slider_cluster(
            ctrl.rv, ctrl.ev, ctrl.m.d.spike_clusters.min(), ctrl.m.d.spike_clusters.max())
        self._make_slider_range(ctrl.vmin, ctrl.vmax)
        self._make_button_filter(ctrl)

    def _make_slider_ms(self, raster_view):
        # Slider controlling the marker size.
        self._slider_ms = self._gui.control(
            'slider_float', 'marker size', vmin=.1, vmax=30)

        @self._slider_ms.connect
        def on_ms_change(x):
            raster_view.change_marker_size(x)

    def _make_slider_cluster(self, raster_view, ephys_view, cmin, cmax):
        # Slider controlling the cluster to highlight.
        self._slider_cluster = self._gui.control(
            'slider_int', 'cluster', vmin=cmin, vmax=cmax)

        @self._slider_cluster.connect
        def on_cluster_change(cl):
            # Cluster line in raster view.
            x, y, color = self.m.get_cluster_spikes(cl)
            raster_view.show_line(x, y, color)

            # Cluster line in ephys view.
            t0, t1 = self.ctrl.t0, self.ctrl.t1
            e = self.m.get_cluster_spikes(cl, (t0, t1))
            if e:
                ephys_view.show_line(*e)

    def _make_slider_range(self, vmin, vmax):
        # Slider controlling the imshow value range.
        self._slider_range = self._gui.control(
            'slider_float2', 'vrange', vmin=vmin, vmax=vmax)

        @self._slider_range.connect
        def on_vrange(i, j):
            self.ctrl.set_vrange(i, j)

    def _make_button_filter(self, ctrl):
        self._button_filter = self._gui.control('button', 'next filter')

        @self._button_filter.connect
        def on_click(e):
            ctrl.next_filter()


# -------------------------------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------------------------------

def get_eid(probe_idx=0):
    return 'f25642c6-27a5-4a97-9ea0-06652db79fbd', 'bebe7c8f-0f34-4c3a-8fbb-d2a5119d2961'

    one = ONE()
    insertions = one.alyx.rest(
        'insertions', 'list', dataset_type='channels.mlapdv')
    insertion_id = insertions[probe_idx]['id']
    return insertion_id, insertions[probe_idx]['session_info']['id']


if __name__ == '__main__':
    probe_idx = 0
    eid, probe_id = get_eid(probe_idx)
    m = Model(eid, probe_id, probe_idx=0)

    # Create the Datoviz view.
    c = canvas(width=1600, height=1200, show_fps=True)
    scene = c.scene(rows=2, cols=1)

    # Panels.
    p0 = scene.panel(row=0, controller='axes', hide_grid=False)
    p1 = scene.panel(row=1, controller='axes', hide_grid=True)

    # Views.
    rv = RasterView(c, p0)
    ev = EphysView(c, p1, n_channels=m.n_channels)

    # Controller
    ctrl = Controller(m, rv, ev)
    gui = GUI(ctrl)

    run()
