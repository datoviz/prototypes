"""
Python example of an interactive raw ephys data viewer.
"""

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

from functools import lru_cache
import logging
from pathlib import Path
import sys

from joblib import Memory
import numpy as np
import numpy.random as nr

from ibllib.atlas import AllenAtlas
# from ibllib.io.spikeglx import download_raw_partial
from ibllib.dsp import voltage
from one.api import ONE
from ibllib.pipes.ephys_alignment import EphysAlignment
from brainbox.io.spikeglx import stream
from brainbox.io.one import load_channels_from_insertion
from ibllib.ephys.neuropixel import SITES_COORDINATES


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

    return url_cbin, url_ch


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
SAMPLE_SKIP = 0  # DEBUG. 200  # Skip beginning for show, otherwise blurry due to filter


@memory.cache
def _load_spikes(probe_id):
    one = ONE()
    dtypes = [
        'spikes.times', 'spikes.amps', 'spikes.clusters', 'spikes.depths']
    dsets = one.alyx.rest('datasets', 'list', probe_insertion=probe_id)
    dsets_int = [[d for d in dsets if d['dataset_type'] in _][0]
                 for _ in dtypes]

    st, sa, sc, sd = (
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


@memory.cache
def _load_brain_regions(eid, probe_idx=0):
    one = ONE()
    ba = AllenAtlas()

    probe = 'probe0%d' % probe_idx
    ins = one.alyx.rest('insertions', 'list', session=eid, name=probe)[0]

    xyz_chans = load_channels_from_insertion(
        ins, depths=SITES_COORDINATES[:, 1], one=one, ba=ba)
    region, region_label, region_color, _ = EphysAlignment.get_histology_regions(
        xyz_chans, SITES_COORDINATES[:, 1], brain_atlas=ba)
    return region, region_label, region_color


class Model:
    def __init__(self, eid, probe_id, probe_idx=0, one=None):
        self.eid = eid
        self.probe_id = probe_id
        self.probe_idx = probe_idx
        assert one
        self.one = one

        self._download_chunk = lru_cache(self._download_chunk)

        # Ephys data
        logger.info(
            f"Downloading first chunk of ephys data {eid}, probe #{probe_idx}")
        info, arr = self._download_chunk(0)
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

        # Brain regions.
        r, rl, rc = _load_brain_regions(eid, probe_idx)
        self.regions = Bunch(r=r, rl=rl, rc=rc)

    # return tuple (info, array)
    def _download_chunk(self, chunk_idx):
        # url_cbin, url_ch = get_data_urls(eid, probe_idx=probe_idx, one=self.one)
        # reader = download_raw_partial(
        #     url_cbin, url_ch, chunk_idx, chunk_idx)
        # return reader._raw.cmeta, reader[:]
        try:
            sr, t0 = stream(self.probe_id, chunk_idx, nsecs=1, one=self.one)
        except BaseException as e:
            print(f'PID {probe_id} : recording shorter than {int(chunk_idx / 60.0)} min')
            return
        raw = sr[:, :-1].T
        destripe = voltage.destripe(raw, fs=sr.fs)
        X = destripe[:, :].T  # :int(DISPLAY_TIME * sr.fs)].T
        Xs = X[SAMPLE_SKIP:]  # Remove artifact at begining
        # Tplot = Xs.shape[1] / sr.fs
        info = sr._raw.cmeta
        return info, Xs

    def get_chunk(self, chunk_idx):
        return self._download_chunk(chunk_idx)[1]

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
        return s, idx

    def get_spike_pos_colors(self, s, idx):
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
        self.v_spikes = self.panel.visual('line_strip')
        self.v_spikes.data('pos', np.zeros((2, 3)))

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
        self.v_spikes.data('pos', p)
        self.v_spikes.data('color', color)

    def set_vert(self, x0, x1):
        self.v_vert.data('pos', np.array([
            [x0, self.ymin, 0], [x0, self.ymax, 0],
            [x1, self.ymin, 0], [x1, self.ymax, 0],
        ]))

    def change_marker_size(self, x):
        assert 0 <= x and x <= 30
        self.v_point.data('ms', np.array([x]))

    def change_colors(self, sc):
        self.v_point.data('color', sc)


class EphysView:
    def __init__(self, canvas, panel, n_channels, dmin, dmax):
        self.canvas = canvas
        self.panel = panel
        self.dmin = dmin
        self.dmax = dmax
        self.colors = None
        self.alpha = None

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

        self.v_spikes = None

        self.set_xrange(0, 1)
        self._set_tex_coords(1)

    def _set_tex_coords(self, x=1):
        # Top left, top right, bottom right, bottom left
        self.v_image.data('texcoords', np.atleast_2d([0, 0]), idx=0)
        self.v_image.data('texcoords', np.atleast_2d([0, 1]), idx=1)
        self.v_image.data('texcoords', np.atleast_2d([x, 1]), idx=2)
        self.v_image.data('texcoords', np.atleast_2d([x, 0]), idx=3)

    def set_xrange(self, t0, t1):
        # Top left, top right, bottom right, bottom left
        self.v_image.data('pos', np.array([[t0, self.dmin, 0]]), idx=0)
        self.v_image.data('pos', np.array([[t1, self.dmin, 0]]), idx=1)
        self.v_image.data('pos', np.array([[t1, self.dmax, 0]]), idx=2)
        self.v_image.data('pos', np.array([[t0, self.dmax, 0]]), idx=3)

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

    def _set_colors(self, colors):
        if self.alpha is not None:
            colors[:, 3] = np.clip(int(self.alpha * 255), 0, 255)
        self.v_spikes.data('color', colors)

    def show_spikes(self, times, depths, colors):
        n = len(times)
        if n == 0:
            return
        if self.v_spikes is None:
            self.v_spikes = self.panel.visual('rectangle')
        p = np.zeros((n, 3))
        p[:, 0] = times
        p[:, 1] = depths

        k = np.array([[.001, 100, 0]])

        self.v_spikes.data('pos', p - k, idx=0)
        self.v_spikes.data('pos', p + k, idx=1)
        self._set_colors(colors)
        self.colors = colors

    def change_colors(self, colors):
        self._set_colors(colors)
        self.colors = colors

    def change_alpha(self, alpha):
        colors = self.colors.copy()
        self.alpha = alpha
        self._set_colors(colors)
        self.v_spikes.data('color', colors)


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
        sc = self.m.d.spike_colors
        imin = np.searchsorted(st, self.t0)
        imax = np.searchsorted(st, self.t1)
        times = st[imin:imax]
        depths = sd[imin:imax]
        colors = sc[imin:imax].copy()
        colors[:, 3] = 128
        self.ev.show_spikes(times, depths, colors)

        # Update the image.
        self.img = img
        self.ev.set_image(self.img)

        # self.ev.hide_line()

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
        self._make_slider_spikes(ctrl.ev)
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
            'slider_int', 'cluster', vmin=cmin - 1, vmax=cmax)

        @self._slider_cluster.connect
        def on_cluster_change(cl):
            # Highlight clusters in raster view.
            s, idx = self.m.get_cluster_spikes(cl)
            e = self.m.get_spike_pos_colors(s, idx)
            if e:
                _, _, color = e
                sc = self.m.d.spike_colors.copy()
                sc[~idx] = [128, 128, 128, 32]
                raster_view.change_colors(sc)
            else:
                raster_view.change_colors(self.m.d.spike_colors)

    def _make_slider_range(self, vmin, vmax):
        # Slider controlling the imshow value range.
        self._slider_range = self._gui.control(
            'slider_float2', 'vrange', vmin=vmin, vmax=vmax)

        @self._slider_range.connect
        def on_vrange(i, j):
            self.ctrl.set_vrange(i, j)

    def _make_slider_spikes(self, ephys_view):
        # Slider controlling the marker size.
        self._slider_spikes = self._gui.control(
            'slider_float', 'spikes opacity', vmin=0, vmax=1)

        @self._slider_spikes.connect
        def on_opacity_change(x):
            ephys_view.change_alpha(x)

    def _make_button_filter(self, ctrl):
        self._button_filter = self._gui.control('button', 'next filter')

        @self._button_filter.connect
        def on_click(e):
            ctrl.next_filter()


# -------------------------------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------------------------------

def get_eid_default():
    # return 'f25642c6-27a5-4a97-9ea0-06652db79fbd', 'bebe7c8f-0f34-4c3a-8fbb-d2a5119d2961'
    return '15948667-747b-4702-9d53-354ac70e9119', '4e6dfe08-cab0-4a05-903b-94283cb9f8e7'


def get_eid_one(probe_idx=0):
    one = ONE()
    insertions = one.alyx.rest(
        'insertions', 'list', dataset_type='channels.mlapdv')
    probe_id = insertions[probe_idx]['id']
    eid = insertions[probe_idx]['session_info']['id']
    return eid, probe_id


def get_eid_argv():
    if len(sys.argv) <= 1:
        return get_eid_default()
    eid = sys.argv[1]
    probe_idx = int(sys.argv[2]) if len(sys.argv) == 3 else 0
    probe = 'probe0%d' % probe_idx
    logger.info("Finding insertion id #%d for eid %s...", probe_idx, eid)
    one = ONE()
    ins = one.alyx.rest('insertions', 'list', session=eid, name=probe)[0]
    probe_id = ins['id']
    logger.info("Found insertion id: %s", probe_id)
    return eid, probe_id


def plot_brain_regions(panel, regions):
    r = regions.r
    rc = regions.rc

    n = r.shape[0]
    p0 = np.zeros((n, 3))
    p0[:, 0] = 0
    p0[:, 1] = r[:, 0]

    p1 = np.zeros((n, 3))
    p1[:, 0] = 1
    p1[:, 1] = r[:, 1]

    v = panel.visual('rectangle')
    v.data('pos', p0, idx=0)
    v.data('pos', p1, idx=1)
    v.data('color', np.c_[rc, 255 * np.ones(n)].astype(np.uint8))


if __name__ == '__main__':
    eid, probe_id = get_eid_argv()
    one = ONE()
    m = Model(eid, probe_id, probe_idx=0, one=one)

    # Create the Datoviz view.
    c = canvas(width=1200, height=800, show_fps=True)
    scene = c.scene(rows=2, cols=2)

    # Panels.
    p0 = scene.panel(row=0, controller='axes', hide_grid=False)
    p1 = scene.panel(row=1, controller='axes', hide_grid=True)

    # Brain regions in the right panels.
    ps0 = scene.panel(row=0, col=1, controller='axes', hide_grid=True)
    ps1 = scene.panel(row=1, col=1, controller='axes', hide_grid=True)
    ps0.size('x', .2)

    plot_brain_regions(ps0, m.regions)
    plot_brain_regions(ps1, m.regions)

    # Views.
    rv = RasterView(c, p0)
    ev = EphysView(c, p1, m.n_channels, m.depth_min, m.depth_max)

    # Controller
    ctrl = Controller(m, rv, ev)
    gui = GUI(ctrl)

    run()
