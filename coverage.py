"""
Seeing coverage volume.
"""

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

from pathlib import Path

import numpy as np

from ibllib.atlas import AllenAtlas
from datoviz import canvas, run, colormap


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


def normalize_volume(v):
    v = v.astype(np.float32)
    v -= v.min()
    v /= v.max()
    assert np.all(0 <= v)
    assert np.all(v <= 1)
    return v


def transpose_volume(v):
    """Return the flattened buffer of a 3D array, using the strides corresponding to
    Vulkan instead of NumPy"""
    x, y, z = v.shape
    its = np.dtype(v.dtype).itemsize
    strides = (z * its, x * z * its, its)
    out = as_strided(v, shape=v.shape, strides=strides).ravel()
    assert out.flags['C_CONTIGUOUS']
    out = out.reshape((y, x, z))
    return out


def save_coverage():
    from oneibl.one import ONE
    from ibllib.pipes.histology import coverage
    from ibllib.atlas import AllenAtlas
    ba = AllenAtlas()

    one = ONE()
    trajs = one.alyx.rest(
        'trajectories', 'list',
        django=(
            'provenance,70,'
            'probe_insertion__session__project__name__icontains,ibl_neuropixel_brainwide_01,'
            'probe_insertion__session__qc__lt,50,'
            'probe_insertion__json__extended_qc__alignment_count__gt,0,'
            'probe_insertion__session__extended_qc__behavior,1'
        )
    )
    vol = coverage(trajs=trajs, ba=ba)
    vol[np.isnan(vol)] = 0
    np.save('coverage.npy', vol)


def save_atlas():
    from ibllib.atlas import AllenAtlas
    ba = AllenAtlas()
    vol = np.ascontiguousarray(ba.image)
    np.save('atlas.npy', vol)


def save_labels():
    from ibllib.atlas import AllenAtlas
    ba = AllenAtlas()
    vol = np.ascontiguousarray(ba.label)
    np.save('label.npy', vol)


# -------------------------------------------------------------------------------------------------
# Atlas model
# -------------------------------------------------------------------------------------------------

class AtlasModel:
    def __init__(self):

        self.atlas = AllenAtlas(25)
        self.xlim = self.atlas.bc.xlim
        self.ylim = self.atlas.bc.ylim[::-1]
        self.zlim = self.atlas.bc.zlim[::-1]

        # Coverage
        if not Path('coverage.npy').exists():
            save_coverage()
        cov = np.load('coverage.npy')
        cov = normalize_volume(cov)
        cov *= 255
        cov = cov.astype(np.uint8)
        self.shape = cov.shape

        # Atlas
        if not Path('label.npy').exists():
            save_labels()
        atlas = np.load('label.npy')

        # Brain region colors
        atlas_idx = _index_of(atlas.ravel(), self.atlas.regions.id).reshape(atlas.shape)
        atlas = self.atlas.regions.rgb[atlas_idx]
        atlas = np.concatenate((atlas, 255 * np.ones((atlas.shape[:3] + (1,)), dtype=atlas.dtype)), axis=3)
        atlas = np.ascontiguousarray(atlas)

        # Merge coverage and atlas
        assert cov.shape == atlas.shape[:3]
        _idx = cov != 0
        atlas[_idx, :] = cov[_idx][:, np.newaxis]

        atlas = np.transpose(atlas, (1, 2, 0, 3))
        self.vol = atlas

        # cov = normalize_volume(cov)
        # atlas = normalize_volume(atlas)

        # atlas += cov
        # atlas = np.clip(atlas, 0, 1)
        # atlas = np.ascontiguousarray(atlas)
        # atlas = np.transpose(atlas, (1, 2, 0))

        # self.vol = atlas



# -------------------------------------------------------------------------------------------------
# Atlas view
# -------------------------------------------------------------------------------------------------

class AtlasView:
    tex = None

    def __init__(self, canvas, panel, tex, axis, xlim, ylim, zlim):
        self.canvas = canvas
        self.panel = panel
        self.axis = axis
        self.tex = tex

        self.visual = panel.visual('volume_slice')
        self.visual.texture(self.tex)

        xmin, xmax = xlim
        ymin, ymax = ylim
        zmin, zmax = zlim

        # Do not use colormap, sample RGBA texture directly from 3D volume.
        self.visual.data('colormap', np.array([-1]))

        # Top left, top right, bottom right, bottom left
        self.visual.data('pos', np.array([xmin, zmax, 0]), idx=0)
        self.visual.data('pos', np.array([xmax, zmax, 0]), idx=1)
        self.visual.data('pos', np.array([xmax, zmin, 0]), idx=2)
        self.visual.data('pos', np.array([xmin, zmin, 0]), idx=3)

        self.update_tex_coords(.5)

    def update_tex_coords(self, w):
        if self.axis == 0:
            i = np.array([0, 1, 2])
        elif self.axis == 1:
            i = np.array([0, 2, 1])

        # Top left, top right, bottom right, bottom left
        self.visual.data('texcoords', np.array([0, 0, w])[i], idx=0)
        self.visual.data('texcoords', np.array([0, 1, w])[i], idx=1)
        self.visual.data('texcoords', np.array([1, 1, w])[i], idx=2)
        self.visual.data('texcoords', np.array([1, 0, w])[i], idx=3)



# -------------------------------------------------------------------------------------------------
# Atlas controller
# -------------------------------------------------------------------------------------------------

class AtlasController:
    def __init__(self, model):
        self.m = model

        # Canvas.
        self.canvas = canvas(cols=2, show_fps=True, width=1600, height=700, clear_color='black')

        # Shared 3D texture.
        self.tex = self.canvas.volume(self.m.vol)


        # Left panel.
        self.p0 = self.canvas.panel(col=0, controller='axes', hide_grid=True)
        assert self.p0.col == 0

        vargs = (self.m.xlim, self.m.ylim, self.m.zlim)

        # Left view.
        self.view0 = AtlasView(self.canvas, self.p0, self.tex, 0, *vargs)


        # Right panel.
        self.p1 = self.canvas.panel(col=1, controller='axes', hide_grid=True)
        assert self.p1.col == 1

        # Right view.
        self.view1 = AtlasView(self.canvas, self.p1, self.tex, 1, *vargs)


        # GUI
        self.gui = self.canvas.gui("GUI")

        self.gui.control(
            'slider_float', 'ap', vmin=self.m.zlim[0], vmax=self.m.zlim[1])(self.slice_z)

        self.gui.control(
            'slider_float', 'ml', vmin=self.m.ylim[0], vmax=self.m.ylim[1])(self.slice_y)

    def _slice(self, axis, value):
        lim = self.m.zlim if axis == 0 else self.m.ylim
        v = self.view0 if axis == 0 else self.view1
        v.update_tex_coords((value - lim[0]) / (lim[1] - lim[0]))

    def slice_z(self, value):
        return self._slice(0, value)

    def slice_y(self, value):
        return self._slice(1, value)


# -------------------------------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    model = AtlasModel()
    c = AtlasController(model)

    run()
