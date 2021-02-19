"""
Seeing coverage volume.
"""

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import numpy as np

from datoviz import canvas, run, colormap


# -------------------------------------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------------------------------------

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


# -------------------------------------------------------------------------------------------------
# Atlas model
# -------------------------------------------------------------------------------------------------

class AtlasModel:
    def __init__(self):
        cov = np.load('coverage.npy')
        atlas = np.load('atlas.npy')

        cov = normalize_volume(cov)
        atlas = normalize_volume(atlas)

        assert cov.shape == atlas.shape
        self.shape = cov.shape

        atlas += cov
        atlas = np.clip(atlas, 0, 1)
        atlas = np.ascontiguousarray(atlas)
        atlas = np.transpose(atlas, (1, 2, 0))

        self.atlas = atlas


# -------------------------------------------------------------------------------------------------
# Atlas view
# -------------------------------------------------------------------------------------------------

class AtlasView:
    tex = None

    def __init__(self, canvas, panel, tex, axis):
        self.canvas = canvas
        self.panel = panel
        self.axis = axis
        self.tex = tex

        self.visual = panel.visual('volume_slice')
        self.visual.texture(self.tex)

        # TODO: coordinates with brain atlas
        x, y, z = 1, 1, 1

        # Top left, top right, bottom right, bottom left
        self.visual.data('pos', np.array([0, y, 0]), idx=0)
        self.visual.data('pos', np.array([x, y, 0]), idx=1)
        self.visual.data('pos', np.array([x, 0, 0]), idx=2)
        self.visual.data('pos', np.array([0, 0, 0]), idx=3)

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
        self.tex = self.canvas.volume(self.m.atlas)


        # Left panel.
        self.p0 = self.canvas.panel(col=0, controller='axes', hide_grid=True)
        assert self.p0.col == 0

        # Left view.
        self.view0 = AtlasView(self.canvas, self.p0, self.tex, 0)


        # Right panel.
        self.p1 = self.canvas.panel(col=1, controller='axes', hide_grid=True)
        assert self.p1.col == 1

        # Right view.
        self.view1 = AtlasView(self.canvas, self.p1, self.tex, 1)


        # GUI
        self.gui = self.canvas.gui("GUI")
        self.gui.control('slider_float', 'z')(self.view0.update_tex_coords)
        self.gui.control('slider_float', 'y')(self.view1.update_tex_coords)



# -------------------------------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    model = AtlasModel()
    c = AtlasController(model)

    run()
