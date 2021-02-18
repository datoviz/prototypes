import numpy as np

from datoviz import canvas, run, colormap


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


cov = np.ascontiguousarray(np.load('coverage.npy'))
atlas = np.ascontiguousarray(np.load('atlas.npy'))

cov = normalize_volume(cov)
atlas = normalize_volume(atlas)

assert cov.shape == atlas.shape
x, y, z = cov.shape

atlas += cov
atlas = np.transpose(atlas, (1, 2, 0))


c = canvas(rows=1, cols=2, show_fps=True, width=1600, height=600)

p0 = c.panel(col=0, controller='panzoom', hide_grid=True)
v0 = p0.visual('volume_slice')

# Top left, top right, bottom right, bottom left
v0.data('pos', np.array([0, y, 0]), idx=0)
v0.data('pos', np.array([x, y, 0]), idx=1)
v0.data('pos', np.array([x, 0, 0]), idx=2)
v0.data('pos', np.array([0, 0, 0]), idx=3)

v0.volume(atlas)

def update_tex_coords(w):
    # Top left, top right, bottom right, bottom left
    v0.data('texcoords', np.array([0, 0, w]), idx=0)
    v0.data('texcoords', np.array([0, 1, w]), idx=1)
    v0.data('texcoords', np.array([1, 1, w]), idx=2)
    v0.data('texcoords', np.array([1, 0, w]), idx=3)

update_tex_coords(.5)

gui = c.gui("GUI")

@gui.control('slider_float', 'z')
def change_depth(w):
    update_tex_coords(w)

run()
