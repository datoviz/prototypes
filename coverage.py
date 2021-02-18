import numpy as np

from ibllib.atlas import AllenAtlas

from datoviz import canvas, run, colormap


def save_volume():
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



vol = np.load('coverage.npy')
vol = np.transpose(vol, (2, 1, 0))
x, y, z = vol.shape

ba = AllenAtlas()
vola = np.ascontiguousarray(ba.image)
vola = np.transpose(vola, (1, 2, 0))
# assert vola.shape == vol.shape

c = canvas(show_fps=True)
panel = c.panel(controller='panzoom', hide_grid=True)
visual = panel.visual('volume_slice')

# Top left, top right, bottom right, bottom left
visual.data('pos', np.array([0, y, 0]), idx=0)
visual.data('pos', np.array([x, y, 0]), idx=1)
visual.data('pos', np.array([x, 0, 0]), idx=2)
visual.data('pos', np.array([0, 0, 0]), idx=3)

visual.data('scale', np.array([150]))

def update_tex_coords(w):
    # Top left, top right, bottom right, bottom left
    visual.data('texcoords', np.array([0, 0, w]), idx=0)
    visual.data('texcoords', np.array([0, 1, w]), idx=1)
    visual.data('texcoords', np.array([1, 1, w]), idx=2)
    visual.data('texcoords', np.array([1, 0, w]), idx=3)

# visual.data('colormap', np.array([117]))

update_tex_coords(.5)

visual.volume(vola)

gui = c.gui("GUI")

# w = .5
@gui.control('slider_float', 'z')
def change_depth(w):
    update_tex_coords(w)
    # global w
    # if modifiers == ('control',):
    #     w += .001 * dy

run()
