from pathlib import Path

import numpy as np
import numpy.random as nr

from datoviz import canvas, run, colormap


c = canvas(show_fps=True)
panel = c.panel(controller='arcball')
visual = panel.visual('volume')

texshape = np.array([456, 320, 528])
coef = .004
shape = coef * texshape[[1, 0, 2]]

visual.data('pos', np.atleast_2d(-shape / 2), idx=0)
visual.data('pos', np.atleast_2d(+shape / 2), idx=1)

visual.data('length', np.atleast_2d(shape))


# Transfer function.
fun = np.linspace(0, 1, 256).astype(np.float32)
transfer = c.transfer(fun)
visual.texture(transfer)

# 3D texture with the volume
ROOT = Path(__file__).resolve().parent.parent / 'datoviz'
vol = np.fromfile(ROOT / "data/volume/atlas_25.img", dtype=np.uint16)
vol = vol.reshape(texshape)
vol *= 10
V = c.volume(vol)

gui = c.gui("GUI")
@gui.control("slider_float2", "transferx", vmin=0, vmax=1, force_increasing=True)
def on_change(x0, x1):
    visual.data('transferx', np.array([x0, x1]))

# Set the texture to the visual.
visual.texture(V)

run()
