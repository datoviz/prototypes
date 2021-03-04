from pathlib import Path

import numpy as np
import numpy.random as nr

from datoviz import canvas, run, colormap


c = canvas(show_fps=True, pick=True)
panel = c.panel(controller='arcball')
visual = panel.visual('volume')

texshape = np.array([456, 320, 528])
coef = .004
shape = coef * texshape[[1, 0, 2]]

visual.data('pos', np.atleast_2d(-shape / 2), idx=0)
visual.data('pos', np.atleast_2d(+shape / 2), idx=1)

# Box shape.
visual.data('length', np.atleast_2d(shape))

# Transfer function.
fun = np.linspace(0, .05, 256).astype(np.float32)
transfer = c.transfer(fun)
visual.texture(transfer)

# X range of the transfer function
visual.data('transferx', np.array([0, .5]))

# 3D texture with the volume
ROOT = Path(__file__).resolve().parent.parent / 'datoviz'
vol = np.fromfile(ROOT / "data/volume/atlas_25.img", dtype=np.uint16)
vol = vol.reshape(texshape)
vol *= 100
V = c.volume(vol)

@c.connect
def on_mouse_click(x, y, button, modifiers=()):
    u, v, w, _ = c.pick(x, y)
    u /= 255.0
    v /= 255.0
    w /= 255.0
    print(f"Texture coordinates are {u=:.4f}, {v=:.4f}, {w=:.4f}")

# GUI
gui = c.gui("GUI")

# Transfer function slider
@gui.control("slider_float2", "transferx", vmin=0, vmax=1, force_increasing=True)
def on_change(x0, x1):
    visual.data('transferx', np.array([x0, x1]))

# Clipping
clip = np.zeros(4, dtype=np.float32)
clip[2] = +1

@gui.control("slider_float", "clip offset", vmin=0, vmax=1, value=0)
def on_change(x):
    clip[3] = -x
    visual.data('clip', clip)

# Set the texture to the visual.
visual.texture(V)

run()
