import numpy as np
from datoviz import canvas, run, colormap

# Data loading.
x = np.load('x_position.npy')
y = np.load('y_position.npy')
z = np.load('z_position.npy')
pos = np.c_[x, z, y]

st = np.load('spike_times.npy') / 3e4
amp = np.load('amplitudes.npy')
alpha = np.load('alphas.npy')

# Color.
log_ptp = amp
log_ptp[log_ptp >= 30] = 30
ptp_rescaled = (log_ptp - log_ptp.min())/(log_ptp.max() - log_ptp.min())
color = colormap(ptp_rescaled, alpha=5. / 255, cmap='spring')

# Create the visual.
c = canvas(width=800, height=1000, show_fps=True)
v = c.scene().panel(
    controller='arcball').visual('point', depth_test=True)

# Visual prop data.
v.data('pos', pos)
v.data('color', color)
v.data('ms', np.array([2]))

# GUI with slider.
t = 0
dt = 20.0
gui = c.gui("GUI")
slider_offset = gui.control(
    "slider_float", "time offset", vmin=0, vmax=st[-1] - dt, value=0)
slider_dt = gui.control(
    "slider_float", "time interval", vmin=0.1, vmax=100, value=dt)
slider_ms = gui.control(
    "slider_float", "marker size", vmin=0.01, vmax=20, value=2)


@slider_offset.connect
def change_offset(value):
    assert value >= 0
    global t
    t = value
    i, j = np.searchsorted(st, [t, t + dt])
    color[:, 3] = 10
    color[i:j, 3] = 200
    v.data('color', color)


@slider_dt.connect
def on_dt_changed(value):
    assert value >= 0
    global t, dt
    dt = value
    change_offset(t)


@slider_ms.connect
def on_ms_changed(value):
    v.data('ms', np.array([value]))


change_offset(0)
run()


# def on_timer():
#     global t
#     i, j = np.searchsorted(st, [t, t + 5])
#     color[:, 3] = 5
#     color[i:j, 3] = 200
#     v.data('color', color)
#     t += 5
#     t = t % 1000

# c._connect('timer', on_timer, 1. / 10)
