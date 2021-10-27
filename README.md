# Datoviz experiments and prototypes

This repository contains various ongoing experiments and GUI prototypes.

All prototypes are experimental. Use at your own risks!

## Installation instructions

* [Install Datoviz](https://datoviz.org/tutorials/install/#how-to-install-datoviz)
* Make sure the Datoviz demo is working:

    ```bash
    python -c "import datoviz; datoviz.demo()"
    ```
* Install the `develop` branch of [ibllib](https://github.com/int-brain-lab/ibllib)
* Clone the prototypes repository:

    ```bash
    git clone https://github.com/datoviz/prototypes.git dvzproto
    ```

## IBL prototypes

### Ephys data viewer

* Usage example (passing the session eid and probe idx):

    ```bash
    python ephysview.py f25642c6-27a5-4a97-9ea0-06652db79fbd 0
    ```

* Description: shows the full raster plot of an ephys session at the top, and a small section of the raw data on the bottom.
* Use: control+click in the top panel to select a particular time, and load (download + on-the-fly decompression) the corresponding chunk of raw data in the bottom panel.

![](images/ephysview.jpg)


### Spike localization viewer

* Requires the following files (ask Julien Boussard)
    * `spikeloc/x_position.npy`
    * `spikeloc/y_position.npy`
    * `spikeloc/z_position.npy`
    * `spikeloc/spike_times.npy`
    * `spikeloc/amplitudes.npy`
    * `spikeloc/alphas.npy`
* Usage example: `python spikeloc.py`

![](images/spikeloc.jpg)


### Coverage viewer

* Usage example: `python coverage.py`
* Requires: `coverage.npy`

![](images/coverage.jpg)


## Other prototypes

* **High-res humain brain viewer**: `brain_highres.py`
* **Molecule viewer**: `molecule.py`
