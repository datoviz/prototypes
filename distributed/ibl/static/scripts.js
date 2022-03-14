
/*************************************************************************************************/
/*  Constants                                                                                    */
/*************************************************************************************************/

const COUNT = 2000000; // max number of spikes
const DEFAULT_EID = "0851db85-2889-4070-ac18-a40e8ebd96ba";

window.zoom = 1;
window.shift = 0;

window.sizeMin = 0.01;
window.sizeMax = 1;

window.websocket = null;



/*************************************************************************************************/
/*  Utils                                                                                        */
/*************************************************************************************************/

function throttle(func, wait, options) {
    var context, args, result;
    var timeout = null;
    var previous = 0;
    if (!options) options = {};
    var later = function () {
        previous = options.leading === false ? 0 : Date.now();
        timeout = null;
        result = func.apply(context, args);
        if (!timeout) context = args = null;
    };
    return function () {
        var now = Date.now();
        if (!previous && options.leading === false) previous = now;
        var remaining = wait - (now - previous);
        context = this;
        args = arguments;
        if (remaining <= 0 || remaining > wait) {
            if (timeout) {
                clearTimeout(timeout);
                timeout = null;
            }
            previous = now;
            result = func.apply(context, args);
            if (!timeout) context = args = null;
        } else if (!timeout && options.trailing !== false) {
            timeout = setTimeout(later, remaining);
        }
        return result;
    };
};

// Display an array buffer.
function show(arrbuf) {
    const blob = new Blob([arrbuf]);
    const url = URL.createObjectURL(blob);
    const img = document.getElementById('img');
    img.src = url;
}

function scaleSize(x) {
    return Math.log(1 + x);
}



/*************************************************************************************************/
/*  Sliders                                                                                      */
/*************************************************************************************************/

function initSlider(id, initRange, fullRange) {
    noUiSlider.create(document.getElementById(id), {
        start: initRange,
        connect: true,
        range: {
            'min': fullRange[0],
            'max': fullRange[1]
        },
        tooltips: true,
    });
};

function onSliderChange(id, callback) {
    document.getElementById(id).noUiSlider.on('update',
        function (values, handle, unencoded, tap, positions, noUiSlider) {
            min = parseFloat(values[0]);
            max = parseFloat(values[1]);
            callback(min, max);
        });
}



/*************************************************************************************************/
/*  Datoviz JSON requests                                                                        */
/*************************************************************************************************/

// Data payload for a given session.
function sessionJSON(eid) {
    return {
        "eid": eid,
    };
}



// Load the initial JSON.
function loadJSON() {
    var count = COUNT;
    return {
        "version": "1.0",
        "requests": [
            {
                "action": "create",
                "type": "board",
                "id": 1,
                "content": {
                    "width": 800,
                    "height": 600,
                    "background": [255, 255, 255]
                }
            },
            {
                "action": "create",
                "type": "graphics",
                "id": 2,
                "flags": 0,
                "content": {
                    "board": 1,
                    "type": 7
                }
            },


            // MVP: 10
            {
                "action": "create",
                "type": "dat",
                "id": 10,
                "content": {
                    "type": 5,
                    "size": 200
                }
            },
            {
                "action": "bind",
                "type": "dat",
                "id": 2,
                "content": {
                    "dat": 10,
                    "slot_idx": 0,
                }
            },
            {
                "action": "upload",
                "type": "dat",
                "id": 10,
                "content": {
                    "offset": 0,
                    "size": 200,
                    "data": {
                        "mode": "base64",
                        "buffer": "AACAPwAAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAAAAAAAAAAAAACAPwAAgD8AAAAAAAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAAAAAAAAgD8AAIA/AAAAAAAAAAAAAAAAAAAAAAAAgD8AAAAAAAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAAAAAA=="
                    }
                }
            },


            // Viewport: 11
            {
                "action": "create",
                "type": "dat",
                "id": 11,
                "content": {
                    "type": 5,
                    "size": 80
                }
            },
            {
                "action": "bind",
                "type": "dat",
                "id": 2,
                "content": {
                    "dat": 11,
                    "slot_idx": 1,
                }
            },
            {
                "action": "upload",
                "type": "dat",
                "id": 11,
                "content": {
                    "offset": 0,
                    "size": 80,
                    "data": {
                        "mode": "base64",
                        "buffer": "AAAAAAAAAAAAAEhEAAAWRAAAAAAAAIA/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAwAAWAIAAAAAAAAAAAAAIAMAAFgCAAAAAAAAAAAAAAAAAAAAAAAA"
                    }
                }
            },


            // Params: 12
            {
                "action": "create",
                "type": "dat",
                "id": 12,
                "content": {
                    "type": 5,
                    "size": 20
                }
            },
            {
                "action": "bind",
                "type": "dat",
                "id": 2,
                "content": {
                    "dat": 12,
                    "slot_idx": 2,
                }
            },
            {
                "action": "upload",
                "type": "dat",
                "id": 12,
                "content": {
                    "offset": 0,
                    "size": 20,
                    "data": {
                        "mode": "base64",
                        "buffer": paramsData()
                    }
                }
            },


            // Vertex buffer: 3
            {
                "action": "create",
                "type": "dat",
                "id": 3,
                "content": {
                    "type": 2,
                    "size": 20 * count
                }
            },
            {
                "action": "set",
                "type": "vertex",
                "id": 2,
                "content": {
                    "dat": 3
                }
            },
            {
                "action": "upload",
                "type": "dat",
                "id": 3,
                "content": {
                    "offset": 0,
                    "data": {
                        "mode": "ibl_ephys",
                        "count": count,
                        "session": sessionJSON(DEFAULT_EID)
                    }
                }
            },



            {
                "action": "record",
                "type": "begin",
                "id": 1
            },
            {
                "action": "record",
                "type": "viewport",
                "id": 1,
                "content": {
                    "offset": [
                        0,
                        0
                    ],
                    "shape": [
                        0,
                        0
                    ]
                }
            },
            {
                "action": "record",
                "type": "draw",
                "id": 1,
                "content": {
                    "graphics": 2,
                    "first_vertex": 0,
                    "vertex_count": count
                }
            },
            {
                "action": "record",
                "type": "end",
                "id": 1
            }
        ]
    };
}



// Select another session.
function changeSession(eid) {
    var contents = {
        "version": "1.0",
        "requests": [
            {
                "action": "upload",
                "type": "dat",
                "id": 3,
                "content": {
                    "offset": 0,
                    "data": {
                        "mode": "ibl_ephys",
                        "count": COUNT,
                        "session": sessionJSON(eid)
                    }
                }
            },
        ]
    };
    submit(contents);
}



// Send requests to the server.
function submit(contents) {
    if (!window.websocket) return;
    if (!window.websocket.connected) return;
    window.websocket.emit("request", contents);
}

submit = throttle(submit, 100);



/*************************************************************************************************/
/*  Uniform buffers                                                                              */
/*************************************************************************************************/

// Return a MVP structure for a given pan and zoom.
function mvpData(px, py, zx, zy) {
    // 3 matrices 4x4: model, view, proj, and finally time
    var arr = new StructArray(1, [
        ["model", "float32", 16],
        ["view", "float32", 16],
        ["proj", "float32", 16],
        ["time", "float32", 1],
    ]);

    arr.set(0, "model", [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1]);
    arr.set(0, "view", [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        px, py, -2, 1]);
    arr.set(0, "proj", [
        zx, 0, 0, 0,
        0, zy, 0, 0,
        0, 0, -0.1, 0,
        0, 0, 0, 1]);
    arr.set(0, "time", [0]);

    return arr.b64();
}



// Return a MVP structure for a given pan and zoom.
function paramsData() {
    var arr = new StructArray(1, [
        ["alpha_range", "float32", 2],
        ["size_range", "float32", 2],
        ["cmap_range", "float32", 2],
        ["cmap_id", "uint32", 1],
    ]);

    arr.set(0, "alpha_range", [0, .1]);
    arr.set(0, "size_range", [.01, 1]);
    arr.set(0, "cmap_range", [0, 1]);
    arr.set(0, "cmap_id", [0]);

    return arr.b64();
}



function sizeRangeData() {
    var arr = new StructArray(1, [
        ["size_range", "float32", 2],
    ]);
    arr.set(0, "size_range", [window.sizeMin, window.sizeMax * scaleSize(window.zoom)]);

    return arr.b64();
}



function rangeData(min, max) {
    var arr = new StructArray(1, [
        ["var", "float32", 2],
    ]);
    arr.set(0, "var", [min, max]);
    return arr.b64();
}



function cmapData(cmap_id) {
    var arr = new StructArray(1, [
        ["cmap_id", "uint32", 1],
    ]);
    arr.set(0, "cmap_id", [cmap_id]);
    return arr.b64();
}



function featureData(name, feature) {
    const req = {
        "version": "1.0",
        "requests": [
            {
                "action": "upload",
                "type": "dat",
                "id": 3,
                "content": {
                    "offset": 0,
                    "data": {
                        "mode": "ibl_ephys",
                        "count": COUNT,
                        "session": sessionJSON(DEFAULT_EID),
                        "features": {
                            [name]: feature
                        }
                    }
                }
            },
        ]
    };
    return req;
}



/*************************************************************************************************/
/*  Uniform buffer updates                                                                       */
/*************************************************************************************************/

function paramsUpdateRequest(offset, buffer) {
    return {
        "version": "1.0",
        "requests": [
            {
                "action": "upload",
                "type": "dat",
                "id": 12,
                "content": {
                    "offset": offset,
                    "data": {
                        "mode": "base64",
                        "buffer": buffer
                    }
                }
            },
        ]
    };
}



// Send the updated MVP struct to the server.
function updateUniforms() {
    const req = {
        "version": "1.0",
        "requests": [
            // Change the MVP matrices.
            {
                "action": "upload",
                "type": "dat",
                "id": 10,
                "content": {
                    "offset": 0,
                    "data": {
                        "mode": "base64",
                        "buffer": mvpData(window.shift, 0, window.zoom, 1)
                    }
                }
            },

            // Change the size range data in the uniform params.
            paramsUpdateRequest(8, sizeRangeData())["requests"][0]
        ]
    };

    submit(req);
}



// Reset the view.
function reset() {
    window.zoom = 1;
    window.shift = 0;
    updateUniforms();
}



/*************************************************************************************************/
/*  Setup functions                                                                              */
/*************************************************************************************************/

function setupSliders() {

    initSlider('sliderAlpha', [0, .1], [0, 1]);

    onSliderChange('sliderAlpha', function (min, max) {
        const req = paramsUpdateRequest(0, rangeData(min, max));
        submit(req);
    });



    initSlider('sliderSize', [0.01, 1], [0, 5]);

    onSliderChange('sliderSize', function (min, max) {
        window.sizeMin = min;
        window.sizeMax = max;
        const req = paramsUpdateRequest(8, sizeRangeData());
        submit(req);
    });



    initSlider('sliderColormap', [0, 1], [0, 1]);

    onSliderChange('sliderColormap', function (min, max) {
        const req = paramsUpdateRequest(16, rangeData(min, max));
        submit(req);
    });
}



function setupDropdowns() {

    document.getElementById('selectSession').onchange = function (e) {
        changeSession(e.target.value);
    }

    document.getElementById('selectColormap').onchange = function (e) {
        var cmap_id = parseInt(e.target.value);
        const req = paramsUpdateRequest(24, cmapData(cmap_id));
        submit(req);
    }

    document.getElementById('selectColor').onchange = function (e) {
        var feature = e.target.value;
        const req = featureData("color", feature);
        submit(req);
    }

    document.getElementById('selectAlpha').onchange = function (e) {
        var feature = e.target.value;
        const req = featureData("alpha", feature);
        submit(req);
    }

    document.getElementById('selectSize').onchange = function (e) {
        var feature = e.target.value;
        const req = featureData("size", feature);
        submit(req);
    }
}



function setupPanzoom() {
    const img = document.getElementById('img');

    img.onwheel = function (e) {
        e.preventDefault();
        let d = -e.deltaY / Math.abs(e.deltaY);
        let z = window.zoom;

        var rect = e.target.getBoundingClientRect();
        var x = e.clientX - rect.left; //x position within the element.
        var y = e.clientY - rect.top;  //y position within the element.

        let w = e.target.width;

        window.zoom *= (1 + .5 * d);
        window.zoom = Math.max(1, window.zoom);

        let center = -1 + 2 * x / w;
        window.shift -= center * (1 / z - 1 / window.zoom);

        if (window.zoom != z)
            updateUniforms();
    }

    img.ondblclick = function (e) {
        reset();
    }
}



function setupWebsocket() {
    window.websocket = io();

    window.websocket.on("connect", () => {
        console.log('socket connected');

        var contents = loadJSON();
        submit(contents);
    });

    window.websocket.on("image", (data) => {
        // display the received image
        let img = data["image"];
        if (img instanceof ArrayBuffer) {
            show(img);
        }
    });

    window.websocket.on("disconnect", () => {
        console.log('socket disconnected');
    });
}



/*************************************************************************************************/
/*  Entry point                                                                                  */
/*************************************************************************************************/

// On load, create the websocket and set the onnopen, onmessage, and onclose callbacks.
function load() {

    setupSliders();
    setupDropdowns();
    setupPanzoom();

    setupWebsocket();
}
