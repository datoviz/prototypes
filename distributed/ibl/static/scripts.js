
/*************************************************************************************************/
/*  Constants                                                                                    */
/*************************************************************************************************/

const COUNT = 1000000; // initial number of spikes
const DEFAULT_EID = "0851db85-2889-4070-ac18-a40e8ebd96ba";
const RAW_DATA_URI = (eid, time) => "/raw/" + eid + "/" + time.toFixed(2);

window.params = {
    eid: DEFAULT_EID,

    color: null,
    colormap: 0,
    colormap_range: [0, 1],
    colormap_lims: [0, 1],

    alpha: null,
    alpha_range: [0.01, 0.1],
    alpha_lims: [0.01, 1],

    size: null,
    size_range: [0.01, 1],
    size_lims: [0.01, 10],

    time: 0,
};

window.zoom = 1;
window.shift = 0;

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
    const img = document.getElementById('imgRaster');
    img.src = url;
}

function scaleSize(x) {
    return 1 + Math.log(x);
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

// Data payload for the vertex buffer.
function vertexData() {
    var p = window.params;
    return {
        eid: p.eid,
        color: p.color,
        alpha: p.alpha,
        size: p.size,
    };
}




// Base64 data for the params buffer.
function paramsData() {
    var p = window.params;

    var arr = new StructArray(1, [
        ["alpha_range", "float32", 2],
        ["size_range", "float32", 2],
        ["cmap_range", "float32", 2],
        ["cmap_id", "uint32", 1],
    ]);

    arr.set(0, "alpha_range", p.alpha_range);
    arr.set(0, "size_range", [p.size_range[0], scaleSize(window.zoom) * p.size_range[1]]);
    arr.set(0, "cmap_range", p.colormap_range);
    arr.set(0, "cmap_id", [p.colormap]);

    return arr.b64();
}


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



function recordJSON() {
    return [
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
                "vertex_count":
                {
                    "mode": "ibl_ephys",
                    "session": vertexData()
                }
            }
        },
        {
            "action": "record",
            "type": "end",
            "id": 1
        }
    ];
}



// Load the initial JSON.
function loadJSON() {
    var requests = {
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
                    "size": 20 * COUNT
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
                        "session": vertexData()
                    }
                }
            },



        ]
    };
    requests["requests"].push(...recordJSON());
    return requests;
}



/*************************************************************************************************/
/*  Making WebSocket requests                                                                    */
/*************************************************************************************************/

// Send requests to the server.
function submit(contents) {
    if (!window.websocket) return;
    if (!window.websocket.connected) return;
    window.websocket.emit("request", contents);
}

submit = throttle(submit, 100);



// Update the session duration.
function updateDuration() {
    var duration = JS_CONTEXT["durations"][window.params.eid].toFixed(2);
    document.getElementById("sessionDuration").innerHTML = duration + " seconds";
}



// Update the vertex data.
function updateVertexData(eid) {
    window.params.eid = eid;

    // Update the raster plot
    var requests = {
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
                        "session": vertexData()
                    }
                }
            },
        ]
    };
    requests["requests"].push(...recordJSON());
    submit(requests);

    // Update the session duration.
    updateDuration();
}



// Update the params data.
function updateParamsData() {
    var contents = {
        "version": "1.0",
        "requests": [
            {
                "action": "upload",
                "type": "dat",
                "id": 12,
                "content": {
                    "offset": 0,
                    "data": {
                        "mode": "base64",
                        "buffer": paramsData()
                    }
                }
            },
        ]
    };
    submit(contents);
}



// Send the updated MVP struct to the server.
function updateMvpData() {
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
            {
                "action": "upload",
                "type": "dat",
                "id": 12,
                "content": {
                    "offset": 0,
                    "data": {
                        "mode": "base64",
                        "buffer": paramsData()
                    }
                }
            }
        ]
    };

    submit(req);
}



// Reset the view.
function reset() {
    window.zoom = 1;
    window.shift = 0;
    updateMvpData();
}



/*************************************************************************************************/
/*  Setup functions                                                                              */
/*************************************************************************************************/

function setupSliders() {

    // Alpha slider

    initSlider('sliderAlpha', window.params.alpha_range, window.params.alpha_lims);

    onSliderChange('sliderAlpha', function (min, max) {
        window.params.alpha_range = [min, max];
        updateParamsData();
    });



    // Size slider

    initSlider('sliderSize', window.params.size_range, window.params.size_lims);

    onSliderChange('sliderSize', function (min, max) {
        window.params.size_range = [min, max];
        updateParamsData();
    });



    // Colormap slider

    initSlider('sliderColormap', window.params.colormap_range, window.params.colormap_lims);

    onSliderChange('sliderColormap', function (min, max) {
        window.params.colormap_range = [min, max];
        updateParamsData();
    });

}



function setupDropdowns() {

    document.getElementById('selectSession').onchange = function (e) {
        updateVertexData(e.target.value);
    }

    document.getElementById('selectColormap').onchange = function (e) {
        window.params.colormap = parseInt(e.target.value);
        updateParamsData();
    }

    document.getElementById('selectColor').onchange = function (e) {
        window.params.color = e.target.value;
        updateVertexData(window.params.eid);
    }

    document.getElementById('selectAlpha').onchange = function (e) {
        window.params.alpha = e.target.value;
        updateVertexData(window.params.eid);
    }

    document.getElementById('selectSize').onchange = function (e) {
        window.params.size = e.target.value;
        updateVertexData(window.params.eid);
    }
}



function setupPanzoom() {
    const img = document.getElementById('imgRaster');

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
            updateMvpData();
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
/*  Raw ephys data viewer                                                                        */
/*************************************************************************************************/

function setRawImage(eid, time) {
    window.params.eid = eid;
    window.params.time = time;

    var url = RAW_DATA_URI(window.params.eid, window.params.time);
    const img = document.getElementById('imgRaw');
    img.src = url;
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

    setRawImage(window.params.eid, window.params.time);
    updateDuration();
}
