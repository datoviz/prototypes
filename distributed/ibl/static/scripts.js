
/*************************************************************************************************/
/*  Constants                                                                                    */
/*************************************************************************************************/

const COUNT = 1000000; // initial number of spikes
const DEFAULT_COLORMAP = 239;
const DEFAULT_EID = "0851db85-2889-4070-ac18-a40e8ebd96ba";
const RAW_DATA_URI = (eid, time) => "/raw/" + eid + "/" + time.toFixed(2);

window.params = {
    eid: DEFAULT_EID,

    color: null,
    colormap: DEFAULT_COLORMAP,
    colormap_range: [0, 1],
    colormap_lims: [0, 1],

    alpha: null,
    alpha_range: [0.1, 0.3],
    alpha_lims: [0.01, 1],

    size: null,
    size_range: [0.1, 1.5],
    size_lims: [0.01, 10],

    time: 0,
    duration: 0,

    zoom: 1,
    shift: 0,
};

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

function clamp(x, min, max) {
    return Math.min(Math.max(x, min), max);
};



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
    arr.set(0, "size_range", [p.size_range[0], scaleSize(p.zoom) * p.size_range[1]]);
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
    var p = window.params;
    var duration = JS_CONTEXT["durations"][p.eid];
    p.duration = duration;
    document.getElementById("sessionDuration").innerHTML = p.time.toFixed(3) + " / " + duration.toFixed(1) + " s";
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
                        "buffer": mvpData(window.params.shift, 0, window.params.zoom, 1)
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
    window.params.zoom = 1;
    window.params.shift = 0;
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
    document.getElementById('selectColormap').value = DEFAULT_COLORMAP;

    document.getElementById('selectColor').onchange = function (e) {
        window.params.color = e.target.value;
        updateVertexData(window.params.eid);
    }
    document.getElementById('selectColor').value = "cluster";

    document.getElementById('selectAlpha').onchange = function (e) {
        window.params.alpha = e.target.value;
        updateVertexData(window.params.eid);
    }

    document.getElementById('selectSize').onchange = function (e) {
        window.params.size = e.target.value;
        updateVertexData(window.params.eid);
    }
}



function setupRaster() {
    const img = document.getElementById('imgRaster');
    const line = document.getElementById('rasterLine');

    var x0 = $(img).offset().left;
    var w = $(img).width();

    line.style.height = (img.offsetHeight - 2) + "px";


    // Zooming.
    img.onwheel = function (e) {
        e.preventDefault();
        let d = -e.deltaY / Math.abs(e.deltaY);
        let z = window.params.zoom;

        var rect = img.getBoundingClientRect();
        var x = e.clientX - rect.left; // x position within the element.
        var y = e.clientY - rect.top;  // y position within the element.

        window.params.zoom *= (1 + .5 * d);
        window.params.zoom = Math.max(1, window.params.zoom);

        let center = -1 + 2 * x / w;
        window.params.shift -= center * (1 / z - 1 / window.params.zoom);

        if (window.params.zoom != z) {
            updateMvpData();
        }
    }


    // Update the vertical line position upon pan/zoom.
    img.onload = function (e) {
        setLineOffset();
    }
    window.onresize = function (e) {
        line.style.height = (img.offsetHeight - 2) + "px";
        setLineOffset();
    }


    // Prevent image dragging.
    img.ondragstart = function (e) {
        e.preventDefault();
    }


    // Panning.
    var isPanning = false;
    img.onmousedown = function (e) {
        isPanning = true;
    }
    img.onmousemove = function (e) {
        if (isPanning) {
            var x = 2 * e.movementX / (w * window.params.zoom);
            window.params.shift += x;

            updateMvpData();
        }
    }
    document.onmouseup = function (e) {
        isPanning = false;
    }


    // Reset pan/zoom when double clicking.
    img.ondblclick = function (e) {
        reset();
    }


    // Draggable vertical line.
    $(line).draggable({
        axis: "x",
        containment: "#imgRaster",
        stop: function (e, ui) {
            // When dragging stops, selects the time and update the raw image.
            var offset = $(this).offset();
            var x = offset.left - x0;

            // Update the time.
            window.params.time = px2time(x)
            console.log("select time: " + window.params.time.toFixed(3) + " s");

            // Update the raw image.
            setRawImage();

            // Update the time info.
            updateDuration();
        }
    });
}



function setupRaw() {
    const img = document.getElementById('imgRaw');

    img.ondragstart = function (e) {
        e.preventDefault();
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
/*  Raster viewer                                                                                */
/*************************************************************************************************/

function px2time(px) {
    const img = document.getElementById('imgRaster');
    let w = img.width;

    x = px / w;
    x = clamp(x, 0, 1);

    // Take pan and zoom into account.
    x = .5 * (1 + (-1 + 2 * x) / window.params.zoom - window.params.shift);

    // Scale by the duration to get the time.
    var p = window.params;
    var duration = p.duration;

    return x * duration;
}



function time2px(t) {
    const img = document.getElementById('imgRaster');
    let w = img.width;

    // Scale by the duration to get the time.
    var p = window.params;
    var duration = p.duration;

    var x = t / duration;
    x = .5 * (1 + (x * 2 - 1 + window.params.shift) * window.params.zoom)
    var px = x * w;
    px = clamp(px, 0, w);

    return px;
}



function setLineOffset() {
    const img = document.getElementById('imgRaster');
    var x0 = $(img).offset().left;
    var w = $(img).width();

    const line = document.getElementById('rasterLine');
    $(line).offset({ left: x0 + time2px(window.params.time) });
}



/*************************************************************************************************/
/*  Raw ephys data viewer                                                                        */
/*************************************************************************************************/

function setRawImage() {
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
    setupRaster();
    setupRaw();
    setupWebsocket();

    setRawImage();
    updateDuration();
}



$(document).ready(function () {
    load();
});
