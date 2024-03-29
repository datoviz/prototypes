
/*************************************************************************************************/
/*  Constants                                                                                    */
/*************************************************************************************************/

const COUNT = 1000000; // initial number of spikes
const DEFAULT_COLORMAP = 239;
const DEFAULT_EID = "0851db85-2889-4070-ac18-a40e8ebd96ba";
const WIDTH = 1024;
const HEIGHT = 512;
const MARKER_SIZE = 8;
const HALF_WINDOW = 0.1;
const RAW_DATA_URI = (eid, time) => "/" + eid + "/raw/" + time.toFixed(2);
const SPIKES_DATA_URI = (eid, time) => "/" + eid + "/spikes/" + time.toFixed(2);
const CLUSTER_DATA_URI = (eid, cluster) => "/" + eid + "/cluster/" + cluster;

const DEFAULT_PARAMS = {
    eid: DEFAULT_EID,

    color: "cluster",
    colormap: DEFAULT_COLORMAP,
    colormap_range: [0, 1],
    colormap_lims: [0, 1],

    alpha: "",
    alpha_range: [0.1, 0.3],
    alpha_lims: [0.01, 1],

    size: "",
    size_range: [0.1, 1.5],
    size_lims: [0.01, 10],

    time: 0,
    duration: 1,
    spike_count: 0,
    cluster: 0,

    zoom: 1,
    shift: 0,
};
window.websocket = null;



/*************************************************************************************************/
/*  Utils                                                                                        */
/*************************************************************************************************/

function isEmpty(obj) {
    // https://stackoverflow.com/a/679937/1595060
    return Object.keys(obj).length === 0;
};



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
    let w = img.offsetWidth;

    // Update the raster plot
    // const img = document.getElementById('imgRaster');
    // img.src = url;

    var t0 = px2time(0);
    var t1 = px2time(w);
    var t = .5 * (t0 + t1);

    Plotly.update('imgRaster', {}, {
        "images[0].source": url,
        "xaxis.ticktext": [t0.toFixed(3), t.toFixed(3), t1.toFixed(3)],
    });

    setLineOffset();
};



function scaleSize(x) {
    return 1 + Math.log(x);
};



function scaleAlpha(x) {
    return 1 + .1 * Math.log(x);
};


function clamp(x, min, max) {
    return Math.min(Math.max(x, min), max);
};



// Return the element of the array that is the closest from x.
function closest(arr, x) {
    return arr.reduce(function (prev, curr) {
        return (Math.abs(curr - x) < Math.abs(prev - x) ? curr : prev);
    });
};



/*************************************************************************************************/
/*  Sliders                                                                                      */
/*************************************************************************************************/

function initSlider(id, initRange, fullRange) {

    var el = document.getElementById(id);
    if (el.noUiSlider)
        el.noUiSlider.destroy();

    noUiSlider.create(el, {
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
    var el = document.getElementById(id);
    el.noUiSlider.on('update',
        function (values, handle, unencoded, tap, positions, noUiSlider) {
            min = parseFloat(values[0]);
            max = parseFloat(values[1]);
            callback(min, max);
        });
};



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
};



// Base64 data for the params buffer.
function paramsData() {
    var p = window.params;

    var arr = new StructArray(1, [
        ["alpha_range", "float32", 2],
        ["size_range", "float32", 2],
        ["cmap_range", "float32", 2],
        ["cmap_id", "uint32", 1],
    ]);

    arr.set(0, "alpha_range", [p.alpha_range[0], scaleSize(p.zoom) * p.alpha_range[1]]);
    arr.set(0, "size_range", [p.size_range[0], scaleAlpha(p.zoom) * p.size_range[1]]);
    arr.set(0, "cmap_range", p.colormap_range);
    arr.set(0, "cmap_id", [p.colormap]);

    return arr.b64();
};



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
};



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
};



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
                    "width": WIDTH,
                    "height": HEIGHT,
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
};



/*************************************************************************************************/
/*  Making WebSocket requests                                                                    */
/*************************************************************************************************/

// Send requests to the server.
function submit(contents) {
    if (!window.websocket) return;
    if (!window.websocket.connected) return;
    window.websocket.emit("request", contents);
};

submit = throttle(submit, 100);



// Update the session duration.
function updateDuration() {
    var p = window.params;
    var duration = JS_CONTEXT["sessions"][p.eid]["duration"];
    p.duration = duration;
    document.getElementById("sessionDuration").innerHTML = p.time.toFixed(3) + " / " + duration.toFixed(1) + " s";

    var spike_count = JS_CONTEXT["sessions"][p.eid]["spike_count"];
    p.spike_count = spike_count;
    document.getElementById("spikeCount").innerHTML = spike_count.toLocaleString();
};



// Update the vertex data.
function updateVertexData(eid, extra_requests) {
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
    if (extra_requests) {
        requests["requests"].push(...extra_requests);
    }
    requests["requests"].push(...recordJSON());
    document.documentElement.className = 'wait';
    submit(requests);

    // Update the session duration.
    updateDuration();
};



function paramsDataRequest() {
    return {
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
    };
}


// Update the params data.
function updateParamsData() {
    var contents = {
        "version": "1.0",
        "requests": [
            paramsDataRequest(),
        ]
    };
    submit(contents);
};



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
};



// Reset the view.
function reset() {
    window.params.zoom = 1;
    window.params.shift = 0;
    updateMvpData();
};



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

};



function setupDropdowns() {

    document.getElementById('selectSession').onchange = function (e) {
        updateVertexData(e.target.value);
        setRawImage();
    }

    document.getElementById('selectColormap').onchange = function (e) {
        window.params.colormap = parseInt(e.target.value);
        updateParamsData();
    }

    document.getElementById('selectColor').onchange = function (e) {
        window.params.color = e.target.value;

        var extra_requests = [];

        // HACK: if selecting quality, automatically select the qualmap colormap
        if (window.params.color == "quality") {
            document.getElementById('selectColormap').value = 124;
            window.params.colormap = 124;
            extra_requests = [paramsDataRequest()];
        }

        updateVertexData(window.params.eid, extra_requests);
    }

    document.getElementById('selectAlpha').onchange = function (e) {
        window.params.alpha = e.target.value;
        updateVertexData(window.params.eid);
    }

    document.getElementById('selectSize').onchange = function (e) {
        window.params.size = e.target.value;
        updateVertexData(window.params.eid);
    }


    // Initial values.
    document.getElementById('selectSize').value = window.params.size;
    document.getElementById('selectSession').value = window.params.eid;
    document.getElementById('selectColormap').value = window.params.colormap;
    document.getElementById('selectColor').value = window.params.color;
    document.getElementById('selectAlpha').value = window.params.alpha;
};



function setupInputs() {

    document.getElementById('clusterInput').onchange = function (e) {
        var cl = parseInt(e.target.value);
        var session_info = JS_CONTEXT["sessions"][window.params.eid];
        cl = closest(session_info["cluster_ids"], cl);
        e.target.value = cl;
        window.params.cluster = cl;
        setClusterImage();
    };

};



function setupButtons() {
    document.getElementById('resetButton').onclick = function (e) {
        console.log("reset params");
        resetParams();
    }
};



function setupRaster() {
    const img = document.getElementById('imgRaster');

    Plotly.newPlot('imgRaster', [],
        {
            images: [
                {
                    // "source": url,
                    "xref": "x",
                    "yref": "y",
                    "x": 0,
                    "y": 0,
                    "sizex": window.params.duration,
                    "sizey": 385,
                    "opacity": 1,
                    "xanchor": "left",
                    "yanchor": "bottom",
                    "sizing": "stretch",
                    "layer": "below",
                },
            ],
            xaxis: {
                range: [0, window.params.duration],
                tickmode: "array",
                tickvals: [0, window.params.duration * .5, window.params.duration],
                ticktext: ['0', (window.params.duration * .5).toFixed(3), window.params.duration.toFixed(3)],
                showgrid: true,
            },
            yaxis: {
                range: [0, 385],
                showgrid: false,
            },
            margin: {
                b: 30, t: 10, l: 30, r: 30, pad: 0
            },
            autosize: true,
        },
        {
            scrollZoom: false,
            staticPlot: true
        });

    const line = document.getElementById('rasterLine');

    // Waiting Pointer.
    document.documentElement.className = 'wait';

    // Raster image width.
    var w = $(img).width() - 60;

    line.style.top = 112 + "px";
    line.style.height = (img.offsetHeight - 42) + "px";

    // Zooming.
    img.onwheel = function (e) {
        e.preventDefault();
        if (e.deltaY == 0)
            return;
        let d = -e.deltaY / Math.abs(e.deltaY);

        // macOS hack
        var isTouchPad = e.wheelDeltaY ? e.wheelDeltaY === -3 * e.deltaY : e.deltaMode === 0
        if (isTouchPad)
            d = d * .05;

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
            var x0 = $(img).offset().left + 30;

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
};



function setupRaw() {
    var url = RAW_DATA_URI(window.params.eid, window.params.time);

    const spikes = {
        mode: 'markers',
        type: 'scatter',
        hovertemplate: "cluster %{text}<extra></extra>"
    };

    var iconLeft = {
        'width': 1792,
        'height': 1792,
        'path': 'M1664 896v128q0 53-32.5 90.5t-84.5 37.5h-704l293 294q38 36 38 90t-38 90l-75 76q-37 37-90 37-52 0-91-37l-651-652q-37-37-37-90 0-52 37-91l651-650q38-38 91-38 52 0 90 38l75 74q38 38 38 91t-38 91l-293 293h704q52 0 84.5 37.5t32.5 90.5z',
        'transform': "matrix(1 0 0 -1 0 1850)"
    }
    var iconRight = {
        'width': 1792,
        'height': 1792,
        'path': 'M1600 960q0 54-37 91l-651 651q-39 37-91 37-51 0-90-37l-75-75q-38-38-38-91t38-91l293-293h-704q-52 0-84.5-37.5t-32.5-90.5v-128q0-53 32.5-90.5t84.5-37.5h704l-293-294q-38-36-38-90t38-90l75-75q38-38 90-38 53 0 91 38l651 651q37 35 37 90z',
        'transform': "matrix(1 0 0 -1 0 1850)"
    }

    Plotly.newPlot('imgRaw', [spikes],
        {
            images: [
                {
                    "source": url,
                    "xref": "x",
                    "yref": "y",
                    "x": 0,
                    "y": 0,
                    "sizex": 2 * HALF_WINDOW,
                    "sizey": 385,
                    "opacity": 1,
                    "xanchor": "left",
                    "yanchor": "bottom",
                    "sizing": "stretch",
                    "layer": "below",
                },
            ],
            xaxis: {
                range: [0, 2 * HALF_WINDOW],
                tickmode: "array",
                tickvals: [0, 0.1, 0.2], // HACK: hard-coded half window
                ticktext: ['0.000', '0.100', '0.200'],
                showgrid: false,
            },
            yaxis: {
                range: [0, 385],
                showgrid: false,
            },
            margin: {
                b: 30, t: 10, l: 30, r: 30, pad: 0
            },
            autosize: true,
        },
        {
            scrollZoom: true,
            modeBarButtonsToAdd: [
                {
                    name: 'goLeft',
                    icon: iconLeft,
                    direction: 'up',
                    click: function (gd) { shiftRaw(-.05); }
                },
                {
                    name: 'goRight',
                    icon: iconRight,
                    direction: 'up',
                    click: function (gd) { shiftRaw(+.05); }
                }
            ],
            modeBarButtonsToRemove: ["select2d", "lasso2d"]
        });
    var myPlot = document.getElementById("imgRaw");

    myPlot.onwheel = function (e) {
        return e.preventDefault();
    };

    myPlot.on('plotly_click', function (data) {
        var cluster = 0;
        for (var i = 0; i < data.points.length; i++) {
            var cluster = data.points[i].data.text[data.points[i].pointNumber];
            window.params.cluster = cluster;
            document.getElementById("clusterInput").value = cluster;
            setClusterImage();
        };
    });
};



function setupCluster() {
    const img = document.getElementById('imgCluster');
    img.value = window.params.cluster;

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
            document.documentElement.className = '';
            show(img);
        }
    });

    window.websocket.on("disconnect", () => {
        console.log('socket disconnected');
    });
};



/*************************************************************************************************/
/*  Raster viewer                                                                                */
/*************************************************************************************************/

function px2time(px) {
    const img = document.getElementById('imgRaster');
    var w = $(img).width() - 60;

    x = px / w;
    x = clamp(x, 0, 1);

    // Take pan and zoom into account.
    x = .5 * (1 + (-1 + 2 * x) / window.params.zoom - window.params.shift);

    // Scale by the duration to get the time.
    var p = window.params;
    var duration = p.duration;

    return x * duration;
};



function time2px(t) {
    const img = document.getElementById('imgRaster');
    var w = $(img).width() - 60;

    // Scale by the duration to get the time.
    var p = window.params;
    var duration = p.duration;

    var x = t / duration;
    x = .5 * (1 + (x * 2 - 1 + window.params.shift) * window.params.zoom)
    var px = x * w;
    px = clamp(px, 0, w);

    return px;
};



function setLineOffset() {
    console.log("hello");
    const img = document.getElementById('imgRaster');
    var x0 = $(img).offset().left + 30;
    var w = $(img).width() - 60;

    const line = document.getElementById('rasterLine');
    $(line).offset({ left: x0 + time2px(window.params.time) });
};



/*************************************************************************************************/
/*  Raw ephys data viewer                                                                        */
/*************************************************************************************************/

function shiftRaw(dt) {
    window.params.time += dt;
    setRawImage();
    setLineOffset();
}



function setRawImage() {
    var url = RAW_DATA_URI(window.params.eid, window.params.time);
    var t = window.params.time;
    var h = HALF_WINDOW;
    var t0 = t - h;
    var t1 = t + h;

    $.ajax({
        url: SPIKES_DATA_URI(window.params.eid, window.params.time),
    }).done(function (data) {
        Plotly.update('imgRaw', {
            x: [data.x],
            y: [data.y],
            marker: [{ size: MARKER_SIZE, symbol: 'x', color: data.spike_clusters }],
            text: [data.spike_clusters],
        }, {
            "images[0].source": url,
            "xaxis.ticktext": [t0.toFixed(3), t.toFixed(3), t1.toFixed(3)],
        });
    });

};



/*************************************************************************************************/
/*  Cluster plot                                                                                 */
/*************************************************************************************************/

function setClusterImage() {
    var url = CLUSTER_DATA_URI(window.params.eid, window.params.cluster);
    const img = document.getElementById('imgCluster');
    img.src = url;
};



/*************************************************************************************************/
/*  Params browser persistence                                                                   */
/*************************************************************************************************/

function loadParams() {
    window.params = JSON.parse(localStorage.params || "{}");
    if (isEmpty(window.params)) {
        window.params = DEFAULT_PARAMS;
    }
};



function saveParams() {
    localStorage.params = JSON.stringify(window.params);
};



function resetParams() {
    window.params = DEFAULT_PARAMS;
    saveParams();
    setupSliders();
    setupDropdowns();
};



function setupPersistence() {
    loadParams();
    window.onbeforeunload = saveParams;
};



/*************************************************************************************************/
/*  Entry point                                                                                  */
/*************************************************************************************************/

// On load, create the websocket and set the onnopen, onmessage, and onclose callbacks.
function load() {

    setupPersistence();

    setupSliders();
    setupDropdowns();
    setupInputs();
    setupButtons();
    setupRaster();
    setupRaw();
    setupCluster();
    setupWebsocket();

    setRawImage();
    updateDuration();
};



$(document).ready(function () {
    load();
});
