const COUNT = 2000000; // max number of spikes
const DEFAULT_EID = "0851db85-2889-4070-ac18-a40e8ebd96ba";

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
                        "buffer": getParams()
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

// Generate a JSON request to update to a new EID.
function updateRequest(eid) {
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
                        "session": sessionJSON(eid)
                    }
                }
            },
        ]
    };

    return req;
}

// Send requests to the server.
function submit(contents) {
    if (!window.websocket) return;
    // if (window.websocket.readyState != 1) return;
    if (!window.websocket.connected) return;
    window.websocket.emit("request", contents);
}

submit = throttle(submit, 100);

// Display an array buffer.
function show(arrbuf) {
    const blob = new Blob([arrbuf]);
    const url = URL.createObjectURL(blob);
    const img = document.getElementById('img');
    img.src = url;
}

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

function scaleSize(x) {
    return Math.log(1 + x);
}

// On load, create the websocket and set the onnopen, onmessage, and onclose callbacks.
function load() {

    initSlider('sliderAlpha', [0, 1], [0, 1]);
    initSlider('sliderSize', [0.01, 1], [0, 5]);
    initSlider('sliderColormap', [0, 1], [0, 1]);


    window.websocket = io();

    // window.websocket = new WebSocket("ws://localhost:5000/");
    // window.websocket.binaryType = "arraybuffer";

    // window.websocket.onopen = function () {
    window.websocket.on("connect", () => {
        console.log('socket connected');

        var contents = loadJSON();
        submit(contents);
    });

    // window.websocket.onmessage = ({ data }) => {
    window.websocket.on("image", (data) => {
        // display the received image
        let img = data["image"];
        if (img instanceof ArrayBuffer) {
            show(img);
        }
    });

    // window.websocket.onclose = function () {
    window.websocket.on("disconnect", () => {
        console.log('socket disconnected');
    });




    window.zoom = 1;
    window.shift = 0;

    const img = document.getElementById('img');

    img.onwheel = function (e) {
        e.preventDefault();
        let d = -e.deltaY / Math.abs(e.deltaY);
        let z = window.zoom;

        var rect = e.target.getBoundingClientRect();
        var x = e.clientX - rect.left; //x position within the element.
        var y = e.clientY - rect.top;  //y position within the element.

        let w = e.srcElement.width;

        window.zoom *= (1 + .5 * d);
        window.zoom = Math.max(1, window.zoom);

        let center = -1 + 2 * x / w;
        window.shift -= center * (1 / z - 1 / window.zoom);

        if (window.zoom != z)
            updateMVP();
    }

    img.ondblclick = function (e) {
        reset();
    }



    onSliderChange('sliderColormap', function (min, max) {
        var arr = new StructArray(1, [
            ["cmap_range", "float32", 2],
        ]);
        arr.set(0, "cmap_range", [min, max]);

        const req = {
            "version": "1.0",
            "requests": [
                {
                    "action": "upload",
                    "type": "dat",
                    "id": 12,
                    "content": {
                        "offset": 16,
                        "data": {
                            "mode": "base64",
                            "buffer": arr.b64()
                        }
                    }
                },
            ]
        };

        submit(req);
    });

    onSliderChange('sliderSize', function (min, max) {
        var arr = new StructArray(1, [
            ["size_range", "float32", 2],
        ]);
        window.sizeMin = min;
        window.sizeMax = max;
        arr.set(0, "size_range", [min, max * scaleSize(window.zoom)]);

        const req = {
            "version": "1.0",
            "requests": [
                {
                    "action": "upload",
                    "type": "dat",
                    "id": 12,
                    "content": {
                        "offset": 8,
                        "data": {
                            "mode": "base64",
                            "buffer": arr.b64()
                        }
                    }
                },
            ]
        };

        submit(req);
    });

    onSliderChange('sliderAlpha', function (min, max) {
        var arr = new StructArray(1, [
            ["alpha_range", "float32", 2],
        ]);
        arr.set(0, "alpha_range", [min, max]);

        const req = {
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
                            "buffer": arr.b64()
                        }
                    }
                },
            ]
        };

        submit(req);
    });


    document.getElementById('selectSession').onchange = function (e) {
        select(e.target.value);
    }

    document.getElementById('selectColormap').onchange = function (e) {
        var cmap_id = parseInt(e.target.value);

        var arr = new StructArray(1, [
            ["cmap_id", "uint32", 1],
        ]);
        arr.set(0, "cmap_id", [cmap_id]);

        const req = {
            "version": "1.0",
            "requests": [
                {
                    "action": "upload",
                    "type": "dat",
                    "id": 12,
                    "content": {
                        "offset": 24,
                        "data": {
                            "mode": "base64",
                            "buffer": arr.b64()
                        }
                    }
                },
            ]
        };

        submit(req);
    }

    document.getElementById('selectColor').onchange = function (e) {
        var feature = e.target.value;

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
                                "color": feature
                            }
                        }
                    }
                },
            ]
        };

        submit(req);


    }

}

// Select another session.
function select(eid) {
    var contents = updateRequest(eid);
    submit(contents);
}

window.sizeMin = 0.01;
window.sizeMax = 1;


// Return a MVP structure for a given pan and zoom.
function mvp(px, py, zx, zy) {
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

    return arr;
}

// Return a MVP structure for a given pan and zoom.
function getParams() {
    var arr = new StructArray(1, [
        ["alpha_range", "float32", 2],
        ["size_range", "float32", 2],
        ["cmap_range", "float32", 2],
        ["cmap_id", "uint32", 1],
    ]);

    arr.set(0, "alpha_range", [0, 1]);
    arr.set(0, "size_range", [.01, 1]);
    arr.set(0, "cmap_range", [0, 1]);
    arr.set(0, "cmap_id", [0]);

    return arr.b64();
}

// Send the updated MVP struct to the server.
function updateMVP() {
    arr = mvp(window.shift, 0, window.zoom, 1);


    var arrs = new StructArray(1, [
        ["size_range", "float32", 2],
    ]);
    arrs.set(0, "size_range", [window.sizeMin, window.sizeMax * scaleSize(window.zoom)]);

    const req = {
        "version": "1.0",
        "requests": [
            {
                "action": "upload",
                "type": "dat",
                "id": 10,
                "content": {
                    "offset": 0,
                    "data": {
                        "mode": "base64",
                        "buffer": arr.b64()
                    }
                }
            },
            {
                "action": "upload",
                "type": "dat",
                "id": 12,
                "content": {
                    "offset": 8,
                    "data": {
                        "mode": "base64",
                        "buffer": arrs.b64()
                    }
                }
            },
        ]
    };

    submit(req);
}

// Reset the view.
function reset() {
    window.zoom = 1;
    window.shift = 0;

    updateMVP();
}
