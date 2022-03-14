/*************************************************************************************************/
/*  Array                                                                                        */
/*************************************************************************************************/


/*************************************************************************************************/
/*  Constants                                                                                    */
/*************************************************************************************************/

_DTYPE_MAPPING = {
    "uint8": [Uint8Array, 1],

    "uint16": [Uint16Array, 2],
    "int16": [Int16Array, 2],

    "uint32": [Uint32Array, 4],
    "int32": [Int32Array, 4],

    "float32": [Float32Array, 4],
    "float64": [Float64Array, 8],
};



/*************************************************************************************************/
/*  Utils                                                                                        */
/*************************************************************************************************/

function tob64(buffer) {
    var binary = '';
    var bytes = new Uint8Array(buffer);
    var len = bytes.byteLength;
    for (var i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary);
}



/*************************************************************************************************/
/*  Array class                                                                                  */
/*************************************************************************************************/

function StructArray(count, dtype) {
    this.count = count;
    this.dtype = dtype;
    this.fields = {};
    this.itemsize = 0;
    for (let i = 0; i < dtype.length; i++) {

        let name = dtype[i][0];
        let dt = dtype[i][1];
        let count = dtype[i][2];
        let size = _DTYPE_MAPPING[dt][1];

        this.fields[name] = {
            type: dt,
            count: count,
            itemsize: size,
            offset: this.itemsize
        };
        this.itemsize += count * size;
    }
    this.buffer = new ArrayBuffer(count * this.itemsize);
};

StructArray.prototype.set = function (idx, name, values) {
    let field = this.fields[name];
    let vt = _DTYPE_MAPPING[field.type][0];
    let view = new vt(this.buffer, this.itemsize * idx + field.offset, field.count);
    for (let i = 0; i < field.count; i++) {
        view[i] = values[i];
    }
};

StructArray.prototype.b64 = function () {
    return tob64(this.buffer);
}
