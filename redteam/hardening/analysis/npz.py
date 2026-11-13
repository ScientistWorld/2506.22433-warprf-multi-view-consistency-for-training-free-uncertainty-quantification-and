import zipfile, ast, array, struct
def _parse(buf):
    assert buf[:6] == b"\x93NUMPY"
    major = buf[6]
    if major == 1:
        hl = struct.unpack("<H", buf[8:10])[0]; off = 10
    else:
        hl = struct.unpack("<I", buf[8:12])[0]; off = 12
    hdr = ast.literal_eval(buf[off:off+hl].decode("latin1"))
    data = buf[off+hl:]
    d = hdr["descr"]; shape = hdr["shape"]
    code = {"<f8":"d","<f4":"f","|u1":"B","<i8":"q","<f2":None,"|b1":"B"}[d]
    if code is None:
        # float16
        n = len(data)//2
        vals = [struct.unpack("<e", data[2*i:2*i+2])[0] for i in range(n)]
    else:
        a = array.array(code); a.frombytes(data); vals = a.tolist()
    return shape, vals
def load(path, keys=None):
    out = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            k = n[:-4]
            if keys is not None and k not in keys: continue
            out[k] = _parse(z.read(n))
    return out
