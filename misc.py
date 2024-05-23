import numpy as np


def normalize(npz_ptcloud):
    ans = np.copy(npz_ptcloud)
    minx = 0
    miny = 0
    minz = 0

    maxx = 0
    maxy = 0
    maxz = 0

    for p in ans: #[x, y, z] represents a point in space
        minx = min(minx, p[0])
        miny = min(miny, p[1])
        minz = min(minz, p[2])

        maxx = max(maxx, p[0])
        maxy = max(maxy, p[1])
        maxz = max(maxz, p[2])

    # newx = round((p[0] + 43.5) * 63/68)
    # newy = round((p[1] + 70) * 63/72)
    # newz = round((p[2] + 34) * 63/71)
    
    for p in ans:
        p[0] = (p[0] + 43.5) * 63/68
        p[1] = (p[1] + 70) * 63/72
        p[2] = (p[2] + 34) * 63/71

    return ans

def round_ptcloud(npz_ptcloud):
    ans = np.copy(npz_ptcloud)
    for p in ans:
        p[0] = int(round(p[0]))
        p[1] = int(round(p[1]))
        p[2] = int(round(p[2]))

    return ans
