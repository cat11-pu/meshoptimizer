#!/usr/bin/env python3
# Renders animation-replay comparison frames using the same skinning math as the C++ harness.
# Outputs a PNG per mesh: source surface, plain LOD surface and skinned LOD surface in a bent pose.
import math
import os
import struct
import subprocess
import sys

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))


def read_mesh_dump(path):
    with open(path, "rb") as f:
        data = f.read()

    magic, vcount, icount = struct.unpack_from("<III", data, 0)
    assert magic == 0x534B494E, "bad skin dump magic"
    off = 12
    vertices = []

    for _ in range(vcount):
        px, py, pz = struct.unpack_from("<fff", data, off)
        off += 12
        weights = struct.unpack_from("<ffff", data, off)
        off += 16
        bones = struct.unpack_from("<IIII", data, off)
        off += 16
        vertices.append((px, py, pz, weights, bones))

    indices = list(struct.unpack_from("<%dI" % icount, data, off))
    return vertices, indices


def rotate(point, pivot, axis, angle):
    q = [point[i] - pivot[i] for i in range(3)]
    c, s = math.cos(angle), math.sin(angle)
    dot = sum(q[i] * axis[i] for i in range(3))
    cross = [
        axis[1] * q[2] - axis[2] * q[1],
        axis[2] * q[0] - axis[0] * q[2],
        axis[0] * q[1] - axis[1] * q[0],
    ]
    return [q[i] * c + cross[i] * s + axis[i] * dot * (1 - c) + pivot[i] for i in range(3)]


def skin(vertex, pivots, axes, amps, frame):
    _, _, _, weights, bones = vertex
    base = vertex[:3]
    out = [0.0, 0.0, 0.0]

    for k in range(4):
        if weights[k] == 0:
            continue

        bone = bones[k]
        angle = amps[bone] * math.sin(frame * 2 * math.pi)
        p = rotate(base, pivots[bone], axes[bone], angle)

        for i in range(3):
            out[i] += p[i] * weights[k]

    return out


def project(p, scale, center):
    # simple orthographic projection onto the XY plane
    return (center[0] + p[0] * scale, center[1] - p[1] * scale)


def render(vertices, indices, pivots, axes, amps, frame, color, size):
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)

    positions = [skin(v, pivots, axes, amps, frame) for v in vertices]

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
    scale = (size[1] - 60) / span
    center = (size[0] * 0.5, size[1] * 0.55)

    for t in range(0, len(indices), 3):
        pts = [project(positions[indices[t + e]], scale, center) for e in range(3)]
        draw.polygon(pts, outline=color)

    return img


def main():
    dumps = [
        ("arm_cylinder", "arm"),
        ("cloth_sheet", "cloth"),
        ("leg_knee", "knee"),
    ]

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)

    for name, _ in dumps:
        source_path = os.path.join(HERE, "results", name + ".source.skin")
        plain_path = os.path.join(HERE, "results", name + ".plain.skin")
        skinned_path = os.path.join(HERE, "results", name + ".skinned.skin")

        if not os.path.exists(source_path):
            print("missing %s; run the C++ dumper first" % source_path)
            continue

        src_v, src_i = read_mesh_dump(source_path)
        plain_v, plain_i = read_mesh_dump(plain_path)
        skin_v, skin_i = read_mesh_dump(skinned_path)

        pivots, axes, amps = read_rig(name)
        frame = 0.25  # quarter cycle: close to maximum bend

        size = (360, 360)
        panels = [
            ("source %d tris" % (len(src_i) // 3), src_v, src_i, (90, 90, 90)),
            ("plain simplify %d tris" % (len(plain_i) // 3), plain_v, plain_i, (200, 60, 60)),
            ("skinned simplify %d tris" % (len(skin_i) // 3), skin_v, skin_i, (30, 120, 60)),
        ]

        strip = Image.new("RGB", (size[0] * 3, size[1] + 28), (255, 255, 255))
        draw = ImageDraw.Draw(strip)

        for col, (label, verts, idx, color) in enumerate(panels):
            panel = render(verts, idx, pivots, axes, amps, frame, color, size)
            strip.paste(panel, (col * size[0], 28))
            draw.text((col * size[0] + 8, 8), label, fill=(0, 0, 0))

        out = os.path.join(HERE, "results", name + "_replay.png")
        strip.save(out)
        print("wrote", out)


def read_rig(name):
    # keep rigs in sync with skinned_meshes.h; values are simple enough to mirror here
    if name == "arm_cylinder":
        pivots = [[0, -1, 0], [0, 0, 0]]
        axes = [[1, 0, 0], [1, 0, 0]]
        amps = [0.0, 1.1]
    elif name == "leg_knee":
        pivots = [[0, 0, 0], [0, 0, 0]]
        axes = [[0, 0, 1], [0, 0, 1]]
        amps = [-0.9, 0.9]
    else:
        pivots = [[-0.5, -0.5, 0], [0.5, -0.5, 0], [-0.5, 0.5, 0], [0.5, 0.5, 0]]
        axes = [[0, 0, 1]] * 4
        amps = [-0.5, 0.5, -0.5, 0.5]

    return pivots, axes, amps


if __name__ == "__main__":
    main()
