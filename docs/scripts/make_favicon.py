"""The browser-tab icons of the documentation: the "3D" mark of the logo (the 3 on the D), cut from
docs/figures/3DBAROLO_logo_light.pdf and 3DBAROLO_logo_dark.pdf.

    python3 docs/scripts/make_favicon.py              # writes docs/_static/favicon.*, apple-touch-icon.png
    python3 docs/scripts/make_favicon.py --out /tmp/icons

Files (committed; docs/_templates/layout.html links them):
  favicon.svg           the mark in the colours of the light logo, and of the dark logo when the browser is dark
  favicon.ico           16, 32 and 48 px, light-logo colours (browsers without SVG icons)
  apple-touch-icon.png  180 px on white (iOS/Safari home screen and bookmarks)

Needs mutool (MuPDF: the outlines, read exactly from the PDFs), pdftocairo (poppler: the bitmaps) and Pillow.
The outlines are polylines of thousands of tiny segments: they are simplified (Ramer-Douglas-Peucker, 0.03 pt
in a 109 pt square), which leaves the shape unchanged to the eye and makes the SVG ~20 times smaller.
"""
import argparse
import io
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.dirname(HERE)
LOGO = os.path.join(DOCS, "figures", "3DBAROLO_logo_{}.pdf")
MARGIN = 2.0                       # empty border around the mark (pt)
TOL = 0.03                         # simplification tolerance (pt)


def mark(mode):
    """The two filled paths of the mark in the logo PDF of mode ("light"/"dark"), the D and the 3:
    [(colour '#rrggbb', [ring, ...])], each ring an N x 2 array of PDF points (y down), filled even-odd."""
    trace = subprocess.run(["mutool", "trace", LOGO.format(mode)], check=True, capture_output=True, text=True).stdout
    out = []
    for fill in ET.fromstring(trace).iter("fill_path"):
        if fill.get("winding") != "eofill":     # the dark logo starts with a background rectangle
            continue
        if fill.get("transform") != "1 0 0 1 0 0" or any(e.tag == "curveto" for e in fill):
            raise RuntimeError("the mark is not a plain polyline: this script needs updating")
        rings, ring = [], []
        for e in fill:
            if e.tag == "moveto" and ring:
                rings.append(np.array(ring))
                ring = []
            if e.tag in ("moveto", "lineto"):
                ring.append((float(e.get("x")), float(e.get("y"))))
        if ring:
            rings.append(np.array(ring))
        col = "#" + "".join(f"{round(float(v) * 255):02x}" for v in fill.get("color").split())
        out.append((col, [r for r in rings if len(r) >= 3]))
        if len(out) == 2:
            return out
    raise RuntimeError("the mark was not found in the logo")


def box(paths):
    """The square around the mark: x0, y0, side (pt)."""
    pts = np.vstack([r for _, rings in paths for r in rings])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    side = (hi - lo).max() + 2 * MARGIN
    x0, y0 = (lo + hi) / 2 - side / 2
    return x0, y0, side


def _rdp(pts, tol):
    """Ramer-Douglas-Peucker simplification of a polyline (N x 2 array); keeps the end points."""
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        ab = b - a
        n = np.hypot(*ab)
        seg = pts[i + 1:j] - a
        dist = np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / n if n > 0 else np.hypot(seg[:, 0], seg[:, 1])
        k = int(np.argmax(dist))
        if dist[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return pts[keep]


def path_data(rings, origin):
    """SVG path data of closed rings: simplified, relative to origin (x0, y0), relative moves, 2 decimals."""
    out = []
    for v in rings:
        v = v - origin
        if np.allclose(v[0], v[-1]):
            v = v[:-1]
        # a closed ring: split it at the point farthest from the start, simplify both halves
        far = int(np.argmax(np.hypot(*(v - v[0]).T)))
        ring = np.vstack([v, v[:1]])
        pts = np.vstack([_rdp(ring[:far + 1], TOL), _rdp(ring[far:], TOL)[1:-1]])
        q = np.round(pts * 100).astype(int)
        s = f"M{q[0, 0] / 100:g} {q[0, 1] / 100:g}" + "".join(f"l{dx / 100:g} {dy / 100:g}"
                                                             for dx, dy in np.diff(q, axis=0))
        out.append(s.replace(" -", "-") + "z")
    return "".join(out)


def favicon_svg(light, dark, bx):
    """The SVG icon: light-logo colours, dark-logo colours under prefers-color-scheme: dark."""
    (dl, d), (tl, t) = light
    (dd, _), (td, _) = dark
    x0, y0, side = bx
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {side:.2f} {side:.2f}">'
            f"<style>.d{{fill:{dl}}}.t{{fill:{tl}}}"
            f"@media (prefers-color-scheme:dark){{.d{{fill:{dd}}}.t{{fill:{td}}}}}</style>"
            f'<path class="d" fill-rule="evenodd" d="{path_data(d, (x0, y0))}"/>'
            f'<path class="t" fill-rule="evenodd" d="{path_data(t, (x0, y0))}"/></svg>\n')


def render(size, bx, oversample=8):
    """The mark as an RGBA image of size x size px, rendered from the light logo PDF and averaged down
    (the dark logo PDF has an opaque background)."""
    x0, y0, side = bx
    big = size * oversample
    dpi = 72 * big / side
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "mark")
        subprocess.run(["pdftocairo", "-png", "-transp", "-singlefile", "-r", f"{dpi:.6f}",
                        "-x", str(round(x0 * dpi / 72)), "-y", str(round(y0 * dpi / 72)),
                        "-W", str(big), "-H", str(big), LOGO.format("light"), base], check=True)
        im = Image.open(base + ".png").convert("RGBA")
        im.load()
    return im.resize((size, size), Image.BOX)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(DOCS, "_static"), help="output folder (default docs/_static)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    light, dark = mark("light"), mark("dark")
    if any(len(a) != len(b) or not all(np.array_equal(u, v) for u, v in zip(a, b))
           for (_, a), (_, b) in zip(light, dark)):
        raise RuntimeError("the mark differs between the light and the dark logo")
    bx = box(light)
    with open(os.path.join(args.out, "favicon.svg"), "w") as f:
        f.write(favicon_svg(light, dark, bx))

    # ICO: each size rendered on its own (not resized from the largest)
    sizes = (16, 32, 48)
    icons = {s: render(s, bx) for s in sizes}
    buf = io.BytesIO()
    icons[48].save(buf, format="ICO", sizes=[(s, s) for s in sizes], append_images=[icons[s] for s in sizes[:-1]])
    with open(os.path.join(args.out, "favicon.ico"), "wb") as f:
        f.write(buf.getvalue())

    # Apple touch icon: opaque (iOS fills transparency with black), the mark at ~3/4 of the square
    touch = Image.new("RGBA", (180, 180), (255, 255, 255, 255))
    touch.alpha_composite(render(138, bx), (21, 21))
    touch.convert("RGB").save(os.path.join(args.out, "apple-touch-icon.png"), optimize=True)

    for name in ("favicon.svg", "favicon.ico", "apple-touch-icon.png"):
        p = os.path.join(args.out, name)
        print(f"{p}  {os.path.getsize(p)} bytes")


if __name__ == "__main__":
    main()
