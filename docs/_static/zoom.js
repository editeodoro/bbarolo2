/* BBarolo documentation: a zoom viewer for the large figures (the corner plots). An image with the class bb-zoom
   (in MyST: ":class: bb-zoom" in the figure directive) opens, on a click or Enter, in a window-wide overlay: fitted
   to the window, zoomed with the mouse wheel, a pinch, the keys + and -, or a double click (fit <-> 1:1), moved by
   dragging or with the arrow keys; Esc, the close button or a click on the background close it. */
(function () {
  "use strict";
  let box, img, s = 1, tx = 0, ty = 0, fit = 1, opener = null;
  const pts = new Map();          // the pointers down: id -> {x, y}
  let pinch = null, travel = 0, downOnImage = false;

  function build() {
    box = document.createElement("div");
    box.className = "bb-zoom-box";
    box.setAttribute("role", "dialog");
    box.setAttribute("aria-modal", "true");
    box.innerHTML = '<img alt=""><button type="button" class="bb-zoom-close" aria-label="Close">&times;</button>' +
      '<div class="bb-zoom-hint">Scroll or pinch to zoom, drag to move, double click for 1:1, Esc to close</div>';
    img = box.querySelector("img");
    img.draggable = false;
    box.querySelector(".bb-zoom-close").addEventListener("click", close);
    box.addEventListener("wheel", onWheel, { passive: false });
    box.addEventListener("pointerdown", onDown);
    box.addEventListener("pointermove", onMove);
    box.addEventListener("pointerup", onUp);
    box.addEventListener("pointercancel", onUp);
    box.addEventListener("dblclick", (e) => {
      if (onImage(e)) zoomAt(e.clientX, e.clientY, Math.abs(s - fit) < 1e-3 ? Math.max(1, 2 * fit) : fit);
    });
    document.addEventListener("keydown", onKey);
    document.body.appendChild(box);
  }

  function apply() {
    // keep the image centred when it is smaller than the window, and on the window when it is larger
    const W = box.clientWidth, H = box.clientHeight, w = img.naturalWidth * s, h = img.naturalHeight * s;
    tx = w <= W ? (W - w) / 2 : Math.min(0, Math.max(W - w, tx));
    ty = h <= H ? (H - h) / 2 : Math.min(0, Math.max(H - h, ty));
    img.style.transform = `translate(${tx}px, ${ty}px) scale(${s})`;
  }

  // (the events of a captured pointer have the overlay as target: where they are says whether they are on the image)
  function onImage(e) {
    const r = img.getBoundingClientRect();
    return e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
  }

  function isOpen() {
    return box && box.classList.contains("bb-zoom-open");
  }

  function zoomAt(x, y, ns) {
    ns = Math.min(Math.max(ns, fit), Math.max(2, 2 * fit));
    tx = x - (x - tx) * ns / s;
    ty = y - (y - ty) * ns / s;
    s = ns;
    apply();
  }

  function reset() {
    fit = Math.min(box.clientWidth / img.naturalWidth, box.clientHeight / img.naturalHeight);
    s = fit;
    apply();
  }

  function open(src, alt, from) {
    if (!box) build();
    opener = from;
    img.style.transform = "";
    img.alt = alt || "";
    box.classList.add("bb-zoom-open");
    document.documentElement.classList.add("bb-zoom-noscroll");
    box.querySelector(".bb-zoom-close").focus();
    // fitted once, when the image is there (at once if the browser has it: the image of the page)
    img.onload = () => { img.onload = null; reset(); };
    img.src = src;
    if (img.complete && img.naturalWidth) img.onload();
  }

  function close() {
    box.classList.remove("bb-zoom-open");
    document.documentElement.classList.remove("bb-zoom-noscroll");
    pts.clear();
    pinch = null;
    if (opener) opener.focus();
  }

  function onWheel(e) {
    e.preventDefault();
    zoomAt(e.clientX, e.clientY, s * Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015)));
  }

  function onDown(e) {
    if (e.target.closest(".bb-zoom-close")) return;
    box.setPointerCapture(e.pointerId);
    pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pts.size === 1) {
      travel = 0;
      downOnImage = onImage(e);
    }
    if (pts.size === 2) {
      const [a, b] = [...pts.values()];
      pinch = { d: Math.hypot(a.x - b.x, a.y - b.y), s: s };
    }
  }

  function onMove(e) {
    const p = pts.get(e.pointerId);
    if (!p) return;
    const dx = e.clientX - p.x, dy = e.clientY - p.y;
    p.x = e.clientX;
    p.y = e.clientY;
    travel += Math.abs(dx) + Math.abs(dy);
    if (pts.size === 2 && pinch) {
      const [a, b] = [...pts.values()];
      zoomAt((a.x + b.x) / 2, (a.y + b.y) / 2, pinch.s * Math.hypot(a.x - b.x, a.y - b.y) / pinch.d);
    } else if (pts.size === 1) {
      tx += dx;
      ty += dy;
      apply();
      box.classList.add("bb-zoom-dragging");
    }
  }

  function onUp(e) {
    const wasDown = pts.delete(e.pointerId);
    if (pts.size < 2) pinch = null;
    box.classList.remove("bb-zoom-dragging");
    // a click (no drag) on the background closes the viewer
    if (wasDown && pts.size === 0 && travel < 5 && e.type === "pointerup" && !downOnImage) close();
  }

  function onKey(e) {
    if (!isOpen()) return;
    const W = box.clientWidth / 2, H = box.clientHeight / 2, step = 80;
    if (e.key === "Escape") close();
    else if (e.key === "+" || e.key === "=") zoomAt(W, H, s * 1.25);
    else if (e.key === "-") zoomAt(W, H, s / 1.25);
    else if (e.key === "0") reset();
    else if (e.key === "ArrowLeft") { tx += step; apply(); }
    else if (e.key === "ArrowRight") { tx -= step; apply(); }
    else if (e.key === "ArrowUp") { ty += step; apply(); }
    else if (e.key === "ArrowDown") { ty -= step; apply(); }
    else if (e.key === "Tab") e.preventDefault();          // the focus stays on the close button
    else return;
    e.preventDefault();
  }

  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("img.bb-zoom").forEach((im) => {
      im.tabIndex = 0;
      im.setAttribute("role", "button");
      im.title = "Click to enlarge";
      const go = (e) => {
        e.preventDefault();               // (Sphinx wraps an image with a width in a link to the file)
        open(im.currentSrc || im.src, im.alt, im);
      };
      im.addEventListener("click", go);
      im.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") go(e); });
    });
    window.addEventListener("resize", () => { if (isOpen()) reset(); });
  });
})();
