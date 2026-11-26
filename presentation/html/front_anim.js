// The pooled Yang front, slowly swinging (slide "The front"). Canvas 2D via plot3d.js; data in assets/front_data.js
// (build_figs.py front): 573 networks of the 4 NSGA-III searches, the 4 points (5-seed means), and the smoothed
// worst-task-accuracy surface that the next slide's top view contours. Drag to turn it by hand.
(function () {
  const tick = v => ({ v, label: String(v) });
  const CFG = {
    X: [-2.85, -0.2], Y: [-2.75, -1.3], Z: [0.6, 1.0],
    XT: [-2.5, -2, -1.5, -1, -0.5].map(tick), YT: [-2.6, -2.2, -1.8].map(tick),
    ZT: [0.6, 0.7, 0.8, 0.9, 1.0].map(v => ({ v, label: v.toFixed(1) })),
    xTitle: "rate cost (log₁₀)", yTitle: "synaptic cost (log₁₀)", zTitle: "worst-task accuracy",
    aspect: [1, 1, 0.66], margin: { l: 30, r: 104, t: 8, b: 62 },
  };
  // camera swings slowly between AZ[0] and AZ[1] degrees (one way in PERIOD / 2 s): always a 3/4 view, axes readable
  const PERIOD = 28, AZ = [-72, -18], EL = 26 * Math.PI / 180;
  const azAt = t => (AZ[0] + AZ[1]) / 2 - (AZ[1] - AZ[0]) / 2 * Math.cos(2 * Math.PI * t / PERIOD);

  const rgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
  const css = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  const clamp = (v, a, b) => Math.min(Math.max(v, a), b);

  function star(ctx, x, y, r) {
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
      const a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * 0.45 : r;
      ctx[i ? "lineTo" : "moveTo"](x + rr * Math.cos(a), y + rr * Math.sin(a));
    }
    ctx.closePath();
  }

  function setup(root) {
    const D = window.FRONT_DATA;
    if (!D) return;
    const canvas = root.querySelector("canvas");
    const P = Plot3D(canvas, CFG), ctx = P.ctx;
    const { Z } = CFG;
    const lut = D.lut.map(rgb);
    const col = z => lut[Math.round(clamp(z, 0, 1) * 100)];

    // surface cells: the 4 corners of each grid square where all are inside the hull
    const cells = [];
    for (let j = 0; j + 1 < D.ys.length; j++) for (let i = 0; i + 1 < D.xs.length; i++) {
      const zs = [D.zz[j][i], D.zz[j][i + 1], D.zz[j + 1][i + 1], D.zz[j + 1][i]];
      if (zs.some(v => v === null)) continue;
      const v = [[D.xs[i], D.ys[j]], [D.xs[i + 1], D.ys[j]], [D.xs[i + 1], D.ys[j + 1]], [D.xs[i], D.ys[j + 1]]]
        .map(([x, y], k) => [x, y, Math.max(zs[k], Z[0])]);
      cells.push({ v, c: col(zs.reduce((s, x) => s + x, 0) / 4) });
    }
    const nets = D.nets.map(([x, y, z, k]) => ({ x, y, z: k === 0 ? Z[0] : z, k, c: col(z) }));

    // rotation: automatic, or by dragging (then it carries on from where it was left)
    let azOff = 0, elNow = EL, drag = null;
    canvas.style.cursor = "grab";
    canvas.addEventListener("pointerdown", e => {
      drag = { x: e.clientX, y: e.clientY, az: azOff, el: elNow }; canvas.setPointerCapture(e.pointerId);
      canvas.style.cursor = "grabbing"; e.stopPropagation();
    });
    canvas.addEventListener("pointermove", e => {
      if (!drag) return;
      azOff = drag.az - (e.clientX - drag.x) * 0.45;
      elNow = clamp(drag.el + (e.clientY - drag.y) * 0.006, 0.05, 1.3);
    });
    const stop = () => { if (drag) { drag = null; canvas.style.cursor = "grab"; } };
    canvas.addEventListener("pointerup", stop); canvas.addEventListener("pointercancel", stop);

    const fitAz = []; for (let a = AZ[0]; a <= AZ[1]; a += 3) fitAz.push(a);
    let last = null;

    function draw(t) {
      if (last !== null && drag) azOff -= azAt(t) - azAt(last);          // hold still while dragging
      last = t;
      P.resize();
      P.fit(fitAz, EL);
      P.az = (azAt(t) + azOff) * Math.PI / 180;
      P.el = elNow;
      ctx.globalAlpha = clamp(t / 0.6, 0, 1);
      P.drawBox();
      P.drawAxes();

      // networks that fail a task: crosses on the floor
      ctx.strokeStyle = "rgba(110,110,115,0.55)"; ctx.lineWidth = 1.2;
      nets.forEach(n => {
        if (n.k !== 0) return;
        const q = P.proj(n.x, n.y, Z[0]), s = 3.2 * q.f;
        P.line({ x: q.x - s, y: q.y - s }, { x: q.x + s, y: q.y + s });
        P.line({ x: q.x - s, y: q.y + s }, { x: q.x + s, y: q.y - s });
      });

      // surface cells and networks together, far to near
      const items = [];
      cells.forEach(c => {
        const q = c.v.map(v => P.proj(...v));
        items.push({ d: (q[0].d + q[1].d + q[2].d + q[3].d) / 4, cell: q, c: c.c });
      });
      nets.forEach(n => { if (n.k) { const q = P.proj(n.x, n.y, n.z); items.push({ d: q.d - 0.02, q, n }); } });
      items.sort((a, b) => b.d - a.d);
      items.forEach(it => {
        if (it.cell) {
          P.poly(it.cell); ctx.fillStyle = css(it.c, 0.5); ctx.fill();
          ctx.lineWidth = 0.6; ctx.strokeStyle = "rgba(255,255,255,0.35)"; ctx.stroke();
          return;
        }
        const { q, n } = it;
        ctx.beginPath();
        if (n.k === 2) {
          ctx.arc(q.x, q.y, 5.2 * q.f, 0, 2 * Math.PI); ctx.fillStyle = css(n.c, 1); ctx.fill();
          ctx.lineWidth = 0.9; ctx.strokeStyle = "rgba(29,29,31,0.7)"; ctx.stroke();
        } else {
          ctx.arc(q.x, q.y, 2.4 * q.f, 0, 2 * Math.PI); ctx.fillStyle = "rgba(90,90,95,0.45)"; ctx.fill();
        }
      });

      // the 4 points on top: drop line and floor mark, then the star
      D.points.forEach(p => {
        const q = P.proj(p.x, p.y, p.z), f = P.proj(p.x, p.y, Z[0]), c = rgb(p.col);
        ctx.setLineDash([3, 3]); ctx.strokeStyle = css(c, 0.8); ctx.lineWidth = 1.4; P.line(q, f); ctx.setLineDash([]);
        ctx.beginPath(); ctx.ellipse(f.x, f.y, 6 * f.f, 2.6 * f.f, 0, 0, 2 * Math.PI); ctx.fillStyle = css(c, 0.35); ctx.fill();
        star(ctx, q.x, q.y, 13 * q.f);
        ctx.lineWidth = 4; ctx.strokeStyle = "rgba(255,255,255,0.9)"; ctx.stroke();
        ctx.fillStyle = p.col; ctx.fill();
        ctx.lineWidth = 1.1; ctx.strokeStyle = "#1d1d1f"; ctx.stroke();
      });
      ctx.globalAlpha = 1;
    }

    Plot3D.onSlide(root, t => { if (t < 0.05) { azOff = 0; elNow = EL; last = null; } draw(t); });
  }

  function init() { document.querySelectorAll(".front-anim").forEach(setup); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
