// Small 3D plotting helper for the deck's canvas animations (nsga_anim.js, front_anim.js): projection, a box with
// back panes, grid, ticks and axis titles, and drawing primitives. No libraries.
//   const P = Plot3D(canvas, cfg)   cfg: {X, Y, Z: [min, max] in world units; XT, YT, ZT: [{v, label}];
//                                         xTitle, yTitle, zTitle; aspect: [ax, ay, az]; margin: {l, r, t, b}}
(function () {
  const INK = "#1d1d1f", MUTED = "#6e6e73", GRID = "#e3e3e8", PANE = "#f7f7f9", EDGE = "#c7c7cc";
  const FONT = 'Inter, "Segoe UI", -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif';

  window.Plot3D = function (canvas, cfg) {
    const ctx = canvas.getContext("2d");
    const { X, Y, Z } = cfg, A = cfg.aspect || [1, 1, 0.8], M = cfg.margin || { l: 60, r: 90, t: 20, b: 50 };
    const P = { ctx, W: 0, H: 0, az: 0, el: 0.4, S: 1, CX: 0, CY: 0, INK, MUTED };
    let fitKey = "";

    P.resize = function () {
      const r = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
      P.W = canvas.offsetWidth; P.H = canvas.offsetHeight;
      const bw = Math.round(r.width * dpr), bh = Math.round(r.height * dpr);
      if (canvas.width !== bw || canvas.height !== bh) { canvas.width = bw; canvas.height = bh; }
      ctx.setTransform(bw / P.W, 0, 0, bh / P.H, 0, 0);
      ctx.clearRect(0, 0, P.W, P.H);
    };

    P.proj = function (x, y, z) {
      const u = A[0] * (2 * (x - X[0]) / (X[1] - X[0]) - 1), v = A[1] * (2 * (y - Y[0]) / (Y[1] - Y[0]) - 1),
            w = A[2] * (2 * (z - Z[0]) / (Z[1] - Z[0]) - 1);
      const xr = u * Math.cos(P.az) - v * Math.sin(P.az), yr = u * Math.sin(P.az) + v * Math.cos(P.az);
      const up = yr * Math.sin(P.el) + w * Math.cos(P.el), depth = yr * Math.cos(P.el) - w * Math.sin(P.el);
      const f = 7 / (7 + depth);
      return { x: P.CX + P.S * xr * f, y: P.CY - P.S * up * f, d: depth, f };
    };

    // scale and centre so the box fits the canvas for every azimuth in azs (degrees) at elevation el (radians):
    // no zoom change while the camera moves
    P.fit = function (azs, el) {
      const key = P.W + "x" + P.H + ":" + azs.join(",") + ":" + el;
      if (key === fitKey) return;
      fitKey = key;
      const keep = [P.az, P.el];
      P.S = 1; P.CX = 0; P.CY = 0; P.el = el;
      let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
      azs.forEach(a => {
        P.az = a * Math.PI / 180;
        for (const x of X) for (const y of Y) for (const z of Z) {
          const q = P.proj(x, y, z);
          x0 = Math.min(x0, q.x); x1 = Math.max(x1, q.x); y0 = Math.min(y0, q.y); y1 = Math.max(y1, q.y);
        }
      });
      P.S = Math.min((P.W - M.l - M.r) / (x1 - x0), (P.H - M.t - M.b) / (y1 - y0));
      P.CX = M.l - P.S * x0 + ((P.W - M.l - M.r) - P.S * (x1 - x0)) / 2;
      P.CY = M.t - P.S * y0;
      [P.az, P.el] = keep;
    };

    P.line = (a, b) => { ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); };
    P.poly = ps => { ctx.beginPath(); ps.forEach((q, i) => (i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y))); ctx.closePath(); };
    P.text = function (s, q, o) {
      o = o || {};
      ctx.save();
      ctx.font = `${o.size || 14}px ${FONT}`; ctx.fillStyle = o.color || MUTED;
      ctx.textAlign = o.align || "center"; ctx.textBaseline = o.base || "middle";
      ctx.translate(q.x, q.y); if (o.rot) ctx.rotate(o.rot);
      ctx.fillText(s, 0, 0); ctx.restore();
    };
    const readable = a => (a > Math.PI / 2 ? a - Math.PI : a < -Math.PI / 2 ? a + Math.PI : a);

    // the x and y walls farther from the camera (xb, yb) and the nearer edges (xf, yf)
    P.sides = function () {
      const ym = (Y[0] + Y[1]) / 2, xm = (X[0] + X[1]) / 2, zm = (Z[0] + Z[1]) / 2;
      const xb = P.proj(X[0], ym, zm).d > P.proj(X[1], ym, zm).d ? X[0] : X[1];
      const yb = P.proj(xm, Y[0], zm).d > P.proj(xm, Y[1], zm).d ? Y[0] : Y[1];
      return { xb, yb, xf: xb === X[0] ? X[1] : X[0], yf: yb === Y[0] ? Y[1] : Y[0] };
    };

    P.drawBox = function () {
      const { xb, yb } = P.sides();
      ctx.lineWidth = 1;
      const panes = [
        [[X[0], Y[0], Z[0]], [X[1], Y[0], Z[0]], [X[1], Y[1], Z[0]], [X[0], Y[1], Z[0]]],
        [[xb, Y[0], Z[0]], [xb, Y[1], Z[0]], [xb, Y[1], Z[1]], [xb, Y[0], Z[1]]],
        [[X[0], yb, Z[0]], [X[1], yb, Z[0]], [X[1], yb, Z[1]], [X[0], yb, Z[1]]],
      ];
      panes.forEach(pn => { P.poly(pn.map(q => P.proj(...q))); ctx.fillStyle = PANE; ctx.fill(); });
      ctx.strokeStyle = GRID;
      cfg.XT.forEach(t => { P.line(P.proj(t.v, Y[0], Z[0]), P.proj(t.v, Y[1], Z[0])); P.line(P.proj(t.v, yb, Z[0]), P.proj(t.v, yb, Z[1])); });
      cfg.YT.forEach(t => { P.line(P.proj(X[0], t.v, Z[0]), P.proj(X[1], t.v, Z[0])); P.line(P.proj(xb, t.v, Z[0]), P.proj(xb, t.v, Z[1])); });
      cfg.ZT.forEach(t => { P.line(P.proj(xb, Y[0], t.v), P.proj(xb, Y[1], t.v)); P.line(P.proj(X[0], yb, t.v), P.proj(X[1], yb, t.v)); });
      ctx.strokeStyle = EDGE;
      panes.forEach(pn => { P.poly(pn.map(q => P.proj(...q))); ctx.stroke(); });
    };

    // ticks and titles on the near floor edges, accuracy on the vertical edge furthest right on screen
    P.drawAxes = function () {
      const { xf, yf } = P.sides();
      const c = P.proj((X[0] + X[1]) / 2, (Y[0] + Y[1]) / 2, Z[0]);
      const ex = [P.proj(X[0], yf, Z[0]), P.proj(X[1], yf, Z[0])], ey = [P.proj(xf, Y[0], Z[0]), P.proj(xf, Y[1], Z[0])];
      const out = e => {
        let nx = -(e[1].y - e[0].y), ny = e[1].x - e[0].x; const l = Math.hypot(nx, ny) || 1; nx /= l; ny /= l;
        const m = { x: (e[0].x + e[1].x) / 2, y: (e[0].y + e[1].y) / 2 };
        if ((m.x - c.x) * nx + (m.y - c.y) * ny < 0) { nx = -nx; ny = -ny; }
        return [nx, ny];
      };
      const [ax, ay] = out(ex), [bx, by] = out(ey);
      cfg.XT.forEach(t => { const q = P.proj(t.v, yf, Z[0]); P.text(t.label, { x: q.x + ax * 18, y: q.y + ay * 18 }, { size: 15 }); });
      cfg.YT.forEach(t => { const q = P.proj(xf, t.v, Z[0]); P.text(t.label, { x: q.x + bx * 22, y: q.y + by * 22 }, { size: 15 }); });
      const mx = P.proj((X[0] + X[1]) / 2, yf, Z[0]), my = P.proj(xf, (Y[0] + Y[1]) / 2, Z[0]);
      P.text(cfg.xTitle, { x: mx.x + ax * 48, y: mx.y + ay * 48 },
             { size: 18, color: INK, rot: readable(Math.atan2(ex[1].y - ex[0].y, ex[1].x - ex[0].x)) });
      P.text(cfg.yTitle, { x: my.x + bx * 54, y: my.y + by * 54 },
             { size: 18, color: INK, rot: readable(Math.atan2(ey[1].y - ey[0].y, ey[1].x - ey[0].x)) });
      const corners = [[X[0], Y[0]], [X[1], Y[0]], [X[1], Y[1]], [X[0], Y[1]]];
      const [zx, zy] = corners.reduce((a, b) => (P.proj(...b, Z[0]).x > P.proj(...a, Z[0]).x ? b : a));
      cfg.ZT.forEach(t => { const q = P.proj(zx, zy, t.v); P.text(t.label, { x: q.x + 14, y: q.y }, { size: 15, align: "left" }); });
      const zm = P.proj(zx, zy, (Z[0] + Z[1]) / 2);
      P.text(cfg.zTitle, { x: zm.x + 60, y: zm.y }, { size: 18, color: INK, rot: -Math.PI / 2 });
    };

    return P;
  };

  // run draw(t_seconds) every frame while root's slide is shown; t restarts at 0 each time the slide is entered
  window.Plot3D.onSlide = function (root, draw) {
    let t0 = null;
    const section = root.closest("section");
    (function frame(now) {
      const shown = !section || section.classList.contains("present");
      if (shown) { if (t0 === null) t0 = now; draw((now - t0) / 1000); } else t0 = null;
      requestAnimationFrame(frame);
    })(performance.now());
  };
})();
