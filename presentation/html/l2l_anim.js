// Like-to-like connectivity, one click per step (slide "Like-to-like connectivity"), on one real network
// (assets/l2l_data.js, build_figs.py l2l). SVG + requestAnimationFrame, no libraries. The step is the number of
// visible reveal.js fragments ".l2l-frag" on the slide (three empty spans in slides.qmd), so arrow keys / clicks
// step forward and back.
//   0    slide shown: empty axes
//   1    connected pair (synapse, the two profiles, r) + the histogram of r over all connected E->E pairs
//   2    unconnected pair + the histogram of all unconnected E->E pairs
//   3    a threshold on r sweeps from +1 to -1, the fractions above it trace the ROC; then area = AUC
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const E = "#1F8AD0", NEUTRAL = "#48484a", INK = "#1d1d1f", MUTED = "#6e6e73", GREY = "#b8b8bd", LINE = "#d1d1d6";
  const FONT = 'Inter, "Segoe UI", -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif';
  const SWEEP = 8.5, AREA = 9;   // step 3: sweep duration, start of the AUC area (s)

  const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
  const ease = t => { t = clamp(t, 0, 1); return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; };
  const lerp = (a, b, t) => a + (b - a) * t;
  function el(parent, tag, attrs, text) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text !== undefined) n.textContent = text;
    parent.appendChild(n);
    return n;
  }
  const txt = (p, x, y, s, o) => el(p, "text", Object.assign({ x, y, "font-size": 15, fill: MUTED, "font-family": FONT }, o), s);

  function setup(root) {
    const D = window.L2L_DATA;
    if (!D) return;
    const cap = root.querySelector(".anim-cap");
    const svg = el(root.querySelector(".anim-svg"), "svg", { viewBox: "0 0 1240 380", width: "100%" });
    const nT = D.pairs[0].a.length;

    // ---------------------------------------------------------------- panel A: two pairs side by side
    const A = el(svg, "g", {});
    txt(A, 232, 20, "two pairs of neurons", { "text-anchor": "middle", fill: INK, "font-size": 17 });
    txt(A, 26, 230, "normalized task variance", { "text-anchor": "middle", "font-size": 13,
                                                  transform: "rotate(-90 26 230)" });
    const COLS = [{ x0: 50, names: ["a", "b"] }, { x0: 250, names: ["c", "d"] }], CW = 170;
    const pairs = COLS.map((col, k) => {
      const p = D.pairs[k], g = el(A, "g", { opacity: 0 }), x0 = col.x0, x1 = x0 + CW;
      const tri = (x, c, name) => {
        el(g, "path", { d: `M ${x} 46 L ${x + 16} 74 L ${x - 16} 74 Z`, fill: "white", stroke: INK, "stroke-width": 1.4 });
        txt(g, x, 69, name, { "text-anchor": "middle", fill: INK, "font-size": 13 });
      };
      tri(x0 + 18, null, col.names[0]); tri(x1 - 18, null, col.names[1]);
      const link = el(g, "g", {});
      if (p.conn) {
        el(link, "line", { x1: x0 + 40, x2: x1 - 44, y1: 64, y2: 64, stroke: E, "stroke-width": 2.6 });
        el(link, "path", { d: `M ${x1 - 36} 64 L ${x1 - 48} 58 L ${x1 - 48} 70 Z`, fill: E });
      } else {
        el(link, "line", { x1: x0 + 40, x2: x1 - 38, y1: 64, y2: 64, stroke: GREY, "stroke-width": 1.5,
                           "stroke-dasharray": "4 4" });
      }
      txt(link, x0 + CW / 2, 52, p.conn ? "synapse" : "no synapse", { "text-anchor": "middle", "font-size": 13,
                                                                         fill: p.conn ? E : MUTED });
      const r = txt(g, x0 + CW / 2, 104, `${p.conn ? "connected" : "unconnected"}, r = ${p.r >= 0 ? "+" : ""}${p.r.toFixed(2)}`,
                    { "text-anchor": "middle", fill: p.conn ? E : INK, "font-size": 15 });
      const prof = [[NEUTRAL, col.names[0], 130], [NEUTRAL, col.names[1], 238]].map(([c, name, y0]) => {
        const H = 86, w = CW / nT;
        el(g, "line", { x1: x0, x2: x1, y1: y0 + H, y2: y0 + H, stroke: INK, "stroke-width": 1 });
        txt(g, x0 + 1, y0 - 3, `neuron ${name}`, { "font-size": 12 });
        const bars = [];
        for (let i = 0; i < nT; i++)
          bars.push(el(g, "rect", { x: x0 + i * w + w * 0.12, width: w * 0.76, fill: c, stroke: c,
                                    "stroke-width": 0.4 }));
        return { bars, y0, H };
      });
      txt(g, x0 + CW / 2, 342, "20 tasks", { "text-anchor": "middle", "font-size": 13 });
      return { g, link, r, prof, p };
    });

    // ---------------------------------------------------------------- panel B: all pairs, histograms, threshold
    const B = el(svg, "g", {});
    const bx0 = 470, bx1 = 830, by0 = 92, by1 = 300;
    const X = r => bx0 + (r + 1) / 2 * (bx1 - bx0);
    const hmax = Math.max(...D.hc, ...D.hu) * 1.08;
    const Yh = v => by1 - v / hmax * (by1 - by0);
    txt(B, (bx0 + bx1) / 2, 20, "all E→E pairs of this network", { "text-anchor": "middle", fill: INK, "font-size": 17 });
    const shade = el(B, "rect", { y: by0 - 10, height: by1 - by0 + 10, fill: E, opacity: 0 });
    const huBars = D.hu.map((v, k) => el(B, "rect", { x: X(D.bins[k]) + 0.5, width: X(D.bins[k + 1]) - X(D.bins[k]) - 1,
                                                       fill: GREY }));
    const hcPath = el(B, "path", { fill: "none", stroke: E, "stroke-width": 2.6, "stroke-linejoin": "round" });
    el(B, "line", { x1: bx0, x2: bx1, y1: by1, y2: by1, stroke: INK, "stroke-width": 1 });
    [-1, -0.5, 0, 0.5, 1].forEach(v => {
      el(B, "line", { x1: X(v), x2: X(v), y1: by1, y2: by1 + 5, stroke: INK });
      txt(B, X(v), by1 + 22, String(v), { "text-anchor": "middle", "font-size": 13 });
    });
    txt(B, (bx0 + bx1) / 2, by1 + 46, "similarity r of the two profiles", { "text-anchor": "middle", "font-size": 14 });
    txt(B, bx0 - 12, (by0 + by1) / 2, "fraction of pairs", { "text-anchor": "middle", "font-size": 13,
                                                              transform: `rotate(-90 ${bx0 - 12} ${(by0 + by1) / 2})` });
    const legC = el(B, "g", { opacity: 0 }), legU = el(B, "g", { opacity: 0 });
    el(legC, "line", { x1: bx0, x2: bx0 + 14, y1: 42, y2: 42, stroke: E, "stroke-width": 2.6 });
    txt(legC, bx0 + 20, 47, `connected (${D.n_conn})`, { "font-size": 13, fill: E });
    el(legU, "rect", { x: bx0, y: 57, width: 14, height: 10, fill: GREY });
    txt(legU, bx0 + 20, 67, `unconnected (${D.n_unconn.toLocaleString("en")})`, { "font-size": 13 });
    // example markers on the r axis
    const marks = D.pairs.map(p => {
      const g = el(B, "g", { opacity: 0 });
      el(g, "path", { d: `M ${X(p.r)} ${by1 - 2} l -7 -12 l 14 0 Z`, fill: p.conn ? E : "#8e8e93" });
      return g;
    });
    const thr = el(B, "g", { opacity: 0 });
    const thrLine = el(thr, "line", { y1: by0 - 6, y2: by1, stroke: INK, "stroke-width": 2 });
    const fracC = txt(B, bx1, 47, "", { "text-anchor": "end", fill: E, "font-size": 14 });
    const fracU = txt(B, bx1, 67, "", { "text-anchor": "end", "font-size": 14 });

    // ---------------------------------------------------------------- panel C: ROC
    const C = el(svg, "g", {});
    const cx0 = 940, cs = 240, cy1 = 300, cy0 = cy1 - cs;
    const RX = f => cx0 + f * cs, RY = t => cy1 - t * cs;
    txt(C, cx0 + cs / 2, 20, "ROC", { "text-anchor": "middle", fill: INK, "font-size": 17 });
    el(C, "rect", { x: cx0, y: cy0, width: cs, height: cs, fill: "none", stroke: INK, "stroke-width": 1 });
    el(C, "line", { x1: RX(0), y1: RY(0), x2: RX(1), y2: RY(1), stroke: GREY, "stroke-width": 1.2, "stroke-dasharray": "4 4" });
    txt(C, RX(0.78), RY(0.66), "chance", { "text-anchor": "middle", "font-size": 12, transform: `rotate(-45 ${RX(0.78)} ${RY(0.66)})` });
    const area = el(C, "path", { fill: E, opacity: 0 });
    const rocPath = el(C, "path", { fill: "none", stroke: E, "stroke-width": 2.6, "stroke-linejoin": "round" });
    const rocDot = el(C, "circle", { r: 6, fill: E, stroke: "white", "stroke-width": 1.5, opacity: 0 });
    const aucTxt = txt(C, RX(0.62), RY(0.25), "", { "text-anchor": "middle", fill: INK, "font-size": 18 });
    [[0, "0"], [1, "1"]].forEach(([v, s]) => {
      txt(C, RX(v), cy1 + 20, s, { "text-anchor": "middle", "font-size": 13 });
      txt(C, cx0 - 10, RY(v) + 5, s, { "text-anchor": "end", "font-size": 13 });
    });
    txt(C, cx0 + cs / 2, cy1 + 40, "unconnected pairs above threshold", { "text-anchor": "middle", "font-size": 13 });
    txt(C, cx0 - 26, cy0 + cs / 2, "connected pairs above threshold", { "text-anchor": "middle", "font-size": 13,
                                                                     fill: E, transform: `rotate(-90 ${cx0 - 26} ${cy0 + cs / 2})` });
    const sgn = v => (v >= 0 ? "+" : "") + v.toFixed(2);
    const CAPS = {
      0: "Each neuron has a profile: its normalized task variance in each of the 20 tasks.",
      1: `A connected pair with similar profiles: r = ${sgn(D.pairs[0].r)}. Blue: r of every connected E→E pair.`,
      2: `An unconnected pair with different profiles: r = ${sgn(D.pairs[1].r)}. Grey: r of every unconnected E→E pair, further left.`,
      3: "Slide a threshold on r: the fractions of connected and of unconnected pairs above it trace the ROC.",
      4: `Area under the ROC: AUC = ${D.auc.toFixed(2)}, the chance that a connected pair is more similar than an unconnected one (0.5: no relation).`,
    };

    function showPair(k, a) { // a: 0..1 build-up of pair k
      const P = pairs[k], p = P.p;
      P.g.setAttribute("opacity", a > 0 ? 1 : 0);
      P.prof.forEach((pf, m) => {
        const v = m === 0 ? p.a : p.b;
        pf.bars.forEach((b, i) => {
          const h = v[i] * pf.H * ease(a * 1.6 - i / nT * 0.6);
          b.setAttribute("y", pf.y0 + pf.H - h); b.setAttribute("height", Math.max(h, 0));
        });
      });
      P.link.setAttribute("opacity", ease(a * 3));
      P.r.setAttribute("opacity", ease(a * 2 - 0.9));
    }

    function draw(step, t) { // t: seconds since this step began (large = finished)
      const c = step === 3 && t >= AREA ? 4 : step;
      if (cap.dataset.s !== String(c)) {
        cap.dataset.s = String(c);
        cap.innerHTML = (step ? `<span class="anim-step">${step}</span>` : "") + CAPS[c];
      }
      // panel A: the two pairs
      showPair(0, step < 1 ? 0 : step === 1 ? t / 3 : 1);
      showPair(1, step < 2 ? 0 : step === 2 ? t / 3 : 1);
      // panel B: connected curve with step 1, unconnected bars with step 2
      const gc = step < 1 ? 0 : step === 1 ? ease((t - 1.2) / 2) : 1;
      const gu = step < 2 ? 0 : step === 2 ? ease((t - 1.2) / 2) : 1;
      D.hu.forEach((v, k) => { const y = Yh(v * gu); huBars[k].setAttribute("y", y); huBars[k].setAttribute("height", by1 - y); });
      let d = `M ${X(D.bins[0])} ${by1}`;
      D.hc.forEach((v, k) => { const y = Yh(v * gc); d += ` L ${X(D.bins[k])} ${y} L ${X(D.bins[k + 1])} ${y}`; });
      hcPath.setAttribute("d", d + ` L ${X(1)} ${by1}`);
      hcPath.setAttribute("opacity", gc > 0 ? 1 : 0);
      legC.setAttribute("opacity", gc); legU.setAttribute("opacity", gu);
      marks[0].setAttribute("opacity", gc); marks[1].setAttribute("opacity", gu);
      // step 3: threshold sweep -> ROC, then the area
      if (step === 3) {
        const n = Math.round(clamp(t / SWEEP, 0, 1) * (D.th.length - 1)), x = X(D.th[n]);
        const sweeping = t < AREA;
        thr.setAttribute("opacity", sweeping ? 1 : 1 - ease(t - AREA));
        thrLine.setAttribute("x1", x); thrLine.setAttribute("x2", x);
        shade.setAttribute("x", x); shade.setAttribute("width", bx1 - x); shade.setAttribute("opacity", sweeping ? 0.07 : 0);
        fracC.textContent = sweeping ? `connected above: ${Math.round(100 * D.tpr[n])}%` : "";
        fracU.textContent = sweeping ? `unconnected above: ${Math.round(100 * D.fpr[n])}%` : "";
        let pd = "";
        for (let k = 0; k <= n; k++) pd += (k ? "L" : "M") + RX(D.fpr[k]).toFixed(1) + " " + RY(D.tpr[k]).toFixed(1);
        rocPath.setAttribute("d", pd);
        rocDot.setAttribute("cx", RX(D.fpr[n])); rocDot.setAttribute("cy", RY(D.tpr[n]));
        rocDot.setAttribute("opacity", sweeping ? 1 : 0);
        let ad = `M ${RX(0)} ${RY(0)}`;
        D.fpr.forEach((f, k) => { ad += ` L ${RX(f).toFixed(1)} ${RY(D.tpr[k]).toFixed(1)}`; });
        area.setAttribute("d", ad + ` L ${RX(1)} ${RY(0)} Z`);
        area.setAttribute("opacity", 0.2 * ease((t - AREA) / 1.2));
        aucTxt.textContent = `AUC = ${D.auc.toFixed(2)}`;
        aucTxt.setAttribute("opacity", ease((t - AREA) / 1.2 - 0.3));
      } else {
        thr.setAttribute("opacity", 0); shade.setAttribute("opacity", 0);
        fracC.textContent = ""; fracU.textContent = "";
        rocPath.setAttribute("d", ""); rocDot.setAttribute("opacity", 0);
        area.setAttribute("opacity", 0); aucTxt.setAttribute("opacity", 0);
      }
    }

    // step = visible fragments; a new step animates from its start; going back (or arriving from a later slide)
    // shows it finished
    const section = root.closest("section");
    let cur = -1, t0 = 0, done = false;
    (function frame(now) {
      const shown = !section || section.classList.contains("present");
      if (shown) {
        const step = section ? Math.min(section.querySelectorAll(".l2l-frag.visible").length, 3) : 3;
        if (step !== cur) { done = cur === -1 ? step > 0 : step < cur; cur = step; t0 = now; }
        draw(step, done ? 1e3 : (now - t0) / 1000);
      } else cur = -1;
      requestAnimationFrame(frame);
    })(performance.now());
  }

  function init() { document.querySelectorAll(".anim-l2l").forEach(setup); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
