// Slide animations for "Synaptic budget" and "Rate budget" (slides.qmd). Plain SVG + requestAnimationFrame,
// no libraries. Each animation restarts when its slide is shown and pauses while it is hidden.
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const E = "#1F8AD0", I = "#E0393E", CUT = "#F28E2B", INK = "#1d1d1f", MUTED = "#8e8e93", LINE = "#d1d1d6";

  function el(parent, tag, attrs, text) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text !== undefined) n.textContent = text;
    parent.appendChild(n);
    return n;
  }
  const ease = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  const lerp = (a, b, t) => a + (b - a) * t;
  const sum = a => a.reduce((s, x) => s + x, 0);

  // run draw(t_seconds) while the element's slide is on screen; t restarts at 0 each time the slide is entered
  function onSlide(root, draw) {
    let t0 = null;
    const section = root.closest("section");
    (function frame(now) {
      const shown = !section || section.classList.contains("present");
      if (shown) {
        if (t0 === null) t0 = now;
        draw((now - t0) / 1000);
      } else t0 = null;
      requestAnimationFrame(frame);
    })(performance.now());
  }

  function setCaption(box, step, text) {
    const key = step + text;
    if (box.dataset.key === key) return;
    box.dataset.key = key;
    box.innerHTML = (step ? `<span class="anim-step">${step}</span>` : "") + text;
  }

  // ---------------------------------------------------------------- synaptic budget: subtractive normalisation
  // theta such that sum(max(w - theta, 0)) = budget (the L1 projection the network uses, Duchi et al. 2008)
  function theta(w, budget) {
    const u = w.filter(x => x > 0).sort((a, b) => b - a);
    let cs = 0, th = 0;
    for (let k = 0; k < u.length; k++) {
      cs += u[k];
      const t = (cs - budget) / (k + 1);
      if (u[k] - t > 0) th = t;
    }
    return Math.max(th, 0);
  }

  function synaptic(root) {
    const cap = root.querySelector(".anim-cap");
    const svg = el(root.querySelector(".anim-svg"), "svg", { viewBox: "0 0 640 420", width: "100%" });
    const W0 = [0.95, 0.42, 0.22, 0.80, 0.10, 0.62, 0.48, 0.88, 0.30, 0.55, 0.72, 0.18, 0.66, 0.36, 0.58, 0.26];
    const isI = W0.map((_, i) => i === 3 || i === 9 || i === 13);
    // one learning step per cycle: the synapses the task needs grow
    const GROW = [
      { 0: 0.30, 3: 0.32, 6: 0.38, 10: 0.36, 12: 0.30, 14: 0.30 },
      { 1: 0.34, 5: 0.32, 7: 0.30, 9: 0.30, 12: 0.28, 0: 0.20 },
      { 6: 0.30, 10: 0.32, 13: 0.32, 3: 0.30, 7: 0.26, 15: 0.24 },
    ];
    const B = sum(W0);
    const n = W0.length;

    // keyframes: {w, after, slice (0..1 opacity of the theta slice), gone (0..1 opacity of x marks), dur, hold, cap}
    const K = [];
    let w = W0.slice();
    K.push({ w: w.slice(), after: w.slice(), slice: 0, dur: 0.01, hold: 1.8, step: 1,
             cap: "At budget: total synaptic strength = budget" });
    GROW.forEach((g, c) => {
      const grown = w.map((x, i) => (x > 0 && g[i] ? x + g[i] : x));
      const th = theta(grown, B);
      const after = grown.map(x => Math.max(x - th, 0));
      K.push({ w: grown, after: grown, slice: 0, dur: 0.9, hold: 1.6, step: 2,
               cap: "Learning step: some synapses grow, the total goes over budget" });
      K.push({ w: grown, after, slice: 1, th, dur: 0.5, hold: 1.8, step: 3,
               cap: "Cut the same amount θ off every synapse (orange)" });
      K.push({ w: after, after, slice: 1, th, dur: 1.0, hold: 2.0, step: 4,
               cap: "Back at budget; synapses shorter than θ are gone (×)" });
      w = after;
    });
    const kept = w.filter(x => x > 0).length;
    K.push({ w: w.slice(), after: w.slice(), slice: 0, dur: 0.4, hold: 3.0, step: 0,
             cap: `After 3 learning steps: ${kept} of ${n} synapses left, total still at budget` });
    const total = K.reduce((s, k) => s + k.dur + k.hold, 0);

    // layout
    const x0 = 48, x1 = 470, yb = 360, H = 300, ymax = 1.45;
    const y = v => yb - (v / ymax) * H;
    const bw = (x1 - x0) / n;
    el(svg, "line", { x1: x0 - 6, x2: x1, y1: yb, y2: yb, stroke: INK, "stroke-width": 1.5 });
    el(svg, "text", { x: (x0 + x1) / 2, y: yb + 26, "text-anchor": "middle", "font-size": 15, fill: MUTED },
       "recurrent synapses (blue from E, red from I)");
    el(svg, "text", { x: 14, y: yb - H / 2, "text-anchor": "middle", "font-size": 15, fill: MUTED,
                      transform: `rotate(-90 14 ${yb - H / 2})` }, "strength |W| / c");
    const bars = [], slices = [], marks = [];
    for (let i = 0; i < n; i++) {
      const bx = x0 + i * bw + bw * 0.16;
      bars.push(el(svg, "rect", { x: bx, width: bw * 0.68, rx: 2, fill: isI[i] ? I : E }));
      slices.push(el(svg, "rect", { x: bx, width: bw * 0.68, rx: 2, fill: CUT }));
      marks.push(el(svg, "text", { x: bx + bw * 0.34, y: yb - 6, "text-anchor": "middle", "font-size": 18,
                                   fill: MUTED, opacity: 0 }, "×"));
    }
    // theta bracket next to the first synapse
    const brk = el(svg, "g", {});
    const brkPath = el(brk, "path", { fill: "none", stroke: CUT, "stroke-width": 2 });
    const brkTxt = el(brk, "text", { "font-size": 18, fill: CUT, "text-anchor": "end" }, "θ");

    // total gauge
    const gx = 516, gw = 46;
    el(svg, "text", { x: gx + gw / 2, y: yb + 26, "text-anchor": "middle", "font-size": 15, fill: MUTED }, "total");
    const gScale = v => yb - (v / B) * (H * 0.62);
    const gBody = el(svg, "rect", { x: gx, width: gw, rx: 3, fill: "#9aa3ad" });
    const gOver = el(svg, "rect", { x: gx, width: gw, rx: 3, fill: CUT });
    el(svg, "line", { x1: gx - 8, x2: gx + gw + 6, y1: gScale(B), y2: gScale(B), stroke: INK,
                      "stroke-width": 2, "stroke-dasharray": "6 4" });
    el(svg, "text", { x: gx + gw + 10, y: gScale(B) + 5, "font-size": 15, fill: INK }, "budget");
    el(svg, "line", { x1: gx - 4, x2: gx + gw + 4, y1: yb, y2: yb, stroke: INK, "stroke-width": 1.5 });
    const keptTxt = el(svg, "text", { x: x1, y: 22, "text-anchor": "end", "font-size": 15, fill: MUTED });

    onSlide(root, t => {
      t = t % total;
      let k = 0;
      while (t > K[k].dur + K[k].hold) { t -= K[k].dur + K[k].hold; k++; }
      const cur = K[k], prev = K[Math.max(k - 1, 0)];
      const a = ease(Math.min(t / cur.dur, 1));
      // bar height: tween prev.w -> cur.w; the orange slice spans [after, height]
      const h = cur.w.map((v, i) => lerp(prev.w[i], v, a));
      const sliceOp = lerp(prev.slice, cur.slice, a);
      for (let i = 0; i < n; i++) {
        const lo = Math.min(cur.after[i], h[i]);
        bars[i].setAttribute("y", y(lo)); bars[i].setAttribute("height", Math.max(yb - y(lo), 0));
        slices[i].setAttribute("y", y(h[i])); slices[i].setAttribute("height", Math.max(y(lo) - y(h[i]), 0));
        slices[i].setAttribute("opacity", h[i] > lo + 1e-6 ? sliceOp : 0);
        const gone = h[i] <= 1e-6 && W0[i] > 0;
        marks[i].setAttribute("opacity", gone ? 1 : 0);
      }
      // bracket on the tallest live synapse while the slice is shown
      if (cur.th && sliceOp > 0.05 && h[0] > cur.after[0] + 1e-3) {
        const bx = x0 + bw * 0.16 - 6;
        const yt = y(h[0]), yl = y(Math.min(cur.after[0], h[0]));
        brkPath.setAttribute("d", `M ${bx + 4} ${yt} H ${bx} V ${yl} H ${bx + 4}`);
        brkTxt.setAttribute("x", bx - 4); brkTxt.setAttribute("y", (yt + yl) / 2 + 6);
        brk.setAttribute("opacity", sliceOp);
      } else brk.setAttribute("opacity", 0);
      // gauge: grey up to the budget, orange above it
      const tot = sum(h);
      const top = gScale(tot), bud = gScale(Math.min(tot, B));
      gBody.setAttribute("y", bud); gBody.setAttribute("height", yb - bud);
      gOver.setAttribute("y", top); gOver.setAttribute("height", Math.max(bud - top, 0));
      keptTxt.textContent = `synapses kept: ${h.filter(v => v > 1e-6).length} / ${n}`;
      setCaption(cap, cur.step, cur.cap);
    });
  }

  // ---------------------------------------------------------------- rate budget: learned weight + rho term
  // Toy model: one rate cost r (in units of the budget), task loss a (r - 3)^2 (the task alone wants 3x the budget),
  // gradient descent on r; lambda follows the training rule (BudgetConstraint: eta 0.01, EMA 0.9, clip, lambda0 0.01).
  function toy(rho, T) {
    const a = 0.15, rTask = 3, lr = 0.05, eta = 0.01;
    let r = rTask, integ = Math.log(0.01), vema = null;
    const R = [], L = [];
    for (let t = 0; t < T; t++) {
      const lam = Math.exp(integ), v = r - 1;
      r = Math.max(r - lr * (2 * a * (r - rTask) + lam + rho * Math.max(v, 0)), 0);
      vema = vema === null ? r - 1 : 0.9 * vema + 0.1 * (r - 1);
      integ = Math.min(Math.max(integ + eta * Math.max(-1, Math.min(1, vema)), Math.log(1e-6)), Math.log(1e4));
      R.push(r); L.push(Math.exp(integ));
    }
    return { R, L };
  }

  function rate(root) {
    const cap = root.querySelector(".anim-cap");
    const svg = el(root.querySelector(".anim-svg"), "svg", { viewBox: "0 0 640 420", width: "100%" });
    const T = 2000, DRAW = 11, HOLD = 4;
    const on = toy(1, T), off = toy(0, T);

    const xa = 70, xb = 620;
    const X = t => xa + (t / T) * (xb - xa);
    // top panel: rate cost / budget
    const ta = 18, tb = 222, rmax = 3.3;
    const Yr = v => tb - (v / rmax) * (tb - ta);
    // bottom panel: lambda, log axis
    const ba = 262, bb = 372, lmin = Math.log10(0.005), lmax = Math.log10(2);
    const Yl = v => bb - ((Math.log10(v) - lmin) / (lmax - lmin)) * (bb - ba);

    el(svg, "rect", { x: xa, y: ta, width: xb - xa, height: Yr(1) - ta, fill: CUT, opacity: 0.07 });
    el(svg, "text", { x: xb - 6, y: ta + 18, "text-anchor": "end", "font-size": 14, fill: CUT }, "over budget");
    el(svg, "line", { x1: xa, x2: xb, y1: Yr(1), y2: Yr(1), stroke: INK, "stroke-width": 2,
                      "stroke-dasharray": "6 4" });
    el(svg, "text", { x: xb - 6, y: Yr(1) + 18, "text-anchor": "end", "font-size": 14, fill: INK }, "budget");
    [0, 1, 2, 3].forEach(v => {
      el(svg, "text", { x: xa - 8, y: Yr(v) + 5, "text-anchor": "end", "font-size": 13, fill: MUTED },
         v === 1 ? "1×" : v + "×");
    });
    el(svg, "line", { x1: xa, x2: xa, y1: ta, y2: tb, stroke: LINE });
    el(svg, "line", { x1: xa, x2: xb, y1: tb, y2: tb, stroke: LINE });
    el(svg, "text", { x: 16, y: (ta + tb) / 2, "text-anchor": "middle", "font-size": 15, fill: MUTED,
                      transform: `rotate(-90 16 ${(ta + tb) / 2})` }, "rate cost / budget");

    [0.01, 0.1, 1].forEach(v => {
      el(svg, "line", { x1: xa, x2: xb, y1: Yl(v), y2: Yl(v), stroke: "#f0f0f3" });
      el(svg, "text", { x: xa - 8, y: Yl(v) + 5, "text-anchor": "end", "font-size": 13, fill: MUTED }, String(v));
    });
    el(svg, "line", { x1: xa, x2: xa, y1: ba, y2: bb, stroke: LINE });
    el(svg, "line", { x1: xa, x2: xb, y1: bb, y2: bb, stroke: LINE });
    el(svg, "text", { x: 16, y: (ba + bb) / 2, "text-anchor": "middle", "font-size": 15, fill: MUTED,
                      transform: `rotate(-90 16 ${(ba + bb) / 2})` }, "weight λ");
    el(svg, "text", { x: (xa + xb) / 2, y: bb + 30, "text-anchor": "middle", "font-size": 15, fill: MUTED },
       "training step (toy model)");

    const style = (col, dash, w) => ({ fill: "none", stroke: col, "stroke-width": w, "stroke-dasharray": dash,
                                       "stroke-linejoin": "round" });
    const rOff = el(svg, "path", style("#a1a1a6", "5 4", 2));
    const lOff = el(svg, "path", style("#a1a1a6", "5 4", 2));
    const rOn = el(svg, "path", style(INK, "", 3));
    const lOn = el(svg, "path", style(I, "", 3));
    const dR = el(svg, "circle", { r: 6, fill: INK });
    const dL = el(svg, "circle", { r: 6, fill: I });
    // legend
    const lg = el(svg, "g", { transform: `translate(${xa + 300}, ${ta + 46})` });
    el(lg, "line", { x1: 0, x2: 28, y1: 0, y2: 0, stroke: INK, "stroke-width": 3 });
    el(lg, "text", { x: 36, y: 5, "font-size": 14, fill: INK }, "with ρ term (as trained)");
    el(lg, "line", { x1: 0, x2: 28, y1: 22, y2: 22, stroke: "#a1a1a6", "stroke-width": 2, "stroke-dasharray": "5 4" });
    el(lg, "text", { x: 36, y: 27, "font-size": 14, fill: MUTED }, "λ only (ρ = 0)");
    // note at the dip of the rho = 0 trace
    const dip = off.R.indexOf(Math.min(...off.R));
    const dipNote = el(svg, "text", { x: X(dip) - 40, y: Yr(off.R[dip]) + 22, "font-size": 14, fill: MUTED,
                                      opacity: 0 }, "λ only: slow start, then λ overshoots and the rate dips below budget");

    const dropNote = el(svg, "text", { x: X(70), y: Yr(on.R[60]) - 14, "font-size": 14, fill: INK, opacity: 0 },
                        "ρ term: immediate push-back");
    const path = (A, Y, n) => {
      let d = "";
      for (let t = 0; t < n; t += 4) d += (t ? "L" : "M") + X(t).toFixed(1) + " " + Y(A[t]).toFixed(1);
      return d;
    };
    const settled = on.R.findIndex((v, t) => t > 50 && on.R.slice(t).every(x => Math.abs(x - 1) < 0.03));

    onSlide(root, s => {
      s = s % (DRAW + HOLD);
      const n = Math.max(2, Math.min(T, Math.round((s / DRAW) * T)));
      rOn.setAttribute("d", path(on.R, Yr, n)); lOn.setAttribute("d", path(on.L, Yl, n));
      rOff.setAttribute("d", path(off.R, Yr, n)); lOff.setAttribute("d", path(off.L, Yl, n));
      const t = n - 1;
      dR.setAttribute("cx", X(t)); dR.setAttribute("cy", Yr(on.R[t]));
      dL.setAttribute("cx", X(t)); dL.setAttribute("cy", Yl(on.L[t]));
      dipNote.setAttribute("opacity", t > dip + 60 ? 1 : 0);
      dropNote.setAttribute("opacity", t > 40 ? 1 : 0);
      if (t < 25) setCaption(cap, 1, "Start: the task alone would run the rate at 3× the budget");
      else if (t < 250) setCaption(cap, 2, "ρ term: pushes back at once, in proportion to the overshoot");
      else if (t < settled) setCaption(cap, 3, "λ keeps rising while the rate is above budget, and pulls it down");
      else setCaption(cap, 4, "On budget: the ρ term is now zero, λ stops changing and holds the rate there");
    });
  }

  function init() {
    document.querySelectorAll(".anim-syn").forEach(synaptic);
    document.querySelectorAll(".anim-rate").forEach(rate);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
