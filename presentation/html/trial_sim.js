// Task simulator (slide "Setup"), laid out like figs/fig_setup.pdf: pick one of the 20 Yang tasks and one of the 4
// networks and play one real trial. Data: assets/trial_data.js (build_figs.py trials): one noise-free trial per task,
// the outputs and the rates of 16 E and 4 I neurons of each point's seed-0 network, its accuracy and costs.
// Plain SVG + requestAnimationFrame, no libraries.
//   left    inputs over time (fixation, stimulus modality 1 and 2, rule), then the 53 input neurons (1 + 16 + 16 + 20)
//   middle  the network: E triangles, I circles, transparent at rest, filled blue / red with their rate (each neuron
//           scaled to its own max over the 20 trials); lines = synapses among the 20 shown (|W|/c > 0.01)
//   right   the 17 output neurons (fixation + response ring; target = open triangle, read-out = arrow), outputs over
//           time, and this network's accuracy on the task, synaptic cost and rate cost
//   bottom  the trial's periods: the slider (click or drag to scrub)
// Plays only after "play", until "pause" or the end of the trial. Time panels: played part in colour, the rest faint.
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const E = "#1F8AD0", I = "#E0393E", INK = "#1d1d1f", MUTED = "#6e6e73", GREY = "#b8b8bd", LINE = "#e5e5ea";
  const ACT = [76, 47, 216];                       // input / output activity colour (purple, as in fig_setup)
  const FONT = 'Inter, "Segoe UI", -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif';
  const PCOL = { control: "#4a4a4a", firing: "#d62728", wiring: "#1f77b4", both: "#2ca02c" };
  const GROUPS = [["Go", "RT Go", "Dly Go", "Anti", "RT Anti", "Dly Anti"],
                  ["DM 1", "DM 2", "Ctx DM 1", "Ctx DM 2", "MultSen DM", "Dly DM 1", "Dly DM 2", "Ctx Dly DM 1",
                   "Ctx Dly DM 2", "MultSen Dly DM"],
                  ["DMS", "DNMS", "DMC", "DNMC"]];
  const EPOCH = { fix1: "fixation", stim1: "stimulus 1", delay1: "delay", stim2: "stimulus 2", delay2: "delay 2",
                  go1: "response" };
  const MS_PER_STEP = 40;                          // half real time (dt = 20 ms)
  const NR_IN = 5.5;                               // radius of every input / output neuron

  function el(parent, tag, attrs, text) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text !== undefined) n.textContent = text;
    parent.appendChild(n);
    return n;
  }
  const txt = (p, x, y, s, o) => el(p, "text", Object.assign({ x, y, "font-size": 13, fill: MUTED, "font-family": FONT }, o), s);
  const h = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text) n.textContent = text; return n; };
  const col = v => { v = Math.min(Math.max(v, 0), 1); return `rgb(${ACT.map(c => Math.round(255 + (c - 255) * v)).join(",")})`; };
  const fmt = v => (v < 0.01 ? v.toExponential(1).replace("e-", "e−") : v.toFixed(v < 0.1 ? 3 : 2));

  function setup(root) {
    const D = window.TRIAL_DATA;
    if (!D) return;
    const byName = Object.fromEntries(D.tasks.map((t, k) => [t.name, k]));
    let task = byName["Dly Go"] ?? 0, point = "control", t = 0, playing = false, last = null;
    const tdim = () => D.tasks[task].tdim;

    // ---------------------------------------------------------------- controls (HTML)
    const ctl = h("div", "sim-ctl");
    const taskRow = h("div", "sim-tasks");
    const taskBtn = {};
    GROUPS.forEach(g => {
      const grp = h("div", "sim-group");
      g.forEach(name => {
        const b = h("button", "sim-btn", name);
        b.onclick = () => { task = byName[name]; t = 0; rebuildTask(); refresh(); };
        taskBtn[name] = b; grp.appendChild(b);
      });
      taskRow.appendChild(grp);
    });
    const row2 = h("div", "sim-row2");
    const ptRow = h("div", "sim-points");
    const ptBtn = {};
    D.points.forEach(p => {
      const b = h("button", "sim-btn sim-pt", D.nets[p].label);
      b.style.setProperty("--pc", PCOL[p]);
      b.onclick = () => { point = p; rebuildNet(); rebuildTask(); refresh(); };
      ptBtn[p] = b; ptRow.appendChild(b);
    });
    const play = h("button", "sim-btn sim-play");
    play.onclick = () => { if (!playing && t >= tdim() - 1) t = 0; playing = !playing; refresh(); };
    const back = h("button", "sim-btn sim-step", "◀");
    back.onclick = () => { playing = false; t = Math.max(0, Math.floor(t) - 1); refresh(); };
    const fwd = h("button", "sim-btn sim-step", "▶");
    fwd.onclick = () => { playing = false; t = Math.min(tdim() - 1, Math.floor(t) + 1); refresh(); };
    const clock = h("span", "sim-clock");
    const transport = h("div", "sim-transport");
    [play, back, fwd, clock].forEach(n => transport.appendChild(n));
    row2.appendChild(ptRow); row2.appendChild(transport);
    ctl.appendChild(taskRow); ctl.appendChild(row2);
    root.appendChild(ctl);

    const svg = el(root, "svg", { viewBox: "0 0 1240 414", width: "100%", class: "sim-svg" });
    const defs = el(svg, "defs", {});

    // ---------------------------------------------------------------- time panels (left: inputs, right: outputs)
    const PX0 = 72, PX1 = 300, QX0 = 960, QX1 = 1188;
    const uid = Math.random().toString(36).slice(2, 7);
    let clipN = 0, dragFrom = null;
    function panel(x0, x1, y0, y1, title, yl) {
      const g = el(svg, "g", {});
      txt(g, (x0 + x1) / 2, y0 - 6, title, { "text-anchor": "middle", fill: INK, "font-size": 14 });
      if (yl) {
        txt(g, x0 - 6, y0 + 9, yl[0], { "text-anchor": "end", "font-size": 11.5 });
        txt(g, x0 - 6, y1, yl[1], { "text-anchor": "end", "font-size": 11.5 });
      }
      const faint = el(g, "g", { opacity: 0.22 }), id = `simclip-${uid}-${++clipN}`;
      const clip = el(el(defs, "clipPath", { id }), "rect", { x: x0, y: y0 - 2, height: y1 - y0 + 4, width: 0 });
      const full = el(g, "g", { "clip-path": `url(#${id})` });
      el(g, "line", { x1: x0, x2: x1, y1, y2: y1, stroke: INK, "stroke-width": 1 });
      el(g, "line", { x1: x0, x2: x0, y1: y0, y2: y1, stroke: INK, "stroke-width": 1 });
      const head = el(g, "line", { y1: y0 - 2, y2: y1, stroke: INK, "stroke-width": 1.2, opacity: 0.55 });
      const hit = el(g, "rect", { x: x0, y: y0, width: x1 - x0, height: y1 - y0, fill: "transparent", style: "cursor: pointer" });
      hit.addEventListener("pointerdown", ev => { dragFrom = [x0, x1]; seek(ev); ev.stopPropagation(); });
      return { x0, x1, y0, y1, faint, full, clip, head };
    }
    const P = {
      fixIn: panel(PX0, PX1, 20, 48, "Fixation input"),
      mod1: panel(PX0, PX1, 72, 130, "Stimulus mod 1", ["360°", "0°"]),
      mod2: panel(PX0, PX1, 154, 212, "Stimulus mod 2", ["360°", "0°"]),
      rule: panel(PX0, PX1, 236, 300, "Rule inputs", ["20", "1"]),
      fixOut: panel(QX0, QX1, 20, 48, "Fixation output"),
      resp: panel(QX0, QX1, 78, 146, "Response", ["360°", "0°"]),
    };
    function heat(pn, rows) { // rows: T x n values 0-100 -> an image in both layers
      const T = rows.length, n = rows[0].length, c = document.createElement("canvas");
      c.width = T; c.height = n;
      const cx = c.getContext("2d"), img = cx.createImageData(T, n);
      for (let s = 0; s < T; s++) for (let k = 0; k < n; k++) {
        const v = Math.min(rows[s][k] / 100, 1), o = ((n - 1 - k) * T + s) * 4;   // row 0 at the bottom (0°)
        ACT.forEach((a, j) => { img.data[o + j] = Math.round(255 + (a - 255) * v); });
        img.data[o + 3] = 255;
      }
      cx.putImageData(img, 0, 0);
      const url = c.toDataURL();
      [pn.faint, pn.full].forEach(g => {
        g.textContent = "";
        el(g, "image", { href: url, x: pn.x0 + 1, y: pn.y0, width: pn.x1 - pn.x0 - 1, height: pn.y1 - pn.y0 - 1,
                         preserveAspectRatio: "none", style: "image-rendering: pixelated" });
      });
    }
    function trace(pn, vals, target) {
      const T = vals.length, X = s => pn.x0 + 1 + s / Math.max(T - 1, 1) * (pn.x1 - pn.x0 - 2);
      const Y = v => pn.y1 - 2 - v / 100 * (pn.y1 - pn.y0 - 6);
      const d = vals.map((v, s) => `${s ? "L" : "M"}${X(s).toFixed(1)} ${Y(v).toFixed(1)}`).join("");
      [pn.faint, pn.full].forEach(g => {
        g.textContent = "";
        if (target) el(g, "path", { d: target.map((v, s) => `${s ? "L" : "M"}${X(s).toFixed(1)} ${Y(v ? 80 : 5).toFixed(1)}`).join(""),
                                    fill: "none", stroke: GREY, "stroke-width": 1.3, "stroke-dasharray": "4 3" });
        el(g, "path", { d, fill: "none", stroke: col(1), "stroke-width": 2 });
      });
    }
    const tgtLeg = el(svg, "g", {});
    el(tgtLeg, "line", { x1: QX1 - 64, x2: QX1 - 48, y1: 57, y2: 57, stroke: GREY, "stroke-width": 1.3, "stroke-dasharray": "4 3" });
    txt(tgtLeg, QX1 - 44, 61, "target", { "font-size": 11.5 });

    // ---------------------------------------------------------------- input neurons (1 + 16 + 16 + 20)
    const neuron = (g, x, y) => el(g, "circle", { cx: x, cy: y, r: NR_IN, fill: "white", stroke: INK, "stroke-width": 1 });
    const ringOf = (cx, cy) => {
      const g = el(svg, "g", {});
      el(g, "circle", { cx, cy, r: 40, fill: "#f2f2f4", stroke: LINE, "stroke-width": 3 });
      txt(g, cx, cy + 5, "16", { "text-anchor": "middle", fill: INK, "font-size": 14 });
      return Array.from({ length: 16 }, (_, k) => {
        const a = 2 * Math.PI * k / 16;
        return neuron(g, cx + 30 * Math.cos(a), cy - 30 * Math.sin(a));
      });
    };
    const IX = 400;
    const fixIn = neuron(svg, IX, 34);
    txt(svg, IX - 13, 38.5, "1", { "text-anchor": "end", fill: INK, "font-size": 14 });
    const in1 = ringOf(IX, 101), in2 = ringOf(IX, 183);
    const rules = Array.from({ length: 20 }, (_, k) => neuron(svg, IX - 54 + (k % 10) * 12, 254 + Math.floor(k / 10) * 13));
    txt(svg, IX, 291, "20", { "text-anchor": "middle", fill: INK, "font-size": 14 });

    // ---------------------------------------------------------------- the network
    const NX = 640, NY = 165, NRAD = 125, OX = 880;
    const onCircle = deg => [NX + NRAD * Math.cos(deg * Math.PI / 180), NY + NRAD * Math.sin(deg * Math.PI / 180)];
    const wire = (a, b) => el(svg, "path", { d: `M ${a[0]} ${a[1]} C ${a[0] + 55} ${a[1]}, ${b[0] - 45} ${b[1]}, ${b[0]} ${b[1]}`,
                                            fill: "none", stroke: INK, "stroke-width": 1.4 });
    wire([IX + 8, 34], onCircle(222)); wire([IX + 40, 101], onCircle(196)); wire([IX + 40, 183], onCircle(170));
    wire([IX + 62, 260], onCircle(143));
    wire(onCircle(-38), [OX - 8, 34]); wire(onCircle(-6), [OX - 40, 112]);
    txt(svg, 792, 186, "read out", { "text-anchor": "middle", "font-size": 11.5 });
    txt(svg, 792, 200, "from E only", { "text-anchor": "middle", "font-size": 11.5 });
    el(svg, "circle", { cx: NX, cy: NY, r: NRAD, fill: "#ececef", stroke: INK, "stroke-width": 1.5 });
    const synG = el(svg, "g", {}), nrnG = el(svg, "g", {});
    const synTxt = txt(svg, NX, 304, "", { "text-anchor": "middle", "font-size": 11.5 });
    let nrn = [];
    function rebuildNet() {
      synG.textContent = ""; nrnG.textContent = "";
      const N = D.nets[point], n = N.n_e + N.n_i;
      const order = [];                              // an I neuron after every 4 E: the 80/20 split shows
      for (let e = 0, i = 0; order.length < n;) {
        for (let k = 0; k < 4 && e < N.n_e; k++) order.push(e++);
        if (i < N.n_i) order.push(N.n_e + i++);
      }
      const pos = new Array(n);
      order.forEach((u, k) => { const a = -Math.PI / 2 + 2 * Math.PI * k / n; pos[u] = [NX + 92 * Math.cos(a), NY + 92 * Math.sin(a)]; });
      N.syn.forEach(([a, b]) => {
        const [x1, y1] = pos[a], [x2, y2] = pos[b], mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
        el(synG, "path", { d: `M ${x1} ${y1} Q ${mx + (NX - mx) * 0.35} ${my + (NY - my) * 0.35} ${x2} ${y2}`, fill: "none",
                           stroke: a < N.n_e ? E : I, "stroke-opacity": 0.22, "stroke-width": 1 });
      });
      nrn = pos.map(([x, y], u) => {
        const isE = u < N.n_e, s = 13;
        const node = isE
          ? el(nrnG, "path", { d: `M ${x} ${y - s} L ${x + s * 0.92} ${y + s * 0.62} L ${x - s * 0.92} ${y + s * 0.62} Z` })
          : el(nrnG, "circle", { cx: x, cy: y, r: s * 0.7 });
        node.setAttribute("fill", isE ? E : I); node.setAttribute("fill-opacity", 0);
        node.setAttribute("stroke", isE ? E : I); node.setAttribute("stroke-width", 1.6);
        el(node, "title", {}, `${isE ? "E" : "I"} neuron, most active in ${N.pref[u]}`);
        return node;
      });
      synTxt.textContent = `${N.syn.length} synapses among the 20 shown`;
    }

    // ---------------------------------------------------------------- output neurons (1 + 16)
    const fixOut = neuron(svg, OX, 34);
    txt(svg, OX + 13, 38.5, "1", { fill: INK, "font-size": 14 });
    const out = ringOf(OX, 112);
    const tgt = el(svg, "path", { d: "M 0 -8 L 7 6 L -7 6 Z", fill: "none", stroke: INK, "stroke-width": 1.6, opacity: 0 });
    const dec = el(svg, "g", { opacity: 0 });
    const decLine = el(dec, "line", { x1: OX, y1: 112, stroke: INK, "stroke-width": 2.4 });
    const decHead = el(dec, "path", { d: "M 0 0 l -10 -5 l 0 10 Z", fill: INK });
    const outTxt = txt(svg, OX + 12, 170, "", { "text-anchor": "middle", "font-size": 12 });
    const legO = el(svg, "g", {});
    el(legO, "path", { d: `M ${OX - 38} 198 L ${OX - 32} 210 L ${OX - 44} 210 Z`, fill: "none", stroke: INK, "stroke-width": 1.4 });
    txt(legO, OX - 26, 210, "target", { "font-size": 11.5 });
    el(legO, "line", { x1: OX + 14, x2: OX + 30, y1: 205, y2: 205, stroke: INK, "stroke-width": 2.4 });
    txt(legO, OX + 35, 210, "read out", { "font-size": 11.5 });

    // ---------------------------------------------------------------- this network: accuracy and costs
    const BX0 = 1066, BX1 = 1146, bars = {};
    const netTitle = txt(svg, QX0, 182, "", { fill: INK, "font-size": 14 });
    const lo = k => Math.min(...D.points.map(p => D.nets[p][k])) / 2, hi = k => Math.max(...D.points.map(p => D.nets[p][k])) * 1.5;
    [["acc", "Accuracy (task)", 196], ["syn_cost", "Synaptic cost", 220], ["rate_cost", "Rate cost", 244]].forEach(([k, lab, y]) => {
      txt(svg, BX0 - 7, y + 10.5, lab, { "text-anchor": "end", fill: INK, "font-size": 12.5 });
      const bar = el(svg, "rect", { x: BX0, y, height: 13, fill: "#d1d1d6", stroke: INK, "stroke-width": 1 });
      bars[k] = { bar, val: txt(svg, BX1 + 5, y + 10.5, "", { "font-size": 12 }) };
    });
    el(svg, "line", { x1: BX0, x2: BX0, y1: 192, y2: 262, stroke: INK, "stroke-width": 1.5 });
    const keptTxt = txt(svg, QX0, 284, "", { "font-size": 12 });
    txt(svg, QX0, 300, "costs on a log axis", { "font-size": 11 });
    function setBars() {
      const N = D.nets[point];
      const w = (k, v) => k === "acc" ? v : (Math.log(v) - Math.log(lo(k))) / (Math.log(hi(k)) - Math.log(lo(k)));
      [["acc", N.acc[task]], ["syn_cost", N.syn_cost], ["rate_cost", N.rate_cost]].forEach(([k, v]) => {
        bars[k].bar.setAttribute("width", Math.max(2, w(k, v) * (BX1 - BX0)));
        bars[k].val.textContent = k === "acc" ? v.toFixed(2) : fmt(v);
      });
      netTitle.textContent = `${N.label}, seed 0`;
      netTitle.setAttribute("fill", PCOL[point]);
      keptTxt.textContent = `${(100 * N.kept).toFixed(1)}% of synapses kept`;
    }

    // ---------------------------------------------------------------- counts (as fig_setup)
    el(svg, "line", { x1: 345, x2: 935, y1: 318, y2: 318, stroke: INK, "stroke-width": 1 });
    txt(svg, IX, 338, "53 input neurons", { "text-anchor": "middle", fill: INK, "font-size": 14 });
    txt(svg, NX, 338, "256 recurrent neurons", { "text-anchor": "middle", fill: INK, "font-size": 14 });
    const sub = txt(svg, NX, 355, "", { "text-anchor": "middle", "font-size": 12 });
    [["204 E", E], [" · ", MUTED], ["52 I", I], [" · 20 shown, fill as they fire", MUTED]].forEach(([s, c]) => el(sub, "tspan", { fill: c }, s));
    txt(svg, OX, 338, "17 output neurons", { "text-anchor": "middle", fill: INK, "font-size": 14 });

    // ---------------------------------------------------------------- timeline (the slider)
    const TX0 = PX0, TX1 = QX1, TY = 374, TH = 18;
    const tl = el(svg, "g", { style: "cursor: pointer" });
    const tlBands = el(tl, "g", {});
    const tlHit = el(tl, "rect", { x: TX0, y: TY - 8, width: TX1 - TX0, height: TH + 16, fill: "transparent" });
    const knob = el(svg, "line", { y1: TY - 5, y2: TY + TH + 5, stroke: INK, "stroke-width": 2.5, "pointer-events": "none" });
    const tEnd = txt(svg, TX1, TY + TH + 16, "", { "text-anchor": "end", "font-size": 11.5 });
    txt(svg, TX0, TY + TH + 16, "0 ms", { "font-size": 11.5 });
    const TXs = s => TX0 + s / Math.max(tdim() - 1, 1) * (TX1 - TX0);
    function seek(ev) {
      const r = svg.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width * 1240, [a, b] = dragFrom;
      t = Math.round(Math.min(Math.max((x - a) / (b - a), 0), 1) * (tdim() - 1));
      playing = false; refresh();
    }
    tlHit.addEventListener("pointerdown", ev => { dragFrom = [TX0, TX1]; seek(ev); ev.stopPropagation(); });
    window.addEventListener("pointermove", ev => { if (dragFrom) seek(ev); });
    window.addEventListener("pointerup", () => { dragFrom = null; });

    function rebuildTask() {
      const T = D.tasks[task], N = D.nets[point];
      trace(P.fixIn, T.fix);
      heat(P.mod1, T.ring1); heat(P.mod2, T.ring2);
      heat(P.rule, T.fix.map(() => Array.from({ length: 20 }, (_, k) => (k === task ? 100 : 0))));
      trace(P.fixOut, N.out[task].map(y => y[0]), T.fix_target);
      heat(P.resp, N.out[task].map(y => y.slice(1)));
      tlBands.textContent = "";
      Object.entries(T.epochs).forEach(([k, [a, b]]) => {
        const x0 = TXs(a), x1 = TXs(Math.min(b, T.tdim - 1));
        el(tlBands, "rect", { x: x0, y: TY, width: Math.max(x1 - x0, 1), height: TH, stroke: LINE,
                              fill: k.startsWith("go") ? "#e7e5ff" : k.startsWith("stim") ? "#f1f1f4" : "#fafafa" });
        if (x1 - x0 > 50) txt(tlBands, (x0 + x1) / 2, TY + 13.5, EPOCH[k] || k, { "text-anchor": "middle", "font-size": 12 });
      });
      tEnd.textContent = `${(T.tdim - 1) * D.dt} ms`;
      setBars();
    }

    // ---------------------------------------------------------------- one time step
    function render() {
      const T = D.tasks[task], N = D.nets[point], s = Math.min(Math.floor(t), T.tdim - 1);
      fixIn.setAttribute("fill", col(T.fix[s] / 100));
      in1.forEach((c, k) => c.setAttribute("fill", col(T.ring1[s][k] / 100)));
      in2.forEach((c, k) => c.setAttribute("fill", col(T.ring2[s][k] / 100)));
      rules.forEach((c, k) => c.setAttribute("fill", k === task ? col(1) : "white"));
      const R = N.rates[task][s];
      nrn.forEach((node, u) => node.setAttribute("fill-opacity", (R[u] / 100).toFixed(3)));
      const y = N.out[task][s];
      fixOut.setAttribute("fill", col(y[0] / 100));
      let cx = 0, cy = 0;
      out.forEach((c, k) => {
        const v = y[k + 1] / 100, a = 2 * Math.PI * k / 16;
        c.setAttribute("fill", col(v)); cx += v * Math.cos(a); cy += v * Math.sin(a);
      });
      const g = T.target[s];
      if (g >= 0) {
        tgt.setAttribute("transform", `translate(${OX + 50 * Math.cos(g)} ${112 - 50 * Math.sin(g)}) rotate(${-g * 180 / Math.PI + 270})`);
        tgt.setAttribute("opacity", 1);
      } else tgt.setAttribute("opacity", 0);
      // a response = fixation output below 0.5 (Yang et al. 2019); its direction = the ring's population vector
      const inGo = Object.entries(T.epochs).some(([k, [a, b]]) => k.startsWith("go") && s >= a && s < b);
      if (inGo && y[0] < 50) {
        const a = Math.atan2(cy, cx), x2 = OX + 27 * Math.cos(a), y2 = 112 - 27 * Math.sin(a);
        decLine.setAttribute("x2", x2); decLine.setAttribute("y2", y2);
        decHead.setAttribute("transform", `translate(${x2} ${y2}) rotate(${-a * 180 / Math.PI})`);
        dec.setAttribute("opacity", 1);
        outTxt.textContent = g >= 0 ? `responds, error ${Math.round(Math.abs(Math.atan2(Math.sin(a - g), Math.cos(a - g))) * 180 / Math.PI)}°`
                                    : "responds (should keep fixating)";
      } else {
        dec.setAttribute("opacity", 0);
        outTxt.textContent = !inGo ? "" : T.fix_target[s] ? "keeps fixating: correct here" : "still fixating";
      }
      Object.values(P).forEach(pn => {
        const x = pn.x0 + (s + 1) / T.tdim * (pn.x1 - pn.x0);
        pn.clip.setAttribute("width", x - pn.x0); pn.head.setAttribute("x1", x); pn.head.setAttribute("x2", x);
      });
      knob.setAttribute("x1", TXs(s)); knob.setAttribute("x2", TXs(s));
      const ep = Object.entries(T.epochs).find(([, [a, b]]) => s >= a && s < b);
      clock.textContent = `${s * D.dt} ms · ${ep ? EPOCH[ep[0]] || ep[0] : ""}`;
    }

    function refresh() {
      Object.entries(taskBtn).forEach(([n, b]) => b.classList.toggle("on", byName[n] === task));
      Object.entries(ptBtn).forEach(([p, b]) => b.classList.toggle("on", p === point));
      play.textContent = playing ? "❚❚ pause" : "▶ play";
      play.classList.toggle("on", playing);
      render();
    }
    rebuildNet(); rebuildTask(); refresh();

    const section = root.closest("section");
    (function frame(now) {
      const shown = !section || section.classList.contains("present");
      if (shown && playing) {
        if (last !== null) {
          t = Math.min(t + (now - last) / MS_PER_STEP, tdim() - 1);
          if (t >= tdim() - 1) { playing = false; refresh(); } else render();
        }
        last = now;
      } else last = null;
      requestAnimationFrame(frame);
    })(performance.now());
  }

  function init() { document.querySelectorAll(".trial-sim").forEach(setup); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
