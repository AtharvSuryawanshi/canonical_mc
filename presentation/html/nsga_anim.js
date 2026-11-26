// NSGA-III on the real ms1 search, generations 0 and 1 (slide "Mapping the front: NSGA-III"). Canvas 2D via
// plot3d.js; data in assets/nsga_data.js (build_figs.py nsga). Same story as results/scripts/anim_nsga3.py:
// initialise -> train -> check -> rank -> breed, then train/check/rank generation 1.
// Restarts when the slide is shown, pauses while it is hidden.
(function () {
  const INK = "#1d1d1f", RED = "#E0393E", BLUE = "#1F8AD0";
  const tick = v => ({ v, label: String(v) });
  const CFG = {
    X: [-2.75, -0.6], Y: [-2.8, -1.45], Z: [0.2, 1.0],                 // log10 rate cost, log10 synaptic cost, acc
    XT: [-2.5, -2, -1.5, -1].map(tick), YT: [-2.6, -2.2, -1.8].map(tick),
    ZT: [0.2, 0.4, 0.6, 0.8, 1.0].map(v => ({ v, label: v.toFixed(1) })),
    xTitle: "rate cost (log₁₀)", yTitle: "synaptic cost (log₁₀)", zTitle: "worst-task accuracy",
    aspect: [1, 1, 0.66], margin: { l: 122, r: 108, t: 22, b: 64 },
  };
  const AZ = [-52, -26], EL = 24 * Math.PI / 180;

  const STAGES = [
    { name: "init", gen: 0, dur: 1.8, hold: 0.8, step: 0,
      cap: "Generation 0: 48 random pairs of budgets (rate, synaptic). Accuracy is not known yet." },
    { name: "train", gen: 0, dur: 3.2, hold: 0.9, step: 1,
      cap: "Each network is trained under its budgets: it rises to its worst-task accuracy and moves to the costs it reached." },
    { name: "check", gen: 0, dur: 1.2, hold: 1.3, step: 2,
      cap: "Networks with any task below 0.6 are dropped." },
    { name: "rank", gen: 0, dur: 1.2, hold: 2.0, step: 3,
      cap: "Front (ringed): no other network has higher accuracy and lower rate and synaptic cost." },
    { name: "breed", gen: 1, dur: 1.8, hold: 0.8, step: 4,
      cap: "New budgets are bred from the front and its spread: generation 1." },
    { name: "train", gen: 1, dur: 3.0, hold: 0.7, step: 1,
      cap: "Generation 1 is trained the same way." },
    { name: "check", gen: 1, dur: 1.0, hold: 0.9, step: 2,
      cap: "Networks with any task below 0.6 are dropped." },
    { name: "rank", gen: 1, dur: 1.2, hold: 4.5, step: 3,
      cap: "Parents and children ranked together: the front of generations 0 + 1." },
  ];
  const TOTAL = STAGES.reduce((s, k) => s + k.dur + k.hold, 0);

  const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
  const ease = t => { t = clamp(t, 0, 1); return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; };
  const back = t => { t = clamp(t, 0, 1); const c = 1.7; return 1 + (c + 1) * Math.pow(t - 1, 3) + c * Math.pow(t - 1, 2); };
  const lerp = (a, b, t) => a + (b - a) * t;
  const rgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16));
  const mix = (a, b, t) => a.map((v, i) => Math.round(lerp(v, b[i], t)));
  const css = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

  function stageAt(t) {
    let k = 0;
    while (k < STAGES.length - 1 && t > STAGES[k].dur + STAGES[k].hold) { t -= STAGES[k].dur + STAGES[k].hold; k++; }
    return { k, p: clamp(t / STAGES[k].dur, 0, 1) };
  }
  // progress (0..1) of stage `name` for generation `gen`, at stage k with progress p; 0 before, 1 after
  function prog(k, p, name, gen) {
    const j = STAGES.findIndex(s => s.name === name && s.gen === gen);
    return k < j ? 0 : k > j ? 1 : p;
  }
  // staggered per-point progress: point i of n starts at i/n * spread of the stage
  const stagger = (p, i, n, spread) => clamp((p - (i / n) * spread) / (1 - spread), 0, 1);

  function setup(root) {
    const data = window.NSGA_DATA || [];
    const P = Plot3D(root.querySelector("canvas"), CFG), ctx = P.ctx;
    const { X, Y, Z } = CFG;
    const steps = [...root.querySelectorAll(".nsga-step")];
    const cap = root.querySelector(".nsga-cap");
    const pts = data.map((d, i) => ({ ...d, c: rgb(d.col), idx: data.filter((e, j) => j < i && e.g === d.g).length }));
    const nG = [0, 1].map(g => pts.filter(d => d.g === g).length);
    const nFail = [0, 1].map(g => pts.filter(d => d.g === g && !d.ok).length);
    const nF0 = pts.filter(d => d.f0).length, nFb = pts.filter(d => d.fb).length;
    const azs = []; for (let a = AZ[0]; a <= AZ[1]; a += 2) azs.push(a);

    function plane(alpha) {
      if (alpha <= 0) return;
      const pl = [[X[0], Y[0]], [X[1], Y[0]], [X[1], Y[1]], [X[0], Y[1]]].map(([x, y]) => P.proj(x, y, 0.6));
      P.poly(pl); ctx.fillStyle = css(rgb(RED), 0.07 * alpha); ctx.fill();
      ctx.setLineDash([5, 4]); ctx.strokeStyle = css(rgb(RED), 0.6 * alpha); ctx.stroke(); ctx.setLineDash([]);
      const q = pl.reduce((a, b) => (b.x < a.x ? b : a));
      ctx.globalAlpha *= alpha;
      P.text("every task ≥ 0.6", { x: q.x - 8, y: q.y }, { color: RED, align: "right", size: 14 });
      ctx.globalAlpha /= alpha;
    }

    function draw(t) {
      P.resize();
      P.fit(azs, EL);
      const { k, p } = stageAt(t);
      const st = STAGES[k];
      P.az = (AZ[0] + (AZ[1] - AZ[0]) * (t / TOTAL)) * Math.PI / 180;
      P.el = EL;
      ctx.globalAlpha = clamp(Math.min(t / 0.5, (TOTAL - t) / 0.6), 0, 1);

      P.drawBox();
      plane(prog(k, p, "check", 0));
      P.drawAxes();

      const r0 = prog(k, p, "rank", 0), r1 = prog(k, p, "rank", 1);
      const items = [];
      pts.forEach(d => {
        const born = d.g === 0 ? stagger(prog(k, p, "init", 0), d.idx, nG[0], 0.6)
                               : stagger(prog(k, p, "breed", 1), d.idx, nG[1], 0.6);
        if (born <= 0) return;
        const tr = ease(stagger(prog(k, p, "train", d.g), d.idx, nG[d.g], 0.5));
        const ck = d.ok ? 0 : ease(prog(k, p, "check", d.g) * 1.4 - 0.2);
        const x = lerp(d.bx, d.rx, tr), y = lerp(d.by, d.ry, tr);
        let z = lerp(Z[0], d.acc, tr);
        if (!d.ok) z = lerp(z, Z[0], ck);
        let c = mix(d.g === 0 ? [142, 142, 147] : rgb(BLUE), d.c, tr);
        if (!d.ok) c = mix(c, rgb(RED), clamp(ck * 2, 0, 1));
        // fading: non-front of gen 0 after rank 0; everything off the combined front at rank 1
        let a = 1;
        if (d.g === 0 && !d.f0) a = lerp(1, 0.28, ease(r0));
        if (!d.fb) a = Math.min(a, lerp(1, 0.28, ease(r1)));
        let ring = 0;
        if (d.f0) ring = ease(r0 * 1.5);
        if (r1 > 0) ring = d.fb ? (d.f0 ? 1 : ease(r1 * 1.5)) : (d.f0 ? 1 - ease(r1 * 1.5) : 0);
        items.push({ q: P.proj(x, y, z), x, y, z, c, a, ring, cross: !d.ok && ck > 0.5, pop: back(born) });
      });

      // floor shadows and drop lines first, then points far to near
      items.forEach(it => {
        if (it.z <= Z[0] + 0.005 || it.cross) return;
        const f = P.proj(it.x, it.y, Z[0]);
        ctx.strokeStyle = `rgba(0,0,0,${0.13 * it.a})`; ctx.lineWidth = 1; P.line(it.q, f);
        ctx.beginPath(); ctx.ellipse(f.x, f.y, 4.5 * f.f, 2 * f.f, 0, 0, 2 * Math.PI);
        ctx.fillStyle = `rgba(0,0,0,${0.12 * it.a})`; ctx.fill();
      });
      items.sort((a, b) => b.q.d - a.q.d).forEach(it => {
        const r = 6.8 * it.q.f * it.pop;
        if (it.cross) {
          ctx.strokeStyle = css(rgb(RED), 0.9 * it.a); ctx.lineWidth = 2.4; const s = 5.5 * it.q.f;
          P.line({ x: it.q.x - s, y: it.q.y - s }, { x: it.q.x + s, y: it.q.y + s });
          P.line({ x: it.q.x - s, y: it.q.y + s }, { x: it.q.x + s, y: it.q.y - s });
          return;
        }
        ctx.beginPath(); ctx.arc(it.q.x, it.q.y, Math.max(r, 0), 0, 2 * Math.PI);
        ctx.fillStyle = css(it.c, it.a); ctx.fill();
        ctx.lineWidth = 1; ctx.strokeStyle = `rgba(255,255,255,${0.9 * it.a})`; ctx.stroke();
        if (it.ring > 0) {
          ctx.beginPath(); ctx.arc(it.q.x, it.q.y, r + 3.5 + 6 * (1 - it.ring), 0, 2 * Math.PI);
          ctx.lineWidth = 1.8; ctx.strokeStyle = `rgba(29,29,31,${it.ring})`; ctx.stroke();
        }
      });

      // count, top left
      let label = `Generation ${st.gen}: ${nG[st.gen]} networks`;
      if (st.name === "check") label = `Generation ${st.gen}: ${nFail[st.gen]} of ${nG[st.gen]} fail a task`;
      if (st.name === "rank" && st.gen === 0) label = `Front of generation 0: ${nF0} networks`;
      if (st.name === "rank" && st.gen === 1) label = `Front of generations 0 + 1: ${nFb} networks`;
      P.text(label, { x: 4, y: 14 }, { size: 16, color: INK, align: "left" });
      ctx.globalAlpha = 1;

      steps.forEach((s, i) => s.classList.toggle("on", i === st.step));
      if (cap.dataset.k !== String(k)) { cap.dataset.k = String(k); cap.textContent = st.cap; }
    }

    Plot3D.onSlide(root, t => draw(t % TOTAL));
  }

  function init() { document.querySelectorAll(".nsga-anim").forEach(setup); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
