// Project R.E.D. Pony — figures. Plain canvas, no libraries.
// Every figure says in its caption whether it shows measurements or a schematic.
(function () {
  'use strict';

  // ---------------------------------------------------------------- data (from the experiment logs)
  const DATA = {
    // Drone, zero-error probe: yaw-rate output r3 of the original network at 8 headings (deg).
    probe: { psi: [-180, -120, -60, -45, 0, 60, 120, 150], r3: [0.149, 0.267, 0.054, -0.020, -0.007, -0.104, -0.135, 0.034] },
    // Drone, rewritten yaw command (eq. 10).
    cmd: { K: 1, wmax: 0.8, B: 20, lambda: 2, H: 12 },
    // Drone, yaw mission (pairs: nominal, +30 ms delay).
    mission: {
      ref: [45, 90, -90, 170, 0], jump: [45, 45, 180, 100, 170], orig: [95.8, 140.8, 41.3, 140.8, 50.1],
      err: [[0.26, 0.26], [0.26, 0.26], [3.47, 3.47], [0.50, 0.50], [2.15, 2.08]],
      t10: [[1.6, 1.5], [1.5, 1.5], [4.7, 4.7], [2.8, 2.7], [4.2, 4.2]],
      pos: [[0.7, 0.8], [1.5, 1.4], [0.7, 0.6], [1.6, 1.5], [0.5, 0.5]],
      peak: [[1.2, 1.4], [2.3, 2.0], [2.0, 1.8], [6.4, 4.9], [7.2, 6.0]]
    },
    summary: [
      ['Hold: missions completed', '3/3 · 3/3', '3/3 · 3/3'],
      ['Hold: position error', '2.7 cm', '0.6 / 0.6 cm'],
      ['Hold: steady thrust oscillation', '0.01–0.04 N', '0.013 / 0.033 N'],
      ['Hold: peak vertical speed at takeoff', '1.5 / 1.9–2.1 m/s', '1.1 / 1.4 m/s'],
      ['Tilt at takeoff', '2.5 / 2.9°', '3.0 / 3.6–4.1°'],
      ['Cube: missions completed', '3/3 · 3/3', '3/3 · 3/3'],
      ['Cube: maximum tilt', '44 / 47°', '39 / 42°'],
      ['Cube: time with thrust below 1 N', '2.3 / 2.6–3.0 %', '1.6 / 2.5–2.6 %'],
      ['Yaw: missions completed', '1/1, not flown with delay', '2/3 · 3/3']
    ]
  };

  // ---------------------------------------------------------------- palette (mirrors site.css)
  const C = {
    bg: '#0d1014', panel: '#151a21', grid: '#20272f', line: '#2c343f',
    text: '#e8ecf1', muted: '#a4afbd', faint: '#66717f',
    orig: '#e8604f', hold: '#eeb05a', edit: '#6ad6ad', circuit: '#c6aef0', yaw: '#6fc8e6',
    unit: '#8fa3bd', idle: '#2b333f'
  };
  const F = { ui: '"IBM Plex Sans",system-ui,sans-serif', num: '"IBM Plex Mono",ui-monospace,monospace', math: '"STIX Two Text","Times New Roman",serif' };
  const calm = !!(window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches);

  // ---------------------------------------------------------------- math helpers
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const lerp = (a, b, p) => a + (b - a) * p;
  const elu = u => (u > 0 ? u : Math.expm1(u));
  const delu = u => (u > 0 ? 1 : Math.exp(u));
  const rad = d => d * Math.PI / 180;
  const deg = r => r * 180 / Math.PI;
  const wrapDeg = d => { d = ((d + 180) % 360 + 360) % 360 - 180; return d === -180 ? 180 : d; };
  const wrapRad = r => rad(wrapDeg(deg(r)));
  const fmt = (v, d) => {
    if (!isFinite(v)) return '—';
    const s = Math.abs(v).toFixed(d === undefined ? 1 : d);
    return (v < 0 && Number(s) !== 0 ? '−' : '') + s;
  };
  function interp(xs, ys, x) {
    if (x <= xs[0]) return ys[0];
    const n = xs.length - 1;
    if (x >= xs[n]) return ys[n];
    let i = 0; while (x > xs[i + 1]) i++;
    return lerp(ys[i], ys[i + 1], (x - xs[i]) / (xs[i + 1] - xs[i]));
  }
  function rng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
  function shuffle(n, seed) { const r = rng(seed), a = Array.from({ length: n }, (_, i) => i); for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }

  // ---------------------------------------------------------------- drawing helpers
  function P(ctx, w, h) { this.c = ctx; this.w = w; this.h = h; this.narrow = w < 560; }
  P.prototype.line = function (x1, y1, x2, y2, col, lw, dash, alpha) {
    const c = this.c; c.save(); c.strokeStyle = col || C.muted; c.lineWidth = lw || 1.5; c.globalAlpha = alpha === undefined ? 1 : alpha;
    c.setLineDash(dash || []); c.lineCap = 'round'; c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2); c.stroke(); c.restore();
  };
  P.prototype.poly = function (pts, col, lw, dash, alpha) {
    if (pts.length < 2) return;
    const c = this.c; c.save(); c.strokeStyle = col; c.lineWidth = lw || 2; c.globalAlpha = alpha === undefined ? 1 : alpha;
    c.setLineDash(dash || []); c.lineJoin = 'round'; c.lineCap = 'round'; c.beginPath(); c.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) c.lineTo(pts[i][0], pts[i][1]); c.stroke(); c.restore();
  };
  P.prototype.dot = function (x, y, r, fill, stroke, lw, alpha) {
    const c = this.c; c.save(); c.globalAlpha = alpha === undefined ? 1 : alpha; c.beginPath(); c.arc(x, y, Math.max(0, r), 0, 2 * Math.PI);
    if (fill) { c.fillStyle = fill; c.fill(); }
    if (stroke) { c.strokeStyle = stroke; c.lineWidth = lw || 1.5; c.stroke(); }
    c.restore();
  };
  P.prototype.text = function (s, x, y, o) {
    o = o || {}; const c = this.c; c.save(); c.fillStyle = o.col || C.muted; c.globalAlpha = o.alpha === undefined ? 1 : o.alpha;
    c.font = (o.italic ? 'italic ' : '') + (o.weight || 400) + ' ' + (o.size || 12) + 'px ' + (F[o.font || 'ui']);
    c.textAlign = o.align || 'left'; c.textBaseline = o.base || 'alphabetic'; c.fillText(s, x, y); c.restore();
  };
  P.prototype.arrow = function (x1, y1, x2, y2, col, lw, head) {
    const L = Math.hypot(x2 - x1, y2 - y1); if (L < 0.5) return;
    const a = Math.atan2(y2 - y1, x2 - x1), t = Math.min(head || 9, L * 0.6);
    this.line(x1, y1, x2 - t * 0.6 * Math.cos(a), y2 - t * 0.6 * Math.sin(a), col, lw);
    const c = this.c; c.save(); c.fillStyle = col; c.beginPath(); c.moveTo(x2, y2);
    c.lineTo(x2 - t * Math.cos(a - 0.45), y2 - t * Math.sin(a - 0.45)); c.lineTo(x2 - t * Math.cos(a + 0.45), y2 - t * Math.sin(a + 0.45));
    c.closePath(); c.fill(); c.restore();
  };
  // Axes: returns mapping helpers. o = {x,y,w,h,xr,yr,xt,yt,xf,yf,xl,yl}
  P.prototype.axes = function (o) {
    const X = v => o.x + (v - o.xr[0]) / (o.xr[1] - o.xr[0]) * o.w;
    const Y = v => o.y + o.h - (v - o.yr[0]) / (o.yr[1] - o.yr[0]) * o.h;
    const c = this.c; c.save(); c.strokeStyle = C.grid; c.lineWidth = 1; c.beginPath();
    (o.yt || []).forEach(v => { c.moveTo(o.x, Y(v)); c.lineTo(o.x + o.w, Y(v)); });
    (o.xt || []).forEach(v => { c.moveTo(X(v), o.y); c.lineTo(X(v), o.y + o.h); });
    c.stroke(); c.restore();
    const xf = o.xf || (v => fmt(v, 0)), yf = o.yf || (v => fmt(v, 0));
    (o.yt || []).forEach(v => this.text(yf(v), o.x - 6, Y(v), { size: 11, font: 'num', align: 'right', base: 'middle' }));
    (o.xt || []).forEach(v => this.text(xf(v), X(v), o.y + o.h + 6, { size: 11, font: 'num', align: 'center', base: 'top' }));
    if (o.xl) this.text(o.xl, o.x + o.w, o.y + o.h + 24, { size: 12, align: 'right', base: 'top' });
    if (o.yl) this.text(o.yl, o.x, o.y - 9, { size: 12 });
    const clip = fn => { c.save(); c.beginPath(); c.rect(o.x - 1, o.y - 2, o.w + 2, o.h + 4); c.clip(); fn(); c.restore(); };
    return { X, Y, clip };
  };

  // ---------------------------------------------------------------- figure registry and loop
  const figs = []; let running = false; let last = 0; const t0 = performance.now();
  function fig(id, o) {
    const cv = document.getElementById(id); if (!cv) return null;
    const f = { cv, o, vis: false, dirty: true, w: 0, h: 0, dpr: 1, broken: false };
    f.redraw = () => { f.dirty = true; wake(); };
    figs.push(f); return f;
  }
  function measure(f) {
    const W = Math.max(240, Math.round(f.cv.parentElement.clientWidth));
    const ratio = W < 640 ? (f.o.ratioNarrow || f.o.ratio * 1.6) : f.o.ratio;
    const H = Math.round(clamp(W * ratio, f.o.minH || 220, f.o.maxH || 620));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (W !== f.w || H !== f.h || dpr !== f.dpr) {
      f.w = W; f.h = H; f.dpr = dpr; f.cv.width = Math.round(W * dpr); f.cv.height = Math.round(H * dpr);
      f.cv.style.height = H + 'px'; f.dirty = true;
    }
  }
  function draw(f, t, dt) {
    if (f.broken) return;
    const c = f.cv.getContext('2d'); c.setTransform(f.dpr, 0, 0, f.dpr, 0, 0); c.clearRect(0, 0, f.w, f.h);
    try { f.o.draw(new P(c, f.w, f.h), t, dt); } catch (e) { f.broken = true; console.error(f.cv.id, e); }
    f.dirty = false;
  }
  function frame(now) {
    const dt = Math.min(0.05, Math.max(0, (now - last) / 1000)); last = now;
    const t = (now - t0) / 1000; let again = false;
    for (const f of figs) {
      if (!f.vis) continue;
      const anim = !calm && (typeof f.o.animate === 'function' ? f.o.animate() : !!f.o.animate);
      if (anim) { draw(f, t, dt); again = true; } else if (f.dirty) draw(f, t, 0);
    }
    if (again) requestAnimationFrame(frame); else running = false;
  }
  function wake() { if (running) return; running = true; last = performance.now(); requestAnimationFrame(frame); }

  // ---------------------------------------------------------------- small UI builders
  function slider(host, o) {
    const lab = document.createElement('label'); lab.className = 'slider';
    const row = document.createElement('span'); row.className = 'row';
    const name = document.createElement('span'); name.innerHTML = o.label;
    const out = document.createElement('output');
    const inp = document.createElement('input'); inp.type = 'range'; inp.min = o.min; inp.max = o.max; inp.step = o.step || 1; inp.value = o.value;
    row.append(name, out); lab.append(row, inp); host.appendChild(lab);
    const s = {
      value: Number(inp.value), input: inp,
      set(v, silent) { inp.value = v; s.value = Number(inp.value); out.textContent = (o.fmt || (x => x))(s.value); if (!silent && o.on) o.on(s.value); }
    };
    inp.addEventListener('input', () => s.set(Number(inp.value)));
    s.set(s.value, true); return s;
  }
  function segmented(id, on) {
    const box = document.getElementById(id); if (!box) return null;
    const btns = Array.from(box.querySelectorAll('button'));
    const api = {
      key: (btns.find(b => b.getAttribute('aria-pressed') === 'true') || btns[0]).dataset.k,
      set(k, silent) { api.key = k; btns.forEach(b => b.setAttribute('aria-pressed', String(b.dataset.k === k))); if (!silent) on(k); }
    };
    btns.forEach(b => b.addEventListener('click', () => api.set(b.dataset.k)));
    return api;
  }
  const $ = id => document.getElementById(id);

  // ================================================================= 1. hero network
  (function hero() {
    const L1 = 512, L2 = 512;
    const p1 = shuffle(L1, 11), p2 = shuffle(L2, 29);
    const act1 = new Set(p1.slice(0, 99)), act2 = new Set(p2.slice(0, 22));
    const free1 = p1.slice(99, 127), free2 = p2.slice(22, 27);         // 28 + 5 retired, as in the drone edit
    const fs1 = new Set(free1), fs2 = new Set(free2);
    const r = rng(5);
    const pick = (arr, n) => Array.from({ length: n }, () => arr[Math.floor(r() * arr.length)]);
    const A1 = Array.from(act1), A2 = Array.from(act2);
    const links = [];
    pick(A1, 26).forEach(u => links.push({ a: ['in', Math.floor(r() * 17)], b: ['l1', u], ph: r() }));
    pick(A1, 30).forEach(u => links.push({ a: ['l1', u], b: ['l2', A2[Math.floor(r() * A2.length)]], ph: r() }));
    A2.forEach(u => links.push({ a: ['l2', u], b: ['out', Math.floor(r() * 4)], ph: r() }));
    const circ = [];
    free1.slice(0, 23).forEach((u, i) => circ.push({ a: ['in', 4 + (i % 2) * 4], b: ['l1', u], ph: r() }));   // attitude cos/sin -> sensors
    free1.slice(0, 23).forEach((u, i) => circ.push({ a: ['l1', u], b: ['l2', free2[i % 5]], ph: r() }));
    free2.forEach((u, i) => circ.push({ a: ['l2', u], b: ['out', i % 2 ? 0 : 3], ph: r() }));
    const caps = [
      'Schematic of the drone hover network: 17 inputs, two layers of 512 units, 4 outputs. Unit counts are real, positions are not. Bright units vary across 200 000 states; 413 and 490 dim ones are practically constant.',
      'Idle units are freed: each one outputs a constant, so its contribution is moved into the next bias and its connections are cleared. The network computes the same thing. The final drone edit retired 28 + 5 units.',
      'A circuit is written into the freed units (violet): 23 yaw sensors fed by the attitude inputs, and the paths to the yaw-rate and thrust outputs. Same layer sizes, same file format.'
    ];
    let stage = 0, touched = false, stageT = 0;
    const seg = segmented('hero-seg', k => { touched = true; setStage(Number(k)); });
    function setStage(k) { stage = k; stageT = performance.now(); $('hero-cap').textContent = caps[k]; if (seg) seg.set(String(k), true); f.redraw(); }
    const f = fig('hero-net', {
      ratio: 0.36, ratioNarrow: 0.72, minH: 240, maxH: 440, animate: true,
      draw(g, t) {
        if (!touched && !calm && performance.now() - stageT > 6500) setStage((stage + 1) % 3);
        const W = g.w, H = g.h;
        const s = Math.min((H - 30) / 32, W / (g.narrow ? 44 : 62));
        const gw = 16 * s, gh = 32 * s;
        const gap = (W - 2 * gw) / 3;
        const x1 = gap, x2 = x1 + gw + gap, y0 = (H - gh) / 2 + 6;
        const xin = gap * 0.42, xout = x2 + gw + gap * 0.6;
        const pin = k => [xin, y0 + gh * (k + 0.5) / 17];
        const pout = k => [xout, y0 + gh * (k + 1.5) / 7];
        const pl1 = i => [x1 + s * (i % 16 + 0.5), y0 + s * (Math.floor(i / 16) + 0.5)];
        const pl2 = i => [x2 + s * (i % 16 + 0.5), y0 + s * (Math.floor(i / 16) + 0.5)];
        const pos = n => (n[0] === 'in' ? pin(n[1]) : n[0] === 'out' ? pout(n[1]) : n[0] === 'l1' ? pl1(n[1]) : pl2(n[1]));
        const ru = Math.max(0.9, s * 0.33);
        // links
        const drawLinks = (arr, col, alpha, speed) => {
          for (const L of arr) {
            const a = pos(L.a), b = pos(L.b);
            g.line(a[0], a[1], b[0], b[1], col, 0.8, null, alpha);
            if (!calm) { const p = (t * speed + L.ph) % 1; g.dot(lerp(a[0], b[0], p), lerp(a[1], b[1], p), 1.6, col, null, 0, Math.min(1, alpha * 4)); }
          }
        };
        drawLinks(links, C.unit, 0.13, 0.35);
        const since = (performance.now() - stageT) / 1000;
        if (stage === 2) drawLinks(circ, C.circuit, 0.28 * clamp(since / 0.8, 0, 1), 0.5);
        // units
        const c = g.c;
        const batch = (col, list, pfn, rr) => { c.fillStyle = col; c.beginPath(); for (const i of list) { const p = pfn(i); c.moveTo(p[0] + rr, p[1]); c.arc(p[0], p[1], rr, 0, 2 * Math.PI); } c.fill(); };
        const idle1 = [], idle2 = [], on1 = [], on2 = [];
        for (let i = 0; i < L1; i++) { if (act1.has(i)) on1.push(i); else if (!(stage > 0 && fs1.has(i))) idle1.push(i); }
        for (let i = 0; i < L2; i++) { if (act2.has(i)) on2.push(i); else if (!(stage > 0 && fs2.has(i))) idle2.push(i); }
        batch(C.idle, idle1, pl1, ru); batch(C.idle, idle2, pl2, ru);
        batch(C.unit, on1, pl1, ru); batch(C.unit, on2, pl2, ru);
        if (stage > 0) {
          const pulse = calm ? 1 : 0.6 + 0.4 * Math.sin(t * 4);
          for (const i of free1) { const p = pl1(i); stage === 2 ? g.dot(p[0], p[1], ru * 1.15, C.circuit) : g.dot(p[0], p[1], ru * 1.05, null, C.circuit, 1.1, pulse); }
          for (const i of free2) { const p = pl2(i); stage === 2 ? g.dot(p[0], p[1], ru * 1.15, C.circuit) : g.dot(p[0], p[1], ru * 1.05, null, C.circuit, 1.1, pulse); }
        }
        for (let k = 0; k < 17; k++) { const p = pin(k); g.dot(p[0], p[1], Math.max(2, s * 0.32), k >= 3 && k < 12 ? C.yaw : C.muted); }
        for (let k = 0; k < 4; k++) { const p = pout(k); g.dot(p[0], p[1], Math.max(3, s * 0.5), k === 0 ? C.hold : k === 3 ? C.yaw : C.muted); }
        const lab = { size: g.narrow ? 10.5 : 12, align: 'center', base: 'bottom' };
        g.text(g.narrow ? 'in' : '17 inputs', xin, y0 - 8, lab); g.text('layer 1 · 512', x1 + gw / 2, y0 - 8, lab);
        g.text('layer 2 · 512', x2 + gw / 2, y0 - 8, lab); g.text(g.narrow ? 'out' : '4 outputs', xout, y0 - 8, lab);
        if (!g.narrow) {
          g.text('attitude', xin - 8, pin(7)[1], { size: 11, align: 'right', base: 'middle', col: C.yaw });
          ['T', 'ωx', 'ωy', 'ωz'].forEach((n, k) => g.text(n, pout(k)[0] + 9, pout(k)[1], { size: 13, font: 'math', italic: true, base: 'middle', col: C.text }));
        }
      }
    });
    if (f) setStage(0);
  })();

  // ================================================================= 2. ELU toolkit
  (function kit() {
    const B = {
      pass: { xr: [-3, 3], yr: [-3, 3], p: { label: 'Bias <i>B</i>', min: 0, max: 4, step: 0.1, value: 1, fmt: v => v.toFixed(1) },
        exact: x => x, approx: (x, q) => elu(x + q) - q,
        eq: 'y = elu(x + B) − B   ≈ x   for x > −B',
        src: 'A bias keeps the unit on its straight side, so it hands a value to the next layer unchanged. The H1 circuit uses B = 1 for its registers and pass-throughs. Lower B and the left end bends.' },
      abs: { xr: [-2, 2], yr: [-0.6, 2.2], p: { label: 'Sharpness <i>S</i>', min: 1, max: 60, step: 1, value: 6, fmt: v => v.toFixed(0) },
        exact: x => Math.abs(x), approx: (x, q) => (elu(q * x) + elu(-q * x)) / q,
        eq: '|x| ≈ ( elu(S x) + elu(−S x) ) / S',
        src: 'Two sharp units; the error is about −1/S. The H1 gate reads |sin| + |cos| of the gait clock and |ω_z command| this way with S = 2000.' },
      clamp: { xr: [-3, 3], yr: [-1.8, 1.8], p: { label: 'Sharpness <i>S</i>', min: 2, max: 60, step: 1, value: 8, fmt: v => v.toFixed(0) },
        exact: x => clamp(x, -1, 1), approx: (x, q) => (elu(q * (x + 1)) - elu(q * (x - 1))) / q - 1,
        eq: 'clamp(x, ±L) ≈ ( elu(S (x + L)) − elu(S (x − L)) ) / S − L          (here L = 1)',
        src: 'Bounds the integrators: in H1 the heading error stays within ±0.35 rad and its integral within ±0.30 rad·s (S = 200). Multiply L by a gate and the same clamp resets the register to zero.' },
      sq: { xr: [-2, 2], yr: [-0.4, 4.2], p: { label: 'Input scale <i>s</i>', min: 0.5, max: 10, step: 0.1, value: 2, fmt: v => v.toFixed(1) },
        exact: x => x * x, approx: (x, q) => 2 * q * q * (elu(x / q) + elu(-x / q)),
        eq: 'x² ≈ 2 s² ( elu(x / s) + elu(−x / s) )',
        src: 'elu(u) + elu(−u) ≈ u²/2 only for small u: a larger scale s shrinks the input and the error, at the cost of float32 precision. H1 uses s = 10.' },
      prod: { xr: [-2, 2], yr: [-2, 2], p: { label: 'Input scale <i>s</i>', min: 0.5, max: 10, step: 0.1, value: 2, fmt: v => v.toFixed(1) },
        exact: x => 0.8 * x,
        approx: (x, q) => { const sq = u => 2 * q * q * (elu(u / q) + elu(-u / q)); return (sq(x + 0.8) - sq(x - 0.8)) / 4; },
        eq: 'a · b = ( (a + b)² − (a − b)² ) / 4   →   four square units          (plotted: a · 0.8)',
        src: 'The H1 yaw-rate estimate needs ω_y g_y and ω_z g_z, products of a gyro and a gravity component: four units each, two layers deep.' },
      gate: { xr: [-1, 2], yr: [-0.4, 1.4], p: { label: 'Sharpness <i>S</i>', min: 2, max: 200, step: 1, value: 20, fmt: v => v.toFixed(0) },
        exact: x => clamp(x, 0, 1), approx: (x, q) => (elu(q * x) - elu(q * (x - 1))) / q,
        eq: 'act = ( elu(S z) − elu(S (z − 1)) ) / S,     z = 6 (|sin φ| + |cos φ|) − 4.5 − 120 |ω_z,cmd|',
        src: 'A switch: 1 when the gait clock runs and the turn stick is at zero, 0 otherwise. In H1 it decides when the heading hold is active (S = 200).' }
    };
    let key = 'pass', sl = null;
    const ctl = $('kit-ctl');
    function setKey(k) {
      key = k; const b = B[k];
      ctl.innerHTML = '';
      sl = slider(ctl, Object.assign({}, b.p, { on: () => f.redraw() }));
      $('kit-eq').textContent = b.eq; $('kit-src').textContent = b.src;
      f.redraw();
    }
    segmented('kit-seg', setKey);
    const f = fig('kit', {
      ratio: 0.42, ratioNarrow: 0.8, minH: 260, maxH: 420,
      draw(g) {
        const b = B[key], q = sl ? sl.value : b.p.value;
        const m = { l: 44, r: 16, t: 26, b: 34 };
        const ax = g.axes({ x: m.l, y: m.t, w: g.w - m.l - m.r, h: g.h - m.t - m.b, xr: b.xr, yr: b.yr,
          xt: ticks(b.xr), yt: ticks(b.yr), xf: v => fmt(v, Number.isInteger(v) ? 0 : 1), yf: v => fmt(v, Number.isInteger(v) ? 0 : 1), xl: key === 'gate' ? 'z' : key === 'prod' ? 'a' : 'x' });
        const N = 300, ex = [], ap = []; let err = 0;
        for (let i = 0; i <= N; i++) {
          const x = lerp(b.xr[0], b.xr[1], i / N), e = b.exact(x), a = b.approx(x, q);
          ex.push([ax.X(x), ax.Y(e)]); ap.push([ax.X(x), ax.Y(a)]); err = Math.max(err, Math.abs(a - e));
        }
        ax.clip(() => { g.line(ax.X(0), m.t, ax.X(0), g.h - m.b, C.line, 1); g.line(m.l, ax.Y(0), g.w - m.r, ax.Y(0), C.line, 1);
          g.poly(ap, C.circuit, 3.2); g.poly(ex, C.text, 1.4, [5, 4]); });
        g.text('max error on this range: ' + fmt(err, 3), g.w - m.r, m.t - 10, { size: 12, align: 'right', font: 'num', col: C.circuit });
        g.text('exact', m.l + 4, m.t - 10, { size: 12, col: C.text });
        g.text('ELU units', m.l + 48, m.t - 10, { size: 12, col: C.circuit });
      }
    });
    function ticks(r) { const span = r[1] - r[0], st = span > 4 ? 1 : 0.5, out = []; for (let v = Math.ceil(r[0] / st) * st; v <= r[1] + 1e-9; v += st) out.push(Math.round(v * 10) / 10); return out; }
    if (f) setKey('pass');
  })();

  // ================================================================= 3. before / after the fold
  (function fold() {
    const u0 = -3; let sa, sd;
    const ctl = $('fold-ctl');
    const f = fig('fold-cv', {
      ratio: 0.62, ratioNarrow: 0.85, minH: 260, maxH: 460, animate: true,
      draw(g, t) {
        const a = sa ? sa.value : 0, d = sd ? sd.value : 0;
        const m = { l: 40, r: 70, t: 22, b: 34 };
        const ax = g.axes({ x: m.l, y: m.t, w: g.w - m.l - m.r, h: g.h - m.t - m.b, xr: [-5, 4.5], yr: [-1.5, 4.5],
          xt: [-4, -2, 0, 2, 4], yt: [-1, 0, 1, 2, 3, 4], xl: 'number before the fold' });
        const curve = off => { const p = []; for (let i = 0; i <= 200; i++) { const u = lerp(-5, 4.5, i / 200); p.push([ax.X(u), ax.Y(elu(u) + off)]); } return p; };
        ax.clip(() => {
          g.line(ax.X(0), m.t, ax.X(0), g.h - m.b, C.line, 1);
          g.poly(curve(0), C.faint, 2);
          if (Math.abs(d) > 1e-6) g.poly(curve(d), C.orig, 2, [6, 5]);
        });
        const wob = calm ? 0 : 0.45 * Math.sin(t * 2.4);
        const u = u0 + a, uw = u + wob, y = elu(uw) + d, sl = delu(u);
        // tangent
        ax.clip(() => g.line(ax.X(u - 1.6), ax.Y(elu(u) + d - 1.6 * sl), ax.X(u + 1.6), ax.Y(elu(u) + d + 1.6 * sl), C.circuit, 1.6, [4, 4]));
        // shift arrows
        if (Math.abs(a) > 0.01) g.arrow(ax.X(u0), ax.Y(elu(u0)) + 14, ax.X(u), ax.Y(elu(u0)) + 14, C.edit, 2);
        if (Math.abs(d) > 0.01) g.arrow(ax.X(u0) - 14, ax.Y(elu(u0)), ax.X(u0) - 14, ax.Y(elu(u0) + d), C.orig, 2);
        g.dot(ax.X(u0), ax.Y(elu(u0)), 3.5, null, C.faint, 1.5);
        g.dot(ax.X(uw), ax.Y(y), 6, C.text);
        // input wobble and output reaction
        g.line(ax.X(u - 0.45), g.h - m.b - 6, ax.X(u + 0.45), g.h - m.b - 6, C.yaw, 3);
        const lo = elu(u - 0.45) + d, hi = elu(u + 0.45) + d, xr = g.w - m.r + 26;
        g.line(xr, ax.Y(lo), xr, ax.Y(hi), C.yaw, 4);
        g.dot(xr, ax.Y(y), 4.5, C.text);
        g.text('output', xr, m.t - 6, { size: 11.5, align: 'center' });
        g.text('moves', xr, ax.Y(lo) + 16, { size: 11, align: 'center', col: C.yaw });
        g.text(fmt(hi - lo, 2), xr, ax.Y(lo) + 30, { size: 11, align: 'center', font: 'num', col: C.yaw });
        $('fold-ro').innerHTML = '<span class="ro">before the fold<b>' + fmt(u, 2) + '</b></span><span class="ro">output<b>' + fmt(elu(u) + d, 2) +
          '</b></span><span class="ro">slope (gain)<b style="color:' + (sl > 0.5 ? C.edit : C.orig) + '">' + fmt(sl, 2) + '</b></span>';
      }
    });
    if (!f) return;
    sa = slider(ctl, { label: 'Add <b style="color:var(--edit)">before</b> the fold', min: -2, max: 6, step: 0.05, value: 0, fmt: v => fmt(v, 2), on: () => f.redraw() });
    sd = slider(ctl, { label: 'Add <b style="color:var(--orig)">after</b> the fold', min: -1, max: 3, step: 0.05, value: 0, fmt: v => fmt(v, 2), on: () => f.redraw() });
    segmented('fold-seg', k => {
      const v = { flat: [0, 0], after: [0, 1.45], before: [3.5, 0] }[k];
      sa.set(v[0], true); sd.set(v[1], true); f.redraw();
    });
  })();

  // ================================================================= 4. H1 walk
  (function h1() {
    const V = 0.8, T = 30, PLAY = 10;
    const agents = [
      { name: 'original', col: C.orig, yaw: -2.84, lat: 0 },
      { name: 'heading hold', col: C.hold, yaw: -0.017, lat: -0.0213 },
      { name: 'heading + lateral', col: C.edit, yaw: 0, lat: -0.0032 }
    ];
    function state(a, t) {
      const w = rad(a.yaw);
      let x, y;
      if (Math.abs(w) < 1e-6) { x = V * t; y = 0; } else { x = V / w * Math.sin(w * t); y = V / w * (1 - Math.cos(w * t)); }
      y += a.lat * t;
      return { x, y, psi: w * t };
    }
    let start = -1, playing = false;
    const btn = $('h1-play');
    const f = fig('h1-cv', {
      ratio: 0.4, ratioNarrow: 1.25, minH: 300, maxH: 520, animate: () => playing,
      draw(g) {
        let tt = T;
        if (playing) { tt = clamp((performance.now() - start) / 1000 / PLAY * T, 0, T); if (tt >= T) playing = false; }
        else if (start < 0) tt = 0;
        if (calm && start >= 0) { tt = T; playing = false; }
        const narrow = g.narrow;
        // panel geometry
        const pA = narrow ? { x: 40, y: 24, w: g.w - 56, h: g.h * 0.52 - 40 } : { x: 40, y: 24, w: g.w * 0.5 - 60, h: g.h - 60 };
        const pB = narrow ? { x: 52, y: g.h * 0.52 + 40, w: g.w - 68, h: g.h * 0.48 - 78 } : { x: g.w * 0.5 + 40, y: 24, w: g.w * 0.5 - 56, h: g.h - 60 };
        // panel A: equal scale top view, forward to the right, lateral up
        const xr = [-1, 25], spanX = xr[1] - xr[0];
        const k = pA.w / spanX; const spanY = pA.h / k; const yr = [-spanY + 2, 2];
        const A = g.axes(Object.assign({}, pA, { xr, yr, xt: [0, 5, 10, 15, 20, 25], yt: tickRange(yr, 5), xl: 'forward [m]', yl: 'top view, equal scale [m]' }));
        // panel B: lateral deviation [cm] vs forward [m]
        const Bx = g.axes(Object.assign({}, pB, { xr: [0, 25], yr: [-75, 15], xt: [0, 5, 10, 15, 20, 25], yt: [-60, -40, -20, 0], xl: 'forward [m]', yl: 'lateral deviation [cm]' }));
        A.clip(() => g.line(A.X(0), A.Y(0), A.X(25), A.Y(0), C.faint, 1, [3, 4]));
        Bx.clip(() => g.line(Bx.X(0), Bx.Y(0), Bx.X(25), Bx.Y(0), C.faint, 1, [3, 4]));
        const ro = [];
        agents.forEach(a => {
          const pts = [], ptsB = [];
          for (let i = 0; i <= 150; i++) { const s = state(a, tt * i / 150); pts.push([A.X(s.x), A.Y(s.y)]); ptsB.push([Bx.X(s.x), Bx.Y(s.y * 100)]); }
          A.clip(() => g.poly(pts, a.col, 2.4));
          Bx.clip(() => g.poly(ptsB, a.col, 2.4));
          const s = state(a, tt), px = A.X(s.x), py = A.Y(s.y);
          A.clip(() => { g.dot(px, py, 5, a.col); g.line(px, py, px + 13 * Math.cos(s.psi), py - 13 * Math.sin(s.psi), a.col, 2.4); });
          if (s.y * 100 > -75) Bx.clip(() => g.dot(Bx.X(s.x), Bx.Y(s.y * 100), 4.5, a.col));
          ro.push('<span class="ro"><span style="color:' + a.col + '">' + a.name + '</span><b>' + fmt(deg(s.psi), 1) + '° · ' + (Math.abs(s.y) >= 1 ? fmt(s.y, 1) + ' m' : fmt(s.y * 100, 0) + ' cm') + '</b></span>');
        });
        $('h1-ro').innerHTML = '<span class="ro">time<b>' + fmt(tt, 1) + ' s</b></span>' + ro.join('');
      }
    });
    function tickRange(r, st) { const out = []; for (let v = Math.ceil(r[0] / st) * st; v <= r[1]; v += st) out.push(v); return out; }
    function play() { start = performance.now(); playing = true; if (f) f.redraw(); }
    if (btn) btn.addEventListener('click', play);
    if (f && 'IntersectionObserver' in window) {
      const io = new IntersectionObserver(es => { es.forEach(e => { if (e.isIntersecting && start < 0) { play(); io.disconnect(); } }); }, { threshold: 0.5 });
      io.observe(f.cv);
    }
  })();

  // ================================================================= 5. drone yaw: original vs edited
  (function yaw() {
    const xs = DATA.probe.psi.concat([180]), ys = DATA.probe.r3.concat([DATA.probe.r3[0]]);
    const wOrig = psiDeg => 4 * clamp(interp(xs, ys, wrapDeg(psiDeg)), -1, 1);
    const wEdit = e => { const { K, wmax, B, lambda, H } = DATA.cmd; const h = s => elu(B * (s * Math.sin(e) - lambda * (1 + Math.cos(e)))); return wmax * clamp((K / wmax) * Math.sin(e) + (H / B) * (h(1) - h(-1)), -1, 1); };
    let sRef, sStart, psiO, psiE, tSim = 0, run = false;
    const ctl = $('yaw-ctl');
    function reset() { psiO = rad(sStart.value); psiE = rad(sStart.value); tSim = 0; }
    const f = fig('yaw-cv', {
      ratio: 0.56, ratioNarrow: 0.7, minH: 260, maxH: 440, animate: () => run,
      draw(g, t, dt) {
        if (run && dt > 0) {
          const n = 10, h = dt / n;
          for (let i = 0; i < n; i++) {
            psiO = wrapRad(psiO + h * wOrig(deg(psiO)));
            psiE = wrapRad(psiE + h * wEdit(wrapRad(rad(sRef.value) - psiE)));
          }
          tSim += dt;
          if (tSim > 14) run = false;
        }
        const R = Math.min(g.w / 4 - 22, g.h / 2 - 48);
        const cy = g.h / 2 + 8;
        const dial = (cx, psi, col, title, attract) => {
          g.text(title, cx, 18, { size: g.narrow ? 11.5 : 13, align: 'center', col: col, weight: 500 });
          g.dot(cx, cy, R, null, C.line, 1.4);
          [0, 90, 180, -90].forEach(d => {
            const a = rad(d), c = Math.cos(a), s = Math.sin(a);
            g.line(cx + (R - 5) * c, cy - (R - 5) * s, cx + (R + 5) * c, cy - (R + 5) * s, C.faint, 1.2);
            if (d === 0) g.text('0°', cx + R + 9, cy, { size: 10.5, font: 'num', base: 'middle' });
            if (d === 90) g.text('90°', cx, cy - R - 9, { size: 10.5, font: 'num', align: 'center' });
          });
          if (attract) {
            const a = rad(-49), b = rad(144);
            g.dot(cx + R * Math.cos(a), cy - R * Math.sin(a), 5, C.orig);
            g.dot(cx + R * Math.cos(b), cy - R * Math.sin(b), 5, C.bg, C.orig, 1.6);
            g.text('−50°', cx + (R + 9) * Math.cos(a), cy - (R + 9) * Math.sin(a) + 4, { size: 10.5, font: 'num', col: C.orig, base: 'top' });
          }
          // reference needle
          const rr = rad(sRef.value);
          g.line(cx, cy, cx + (R - 2) * Math.cos(rr), cy - (R - 2) * Math.sin(rr), C.text, 1.6, [4, 5]);
          g.dot(cx + (R - 2) * Math.cos(rr), cy - (R - 2) * Math.sin(rr), 3, C.text);
          // drone (X frame) and nose
          const c = g.c, sz = R * 0.38; c.save(); c.translate(cx, cy); c.rotate(-psi);
          c.strokeStyle = C.muted; c.lineWidth = Math.max(1.5, sz * 0.09); c.lineCap = 'round';
          c.beginPath(); [Math.PI / 4, 3 * Math.PI / 4].forEach(q => { c.moveTo(-sz * Math.cos(q), -sz * Math.sin(q)); c.lineTo(sz * Math.cos(q), sz * Math.sin(q)); }); c.stroke();
          [1, 3, 5, 7].forEach(k => { const q = k * Math.PI / 4; c.beginPath(); c.arc(sz * Math.cos(q), sz * Math.sin(q), sz * 0.42, 0, 2 * Math.PI); c.fillStyle = C.panel; c.fill(); c.strokeStyle = C.faint; c.lineWidth = 1.2; c.stroke(); });
          c.beginPath(); c.arc(0, 0, sz * 0.2, 0, 2 * Math.PI); c.fillStyle = C.text; c.fill();
          c.restore();
          g.arrow(cx, cy, cx + R * 0.86 * Math.cos(psi), cy - R * 0.86 * Math.sin(psi), col, 3.2, 12);
          const err = wrapDeg(sRef.value - deg(psi));
          g.text('error ' + fmt(err, 1) + '°', cx, cy + R + 26, { size: 12, font: 'num', align: 'center', col: C.text });
        };
        dial(g.w * 0.25, psiO, C.orig, 'original (measured)', true);
        dial(g.w * 0.75, psiE, C.edit, 'edited (rewritten command)', false);
        $('yaw-ro').innerHTML = '<span class="ro">time<b>' + fmt(tSim, 1) + ' s</b></span><span class="ro">original nose<b>' + fmt(deg(psiO), 0) + '°</b></span><span class="ro">edited nose<b>' + fmt(deg(psiE), 0) + '°</b></span>';
      }
    });
    if (!f) return;
    sRef = slider(ctl, { label: 'Reference <i>ψ</i><sub>ref</sub>', min: -180, max: 180, step: 1, value: 90, fmt: v => fmt(v, 0) + '°', on: () => { f.redraw(); } });
    sStart = slider(ctl, { label: 'Start heading', min: -180, max: 180, step: 1, value: 20, fmt: v => fmt(v, 0) + '°', on: () => { run = false; reset(); f.redraw(); } });
    reset();
    $('yaw-go').addEventListener('click', () => { reset(); run = true; f.redraw(); wake(); });
  })();

  // ================================================================= 6. yaw sensor bumps
  (function bells() {
    const N = 23, alpha = rad(50), beta = 20, top = beta * (1 - Math.cos(alpha));
    const centres = Array.from({ length: N }, (_, k) => -Math.PI + 2 * Math.PI * k / N);
    let s, touched = false; let psi = rad(30);
    const f = fig('bell-cv', {
      ratio: 0.5, ratioNarrow: 1.35, minH: 300, maxH: 560, animate: () => !touched,
      draw(g, t, dt) {
        if (!touched && dt > 0) { psi = wrapRad(psi + dt * 0.35); if (s) s.set(Math.round(deg(psi)), true); }
        const phi = centres.map(c => elu(beta * (Math.cos(psi - c) - Math.cos(alpha))));
        const narrow = g.narrow;
        const R = narrow ? Math.min(g.w / 2 - 40, g.h * 0.25) : Math.min(g.w * 0.22, g.h / 2 - 34);
        const cx = narrow ? g.w / 2 : R + 40, cy = narrow ? R + 34 : g.h / 2;
        g.dot(cx, cy, R, null, C.line, 1.2);
        centres.forEach((c, i) => {
          const v = phi[i], on = clamp((v + 1) / (top + 1), 0, 1);
          const x = cx + R * Math.cos(c), y = cy - R * Math.sin(c);
          if (v > 0) g.line(x, y, cx + (R + 6 + 22 * v / top) * Math.cos(c), cy - (R + 6 + 22 * v / top) * Math.sin(c), C.circuit, 3.2);
          g.dot(x, y, 3 + 4 * on, v > 0 ? C.circuit : C.idle, v > 0 ? null : C.faint, 1);
        });
        // nose
        g.arrow(cx, cy, cx + R * 0.8 * Math.cos(psi), cy - R * 0.8 * Math.sin(psi), C.yaw, 3, 11);
        g.dot(cx, cy, 4, C.text);
        g.text('nose ψ', cx, cy + R + 30, { size: 12, align: 'center', col: C.yaw });
        // bar chart of the 23 activations
        const bx = narrow ? 44 : cx + R + 70, by = narrow ? cy + R + 58 : 30, bw = g.w - bx - 12, bh = narrow ? g.h - by - 34 : g.h - 66;
        const ax = g.axes({ x: bx, y: by, w: bw, h: bh, xr: [-180, 180], yr: [-1.5, top + 0.5], xt: [-180, -90, 0, 90, 180], yt: [0, 2, 4, 6], xl: 'centre of the unit ψᵢ [°]', yl: 'activation φᵢ' });
        const wbar = Math.max(2, bw / N * 0.62);
        centres.forEach((c, i) => {
          const x = ax.X(deg(c)), y0 = ax.Y(0), y1 = ax.Y(phi[i]);
          g.c.fillStyle = phi[i] > 0 ? C.circuit : C.idle; g.c.fillRect(x - wbar / 2, Math.min(y0, y1), wbar, Math.abs(y1 - y0));
        });
        ax.clip(() => { g.line(ax.X(deg(psi)), by, ax.X(deg(psi)), by + bh, C.yaw, 1.4, [4, 4]); });
      }
    });
    if (!f) return;
    s = slider($('bell-ctl'), { label: 'Nose <i>ψ</i>', min: -180, max: 180, step: 1, value: 30, fmt: v => fmt(v, 0) + '°', on: v => { touched = true; psi = rad(v); f.redraw(); } });
  })();

  // ================================================================= tables
  (function tables() {
    const m = DATA.mission, tb = $('yaw-tab');
    if (tb) tb.innerHTML = m.ref.map((r, i) => '<tr><td>step ' + (i + 2) + '</td><td>' + fmt(r, 0) + '°</td><td>' + m.jump[i] + '°</td><td class="o">' + m.orig[i].toFixed(1) + '°</td><td class="e">' +
      m.err[i][0].toFixed(2) + ' / ' + m.err[i][1].toFixed(2) + '°</td><td>' + m.t10[i][0].toFixed(1) + ' / ' + m.t10[i][1].toFixed(1) + ' s</td><td>' +
      m.pos[i][0].toFixed(1) + ' / ' + m.pos[i][1].toFixed(1) + ' cm</td><td>' + m.peak[i][0].toFixed(1) + ' / ' + m.peak[i][1].toFixed(1) + ' cm</td></tr>').join('');
    const ts = $('drone-tab');
    if (ts) ts.innerHTML = DATA.summary.map(r => '<tr><td>' + r[0] + '</td><td class="o">' + r[1] + '</td><td class="e">' + r[2] + '</td></tr>').join('');
  })();

  // ================================================================= start: sizes, visibility, nav
  function sizeAll() { figs.forEach(measure); wake(); }
  const io = 'IntersectionObserver' in window ? new IntersectionObserver(es => {
    es.forEach(e => { const f = figs.find(q => q.cv === e.target); if (f) { f.vis = e.isIntersecting; if (f.vis) { f.dirty = true; wake(); } } });
  }, { rootMargin: '120px' }) : null;
  figs.forEach(f => { measure(f); if (io) io.observe(f.cv); else f.vis = true; });
  if ('ResizeObserver' in window) { const ro = new ResizeObserver(() => sizeAll()); figs.forEach(f => ro.observe(f.cv.parentElement)); }
  else window.addEventListener('resize', sizeAll);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => figs.forEach(f => { f.dirty = true; wake(); }));
  wake();

  const links = Array.from(document.querySelectorAll('.top nav a'));
  if ('IntersectionObserver' in window) {
    const so = new IntersectionObserver(es => {
      es.forEach(e => { if (e.isIntersecting) links.forEach(a => a.classList.toggle('on', a.getAttribute('href') === '#' + e.target.id)); });
    }, { rootMargin: '-45% 0px -50% 0px' });
    document.querySelectorAll('main section[id]').forEach(s => so.observe(s));
  }
})();
