import React, { useEffect, useRef, useState, useMemo } from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import s from './index.module.css';

/* ════════════════════════════════════════════════════════
   Aicraft — Landing Page v2
   ════════════════════════════════════════════════════════ */

/* ── Scroll reveal ────────────────────────────────────── */
function useReveal(delay = 0) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          if (delay) setTimeout(() => el.classList.add(s.visible), delay);
          else el.classList.add(s.visible);
          io.unobserve(el);
        }
      },
      { threshold: 0.06, rootMargin: '0px 0px -40px 0px' },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [delay]);
  return ref;
}

function R({ children, className = '', delay = 0, as: Tag = 'div' }: any) {
  const ref = useReveal(delay);
  return <Tag ref={ref} className={`${s.reveal} ${className}`}>{children}</Tag>;
}

/* ── Animated number ──────────────────────────────────── */
function AnimNum({ value, suffix = '', label }: { value: number; suffix?: string; label: string }) {
  const [disp, setDisp] = useState('0');
  const ref = useRef<HTMLDivElement>(null);
  const ran = useRef(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting && !ran.current) {
        ran.current = true;
        const dur = 2000, t0 = performance.now();
        const tick = (now: number) => {
          const p = Math.min((now - t0) / dur, 1);
          const ease = 1 - Math.pow(1 - p, 4);
          setDisp(Math.round(ease * value).toLocaleString() + suffix);
          if (p < 1) requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
        io.unobserve(el);
      }
    }, { threshold: 0.3 });
    io.observe(el);
    return () => io.disconnect();
  }, [value, suffix]);
  return (
    <div ref={ref} className={s.statItem}>
      <strong className={s.statNum}>{disp}</strong>
      <span className={s.statLabel}>{label}</span>
    </div>
  );
}

/* ── Typing terminal ──────────────────────────────────── */
function useTyping(lines: string[], speed = 22) {
  const [output, setOutput] = useState<string[]>([]);
  const started = useRef(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting && !started.current) {
        started.current = true;
        let li = 0, ci = 0;
        const cur: string[] = [];
        const tick = () => {
          if (li >= lines.length) return;
          if (ci === 0) cur.push('');
          ci++;
          cur[li] = lines[li].substring(0, ci);
          setOutput([...cur]);
          if (ci >= lines[li].length) { li++; ci = 0; setTimeout(tick, 180); }
          else setTimeout(tick, speed);
        };
        setTimeout(tick, 500);
        io.unobserve(el);
      }
    }, { threshold: 0.2 });
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return { ref, output };
}

/* ── Parallax tilt on mouse ───────────────────────────── */
function useTilt(strength = 8) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const onMove = (e: MouseEvent) => {
      const r = el.getBoundingClientRect();
      const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
      const rx = ((e.clientY - cy) / (r.height / 2)) * -strength;
      const ry = ((e.clientX - cx) / (r.width / 2)) * strength;
      el.style.transform = `perspective(800px) rotateX(${rx}deg) rotateY(${ry}deg)`;
    };
    const onLeave = () => { el.style.transform = ''; };
    el.addEventListener('mousemove', onMove);
    el.addEventListener('mouseleave', onLeave);
    return () => { el.removeEventListener('mousemove', onMove); el.removeEventListener('mouseleave', onLeave); };
  }, [strength]);
  return ref;
}

/* ══════════════════════════════════════════════════════════
   HERO
   ══════════════════════════════════════════════════════════ */
function Hero() {
  const termLines = [
    '$ git clone https://github.com/TobiasTesauri/Aicraft.git',
    '$ cd Aicraft && gcc -O3 demo.c -I./include -o demo',
    '$ ./demo',
    '[ac] init .............. ok',
    '[ac] dense 784->128 .... ok',
    '[ac] dense 128->10 ..... ok',
    '[ac] forward ........... 0.42 ms',
    '[ac] backward .......... 0.87 ms',
    '[ac] loss: 0.0234 ...... ok',
    '[ac] cleanup ........... ok',
  ];
  const { ref: termRef, output } = useTyping(termLines, 16);
  const tiltRef = useTilt(6);

  return (
    <section className={s.hero}>
      <div className={s.heroNoise} />
      <div className={s.heroGrid} />
      <div className={s.heroOrb1} /><div className={s.heroOrb2} /><div className={s.heroOrb3} />

      <div className={s.heroInner}>
        {/* Left */}
        <div className={s.heroText}>
          <div className={`${s.heroBadge} ${s.stagger1}`}>
            <span className={s.badgePulse} />
            <span>PURE C</span>
            <span className={s.badgeSep}>|</span>
            <span>ZERO DEPS</span>
            <span className={s.badgeSep}>|</span>
            <span>MIT</span>
          </div>

          <h1 className={`${s.heroTitle} ${s.stagger2}`}>
            Machine learning,<br />
            <span className={s.heroGrad}>uncompromised.</span>
          </h1>

          <p className={`${s.heroSub} ${s.stagger3}`}>
            A complete deep-learning framework written entirely in pure C.
            SIMD-optimised, Vulkan-accelerated, header-only.
            From training to edge inference in a single&nbsp;
            <code className={s.inlineCode}>#include</code>.
          </p>

          <div className={`${s.heroCta} ${s.stagger4}`}>
            <Link className={s.btnPrimary} to="/docs/getting-started">
              Get started
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
            </Link>
            <Link className={s.btnOutline} to="/docs/api/overview">
              API Reference
            </Link>
            <Link className={s.btnGhost} href="https://github.com/TobiasTesauri/Aicraft">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12 .3a12 12 0 00-3.8 23.38c.6.12.82-.26.82-.58l-.01-2.05c-3.34.73-4.04-1.61-4.04-1.61a3.18 3.18 0 00-1.33-1.76c-1.09-.74.08-.73.08-.73a2.52 2.52 0 011.84 1.24 2.56 2.56 0 003.5 1 2.56 2.56 0 01.76-1.61c-2.66-.3-5.47-1.33-5.47-5.93a4.64 4.64 0 011.24-3.22 4.3 4.3 0 01.11-3.18s1-.33 3.3 1.23a11.4 11.4 0 016 0c2.28-1.56 3.29-1.23 3.29-1.23a4.3 4.3 0 01.12 3.18 4.64 4.64 0 011.23 3.22c0 4.61-2.81 5.63-5.48 5.92a2.87 2.87 0 01.82 2.23l-.01 3.3c0 .32.21.7.82.58A12 12 0 0012 .3"/></svg>
              GitHub
            </Link>
          </div>

          <div className={`${s.heroMeta} ${s.stagger4}`}>
            <span className={s.heroMetaItem}><span className={s.metaDot} style={{ background: '#555961' }} />C11</span>
            <span className={s.heroMetaItem}><span className={s.metaDot} style={{ background: '#22c55e' }} />MIT License</span>
            <span className={s.heroMetaItem}><span className={s.metaDot} style={{ background: '#3b82f6' }} />v1.0</span>
          </div>
        </div>

        {/* Right — terminal */}
        <div className={s.heroVisual} ref={termRef}>
          <div className={s.termWrap} ref={tiltRef}>
            <div className={s.termReflect} />
            <div className={s.termBar}>
              <span className={s.termDot} style={{ background: '#ff5f57' }} />
              <span className={s.termDot} style={{ background: '#febc2e' }} />
              <span className={s.termDot} style={{ background: '#28c840' }} />
              <span className={s.termTitle}>terminal</span>
            </div>
            <div className={s.termBody}>
              {output.map((line, i) => (
                <div key={i} className={s.termLine}>
                  {line.startsWith('$') ? (
                    <><span className={s.termPrompt}>{line.substring(0, 2)}</span>{line.substring(2)}</>
                  ) : line.includes('ok') ? (
                    <>{line.replace('ok', '')}<span className={s.termOk}>ok</span></>
                  ) : line.includes('ms') ? (
                    <>{line.replace(/[\d.]+ ms/, '')}<span className={s.termMs}>{line.match(/[\d.]+ ms/)?.[0]}</span></>
                  ) : line}
                </div>
              ))}
              <span className={s.termCursor} />
            </div>
          </div>
        </div>
      </div>

      <div className={s.scrollHint}>
        <div className={s.scrollMouse}><div className={s.scrollWheel} /></div>
      </div>
    </section>
  );
}

/* ══════════════════════════════════════════════════════════
   INSTALL STRIP
   ══════════════════════════════════════════════════════════ */
function Install() {
  const [copied, setCopied] = useState(false);
  const cmd = 'git clone https://github.com/TobiasTesauri/Aicraft.git && cd Aicraft';

  return (
    <section className={s.install}>
      <div className="container">
        <R>
          <div className={s.installCard}>
            <span className={s.installLabel}>Quick start</span>
            <div className={s.installRow}>
              <code className={s.installCmd}><span className={s.installPrompt}>$</span> {cmd}</code>
              <button
                className={s.installCopy}
                onClick={() => {
                  navigator.clipboard.writeText(cmd);
                  setCopied(true);
                  setTimeout(() => setCopied(false), 2000);
                }}
                aria-label="Copy command"
              >
                {copied ? (
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><path d="M5 13l4 4L19 7" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                ) : (
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none"><rect x="9" y="9" width="13" height="13" rx="2" stroke="currentColor" strokeWidth="1.5"/><path d="M5 15V5a2 2 0 012-2h10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>
                )}
              </button>
            </div>
            <span className={s.installHint}>
              Then <code>#include "aicraft/aicraft.h"</code> and compile. That's it.
            </span>
          </div>
        </R>
      </div>
    </section>
  );
}

/* ══════════════════════════════════════════════════════════
   STATS / NUMBERS
   ══════════════════════════════════════════════════════════ */
function Stats() {
  return (
    <section className={s.stats}>
      <div className="container">
        <R>
          <div className={s.statsGrid}>
            <AnimNum value={5000} suffix="+" label="Lines of C" />
            <AnimNum value={16} label="Header files" />
            <AnimNum value={14} label="GLSL shaders" />
            <AnimNum value={22} label="Autograd ops" />
            <AnimNum value={75} label="Test cases" />
            <AnimNum value={0} label="Dependencies" />
          </div>
        </R>
      </div>
    </section>
  );
}

/* ══════════════════════════════════════════════════════════
   CODE PREVIEW
   ══════════════════════════════════════════════════════════ */
function CodePreview() {
  const lines = [
    { n: 1, c: <><span className={s.cPP}>#include</span> <span className={s.cStr}>"aicraft/aicraft.h"</span></> },
    { n: 2, c: '' },
    { n: 3, c: <><span className={s.cType}>int</span> <span className={s.cFn}>main</span>(<span className={s.cType}>void</span>) {'{'}</> },
    { n: 4, c: <>{'    '}<span className={s.cFn}>ac_init</span>();</> },
    { n: 5, c: '' },
    { n: 6, c: <>{'    '}<span className={s.cCom}>// Build a feedforward network</span></> },
    { n: 7, c: <>{'    '}<span className={s.cType}>AcLayer</span> *net[] = {'{'}</> },
    { n: 8, c: <>{'        '}<span className={s.cFn}>ac_dense</span>(<span className={s.cNum}>784</span>, <span className={s.cNum}>128</span>, <span className={s.cK}>AC_RELU</span>),</> },
    { n: 9, c: <>{'        '}<span className={s.cFn}>ac_dense</span>(<span className={s.cNum}>128</span>, <span className={s.cNum}>10</span>,  <span className={s.cK}>AC_SOFTMAX</span>)</> },
    { n: 10, c: <>{'    '}{'};'}</> },
    { n: 11, c: '' },
    { n: 12, c: <>{'    '}<span className={s.cCom}>// Forward + backprop in one line</span></> },
    { n: 13, c: <>{'    '}<span className={s.cType}>AcTensor</span> *x = <span className={s.cFn}>ac_tensor_rand</span>((<span className={s.cType}>int</span>[]){'{'}<span className={s.cNum}>1</span>,<span className={s.cNum}>784</span>{'}'}, <span className={s.cNum}>2</span>);</> },
    { n: 14, c: <>{'    '}<span className={s.cType}>AcTensor</span> *y = <span className={s.cFn}>ac_forward_seq</span>(net, <span className={s.cNum}>2</span>, x);</> },
    { n: 15, c: <>{'    '}<span className={s.cFn}>ac_backward</span>(y);</> },
    { n: 16, c: '' },
    { n: 17, c: <>{'    '}<span className={s.cFn}>ac_cleanup</span>();</> },
    { n: 18, c: <>{'    '}<span className={s.cK}>return</span> <span className={s.cNum}>0</span>;</> },
    { n: 19, c: <>{'}'}</> },
  ];

  return (
    <section className={s.code}>
      <div className="container">
        <div className={s.codeLayout}>
          <R>
            <div className={s.codeInfo}>
              <p className={s.sLabel}>Dead simple</p>
              <h2 className={s.sTitle}>Include. Compile. Run.</h2>
              <p className={s.codeDesc}>
                No CMake, no vcpkg, no conan. Drop the header folder into your project,
                pass <code>-I./include</code>, and build. One translation unit, zero friction.
              </p>
              <div className={s.pills}>
                {['C11', 'Header-only', 'MIT License', 'Cross-platform', 'Embeddable'].map(t => (
                  <span key={t} className={s.pill}>{t}</span>
                ))}
              </div>
            </div>
          </R>
          <R delay={150}>
            <div className={s.codeWindow}>
              <div className={s.codeBar}>
                <span className={s.dot} style={{ background: '#ff5f57' }} />
                <span className={s.dot} style={{ background: '#febc2e' }} />
                <span className={s.dot} style={{ background: '#28c840' }} />
                <span className={s.codeFilename}>demo.c</span>
              </div>
              <div className={s.codeBody}>
                <div className={s.lineNums}>{lines.map(l => <span key={l.n}>{l.n}</span>)}</div>
                <pre className={s.codePre}><code>
                  {lines.map(l => (
                    <div key={l.n} className={`${s.codeLine} ${l.n === 8 ? s.codeActive : ''}`}>{l.c || '\n'}</div>
                  ))}
                </code></pre>
              </div>
            </div>
          </R>
        </div>
      </div>
    </section>
  );
}

/* ══════════════════════════════════════════════════════════
   ARCHITECTURE
   ══════════════════════════════════════════════════════════ */
function Architecture() {
  const stack = [
    { label: 'Your Application', sub: 'main.c', color: '#8b949e' },
    { label: 'aicraft.h', sub: 'Single include', color: '#3b82f6' },
    { label: 'Layers / Loss / Optimizer', sub: 'High-level API', color: '#a855f7' },
    { label: 'Autograd Engine', sub: '22 ops, DAG-based', color: '#a855f7' },
    { label: 'Tensor Core', sub: 'N-dim, broadcasting', color: '#06b6d4' },
  ];
  const backends = [
    { label: 'SIMD Kernels', sub: 'AVX-512 / NEON', color: '#22c55e' },
    { label: 'Vulkan Compute', sub: '14 GLSL shaders', color: '#22c55e' },
  ];

  const arrow = (
    <div className={s.archArrow}>
      <svg width="10" height="16" viewBox="0 0 10 16">
        <path d="M5 0v16M1 12l4 4 4-4" stroke="currentColor" strokeWidth="1.2" fill="none" strokeLinecap="round" strokeLinejoin="round" opacity="0.25"/>
      </svg>
    </div>
  );

  return (
    <section className={s.arch}>
      <div className="container">
        <R>
          <div className={s.archHead}>
            <p className={s.sLabel}>Architecture</p>
            <h2 className={s.sTitle}>Every layer, one file</h2>
            <p className={s.archDesc}>
              Aicraft is a vertically integrated stack. No external libraries sit between your code and the hardware.
            </p>
          </div>
        </R>
        <R delay={150}>
          <div className={s.archStack}>
            {stack.map((l, i) => (
              <React.Fragment key={i}>
                <div className={s.archBlock} style={{ '--bc': l.color } as any}>
                  <span className={s.archName}>{l.label}</span>
                  <span className={s.archSub}>{l.sub}</span>
                </div>
                {i < stack.length - 1 && arrow}
              </React.Fragment>
            ))}
            {arrow}
            <div className={s.archSplit}>
              {backends.map((b, i) => (
                <div key={i} className={s.archBlock} style={{ '--bc': b.color } as any}>
                  <span className={s.archName}>{b.label}</span>
                  <span className={s.archSub}>{b.sub}</span>
                </div>
              ))}
            </div>
            {arrow}
            <div className={s.archBlock} style={{ '--bc': '#f59e0b' } as any}>
              <span className={s.archName}>Arena Allocator</span>
              <span className={s.archSub}>Checkpoint / restore memory</span>
            </div>
          </div>
        </R>
      </div>
    </section>
  );
}

/* ══════════════════════════════════════════════════════════
   FEATURES
   ══════════════════════════════════════════════════════════ */
const ICONS = {
  bolt: <svg viewBox="0 0 24 24" fill="none" width="26" height="26"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>,
  gpu: <svg viewBox="0 0 24 24" fill="none" width="26" height="26"><rect x="4" y="4" width="16" height="16" rx="2" stroke="currentColor" strokeWidth="1.5"/><path d="M9 9h6M9 12h6M9 15h3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>,
  auto: <svg viewBox="0 0 24 24" fill="none" width="26" height="26"><circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.5"/><path d="M12 2v4M12 18v4M2 12h4M18 12h4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>,
  cube: <svg viewBox="0 0 24 24" fill="none" width="26" height="26"><path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z" stroke="currentColor" strokeWidth="1.5"/><path d="M3.27 6.96L12 12.01l8.73-5.05M12 22.08V12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>,
  mem: <svg viewBox="0 0 24 24" fill="none" width="26" height="26"><rect x="3" y="6" width="18" height="12" rx="2" stroke="currentColor" strokeWidth="1.5"/><path d="M7 2v4M12 2v4M17 2v4M7 18v4M12 18v4M17 18v4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>,
  chart: <svg viewBox="0 0 24 24" fill="none" width="26" height="26"><path d="M4 17l6-6 4 4 6-6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/><path d="M15 7h5v5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>,
};

function Features() {
  const feats = [
    { icon: ICONS.bolt, title: 'SIMD Vectorised', text: 'AVX2, AVX-512, ARM NEON. Every hot path hand-tuned with platform intrinsics and BLIS-style GEMM micro-kernels.', accent: '#3b82f6' },
    { icon: ICONS.gpu, title: 'Vulkan Compute', text: '14 GLSL compute shaders for GEMM, activations, and reductions. Cross-vendor GPU acceleration.', accent: '#a855f7' },
    { icon: ICONS.auto, title: 'Autograd Engine', text: '22 differentiable ops. Dynamic computational graph with reverse-mode autodiff and O(1) cycle detection.', accent: '#06b6d4' },
    { icon: ICONS.cube, title: 'INT8 Quantisation', text: 'Post-training quantisation with asymmetric per-tensor scaling. ~4x model compression for edge.', accent: '#f59e0b' },
    { icon: ICONS.mem, title: 'Arena Allocator', text: 'Checkpoint/restore memory management. Zero per-tensor malloc. Constant memory in training.', accent: '#22c55e' },
    { icon: ICONS.chart, title: 'Training Loop', text: 'SGD, Adam, AdamW optimisers. Cross-entropy, MSE, Huber loss. Full training pipeline.', accent: '#ef4444' },
  ];

  return (
    <section className={s.features}>
      <div className="container">
        <R>
          <div className={s.featHead}>
            <p className={s.sLabel}>Capabilities</p>
            <h2 className={s.sTitle}>Built for performance,<br />designed for simplicity</h2>
          </div>
        </R>
        <div className={s.featGrid}>
          {feats.map((f, i) => (
            <R key={f.title} delay={i * 80}>
              <div className={s.featCard} style={{ '--accent': f.accent } as any}>
                <div className={s.featIcon}>{f.icon}</div>
                <h3>{f.title}</h3>
                <p>{f.text}</p>
                <div className={s.featShine} />
              </div>
            </R>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ══════════════════════════════════════════════════════════
   COMPARISON TABLE
   ══════════════════════════════════════════════════════════ */
function Comparison() {
  const rows: [string, string, string, string][] = [
    ['Binary size',  '~150 KB',        '~800 MB',   '~1.8 GB'],
    ['Dependencies', '0',              '~50',       '~80'],
    ['Language',     'C11',            'C++ / Py',  'C++ / Py'],
    ['GPU backend',  'Vulkan',         'CUDA',      'CUDA'],
    ['SIMD',         'Hand-tuned',     'Generic',   'Generic'],
    ['Memory',       'Arena allocator','malloc/free','Custom'],
    ['Edge deploy',  'MCU-ready',      'No',        'TFLite'],
  ];

  return (
    <section className={s.comp}>
      <div className="container">
        <R>
          <div className={s.compHead}>
            <p className={s.sLabel}>How it compares</p>
            <h2 className={s.sTitle}>Minimal footprint, maximum control</h2>
          </div>
        </R>
        <R delay={100}>
          <div className={s.compTable}>
            <div className={`${s.compRow} ${s.compHeader}`}>
              <span />
              <span className={s.compBrand}>Aicraft</span>
              <span>PyTorch</span>
              <span>TensorFlow</span>
            </div>
            {rows.map(([m, ac, pt, tf]) => (
              <div key={m} className={s.compRow}>
                <span className={s.compMetric}>{m}</span>
                <span className={`${s.compVal} ${s.compBrand}`}>{ac}</span>
                <span className={s.compVal}>{pt}</span>
                <span className={s.compVal}>{tf}</span>
              </div>
            ))}
          </div>
        </R>
      </div>
    </section>
  );
}

/* ══════════════════════════════════════════════════════════
   TESTIMONIAL / PHILOSOPHY
   ══════════════════════════════════════════════════════════ */
function Philosophy() {
  return (
    <section className={s.philo}>
      <div className="container">
        <R>
          <div className={s.philoInner}>
            <svg className={s.philoQuote} viewBox="0 0 24 24" fill="currentColor" width="40" height="40" opacity="0.15">
              <path d="M14.017 21v-7.391c0-5.704 3.731-9.57 8.983-10.609l.995 2.151c-2.432.917-3.995 3.638-3.995 5.849h4v10H14.017zM0 21v-7.391C0 7.905 3.748 4.039 9 3l.996 2.151C7.563 6.068 5.999 8.789 5.999 11h4.011v10H0z"/>
            </svg>
            <blockquote className={s.philoText}>
              The best dependency is the one you never add. Aicraft proves you can train
              a neural network without pulling half the internet into your build.
            </blockquote>
            <div className={s.philoAuthor}>
              <strong>Tobias Tesauri</strong>
              <span>Creator of Aicraft &mdash; T&amp;M Softwares</span>
            </div>
          </div>
        </R>
      </div>
    </section>
  );
}

/* ══════════════════════════════════════════════════════════
   WORKFLOW / HOW IT WORKS
   ══════════════════════════════════════════════════════════ */
function Workflow() {
  const steps = [
    { num: '01', title: 'Include', text: 'Add the single header to your C project. No build system changes needed.', icon: <svg viewBox="0 0 24 24" fill="none" width="22" height="22"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/><path d="M14 2v6h6M16 13H8M16 17H8M10 9H8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg> },
    { num: '02', title: 'Define', text: 'Stack layers, pick a loss function and optimiser. Just like Python, but in C.', icon: <svg viewBox="0 0 24 24" fill="none" width="22" height="22"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87L18.18 22 12 18.56 5.82 22 7 14.14l-5-4.87 6.91-1.01L12 2z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg> },
    { num: '03', title: 'Train', text: 'Forward, backward, step. The autograd engine handles gradient computation.', icon: <svg viewBox="0 0 24 24" fill="none" width="22" height="22"><path d="M4 17l6-6 4 4 6-6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/><path d="M15 7h5v5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg> },
    { num: '04', title: 'Deploy', text: 'Quantise to INT8, serialise, and run on anything from x86 to ARM Cortex-M.', icon: <svg viewBox="0 0 24 24" fill="none" width="22" height="22"><path d="M22 11.08V12a10 10 0 11-5.93-9.14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/><path d="M22 4L12 14.01l-3-3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg> },
  ];

  return (
    <section className={s.workflow}>
      <div className="container">
        <R>
          <div className={s.wfHead}>
            <p className={s.sLabel}>How it works</p>
            <h2 className={s.sTitle}>Four steps to production</h2>
          </div>
        </R>
        <div className={s.wfGrid}>
          {steps.map((st, i) => (
            <R key={st.num} delay={i * 120}>
              <div className={s.wfCard}>
                <span className={s.wfNum}>{st.num}</span>
                <div className={s.wfIcon}>{st.icon}</div>
                <h3>{st.title}</h3>
                <p>{st.text}</p>
              </div>
            </R>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ══════════════════════════════════════════════════════════
   DOCUMENTATION MAP
   ══════════════════════════════════════════════════════════ */
function DocMap() {
  const cards = [
    { title: 'Getting Started', desc: 'Clone, compile, run your first model in 5 minutes.', to: '/docs/getting-started', icon: <svg viewBox="0 0 24 24" fill="none" width="22" height="22"><path d="M5 12h14M12 5l7 7-7 7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg> },
    { title: 'Architecture', desc: 'Internal design: tensors, autograd, memory model.', to: '/docs/architecture', icon: <svg viewBox="0 0 24 24" fill="none" width="22" height="22"><rect x="3" y="3" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="1.5"/><rect x="14" y="3" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="1.5"/><rect x="3" y="14" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="1.5"/><rect x="14" y="14" width="7" height="7" rx="1" stroke="currentColor" strokeWidth="1.5"/></svg> },
    { title: 'API Reference', desc: 'All tensors, layers, loss, optimiser functions.', to: '/docs/api/overview', icon: <svg viewBox="0 0 24 24" fill="none" width="22" height="22"><path d="M4 19.5A2.5 2.5 0 016.5 17H20" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z" stroke="currentColor" strokeWidth="1.5"/></svg> },
    { title: 'Training Guide', desc: 'Forward, backward, optimiser step, full pipeline.', to: '/docs/guides/training', icon: <svg viewBox="0 0 24 24" fill="none" width="22" height="22"><path d="M4 17l6-6 4 4 6-6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/><path d="M15 7h5v5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg> },
    { title: 'Vulkan GPU', desc: 'GPU pipeline: device selection, shader dispatch.', to: '/docs/guides/vulkan', icon: <svg viewBox="0 0 24 24" fill="none" width="22" height="22"><rect x="2" y="6" width="20" height="12" rx="2" stroke="currentColor" strokeWidth="1.5"/><path d="M6 10h4M6 14h3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg> },
    { title: 'Edge Deployment', desc: 'Quantise, serialise, run on MCUs and ARM boards.', to: '/docs/guides/edge-deployment', icon: <svg viewBox="0 0 24 24" fill="none" width="22" height="22"><rect x="5" y="2" width="14" height="20" rx="2" stroke="currentColor" strokeWidth="1.5"/><path d="M12 18h.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/></svg> },
  ];

  return (
    <section className={s.docmap}>
      <div className="container">
        <R>
          <div className={s.docmapHead}>
            <p className={s.sLabel}>Explore</p>
            <h2 className={s.sTitle}>Everything you need</h2>
          </div>
        </R>
        <div className={s.docGrid}>
          {cards.map((c, i) => (
            <R key={c.title} delay={i * 70}>
              <Link to={c.to} className={s.docCard}>
                <div className={s.docIcon}>{c.icon}</div>
                <div className={s.docBody}>
                  <h3>{c.title}</h3>
                  <p>{c.desc}</p>
                </div>
                <svg className={s.docArrow} viewBox="0 0 16 16" fill="none" width="16" height="16">
                  <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
                </svg>
              </Link>
            </R>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ══════════════════════════════════════════════════════════
   FINAL CTA
   ══════════════════════════════════════════════════════════ */
function CTA() {
  return (
    <section className={s.cta}>
      <div className={s.ctaGlow} />
      <div className="container" style={{ position: 'relative' }}>
        <R>
          <div className={s.ctaInner}>
            <p className={s.sLabel}>Open source</p>
            <h2 className={s.ctaTitle}>
              Ready to see what<br />pure C can do?
            </h2>
            <p className={s.ctaSub}>
              Read the docs, explore the source, or start building.
            </p>
            <div className={s.ctaBtns}>
              <Link className={s.btnPrimary} to="/docs/getting-started">
                Get started
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
              </Link>
              <Link className={s.btnOutline} href="https://github.com/TobiasTesauri/Aicraft">
                View on GitHub
              </Link>
            </div>
            <p className={s.ctaCredit}>
              A project by <strong>Tobias Tesauri</strong> — <a href="https://tmsoftwares.eu" target="_blank" rel="noopener noreferrer">T&amp;M Softwares</a>
            </p>
          </div>
        </R>
      </div>
    </section>
  );
}

/* ══════════════════════════════════════════════════════════
   PAGE
   ══════════════════════════════════════════════════════════ */
export default function Home(): React.JSX.Element {
  return (
    <Layout
      title="Home"
      description="Aicraft — Pure C machine-learning framework. Zero dependencies, SIMD-optimised, Vulkan-accelerated."
    >
      <Hero />
      <Install />
      <Stats />
      <CodePreview />
      <Architecture />
      <Features />
      <Workflow />
      <Comparison />
      <Philosophy />
      <DocMap />
      <CTA />
    </Layout>
  );
}
