import React, { useEffect, useState } from 'react';

/* ══════════════════════════════════════════════════════════
   Aicraft Loading — Minimal & Elegant
   ══════════════════════════════════════════════════════════ */

const FADE_MS = 700;
const MIN_SHOW_MS = 2000;

export default function Root({ children }: { children: React.ReactNode }) {
  const [show, setShow] = useState(true);
  const [fade, setFade] = useState(false);

  useEffect(() => {
    const t1 = setTimeout(() => setFade(true), MIN_SHOW_MS);
    const t2 = setTimeout(() => setShow(false), MIN_SHOW_MS + FADE_MS);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, []);

  return (
    <>
      {show && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 99999,
            background: '#0d1117',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            opacity: fade ? 0 : 1,
            transition: `opacity ${FADE_MS}ms ease-out`,
            pointerEvents: fade ? 'none' : 'auto',
          }}
        >
          <style>{`
            @keyframes acFly {
              0% {
                transform: translateX(-100px) translateY(20px);
                opacity: 0;
              }
              15% {
                opacity: 1;
              }
              50% {
                transform: translateX(0) translateY(-10px);
              }
              85% {
                opacity: 1;
              }
              100% {
                transform: translateX(100px) translateY(20px);
                opacity: 0;
              }
            }
            @keyframes acTrailFade {
              0% { opacity: 0.4; transform: scaleX(0); }
              50% { opacity: 0.2; transform: scaleX(1); }
              100% { opacity: 0; transform: scaleX(0.5); }
            }
            @keyframes acPulseDot {
              0%, 100% { transform: scale(1); opacity: 0.3; }
              50% { transform: scale(1.2); opacity: 1; }
            }
          `}</style>

          {/* ── Plane ── */}
          <div style={{
            position: 'relative',
            width: 200,
            height: 80,
            marginBottom: 60,
          }}>
            {/* Trail */}
            <div style={{
              position: 'absolute',
              left: '50%',
              top: '50%',
              width: 60,
              height: 2,
              marginLeft: -50,
              marginTop: -1,
              background: 'linear-gradient(90deg, transparent, rgba(59,130,246,0.3))',
              borderRadius: 2,
              transformOrigin: 'right center',
              animation: 'acTrailFade 2.4s ease-in-out infinite',
            }} />

            {/* The plane */}
            <div style={{
              position: 'absolute',
              left: '50%',
              top: '50%',
              transform: 'translate(-50%, -50%)',
              animation: 'acFly 2.4s ease-in-out infinite',
            }}>
              <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                <path
                  d="M4 24L44 8L28 24L44 40L4 24Z"
                  fill="url(#planeGrad)"
                />
                <path
                  d="M28 24L44 8L44 40L28 24Z"
                  fill="rgba(59,130,246,0.3)"
                />
                <path
                  d="M4 24L28 20L28 28L4 24Z"
                  fill="rgba(255,255,255,0.9)"
                />
                <defs>
                  <linearGradient id="planeGrad" x1="4" y1="24" x2="44" y2="24" gradientUnits="userSpaceOnUse">
                    <stop stopColor="#ffffff" />
                    <stop offset="1" stopColor="#3b82f6" />
                  </linearGradient>
                </defs>
              </svg>
            </div>
          </div>

          {/* ── Logo ── */}
          <div style={{
            fontSize: 32,
            fontWeight: 800,
            letterSpacing: -1,
            color: '#e6edf3',
            marginBottom: 24,
          }}>
            Aicraft
          </div>

          {/* ── Loading dots ── */}
          <div style={{
            display: 'flex',
            gap: 8,
          }}>
            {[0, 1, 2].map(i => (
              <div
                key={i}
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: '50%',
                  background: '#3b82f6',
                  animation: `acPulseDot 1.2s ease-in-out infinite ${i * 0.15}s`,
                }}
              />
            ))}
          </div>
        </div>
      )}
      {children}
    </>
  );
}
