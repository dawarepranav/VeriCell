import React from "react";
import {
  AbsoluteFill,
  Easing,
  Img,
  OffthreadVideo,
  Sequence,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { Audio } from "@remotion/media";
import { loadFont as loadSans } from "@remotion/google-fonts/IBMPlexSans";
import { loadFont as loadSerif } from "@remotion/google-fonts/IBMPlexSerif";
import { loadFont as loadMono } from "@remotion/google-fonts/IBMPlexMono";
import timeline from "./timeline.json";

const sans = loadSans("normal", { weights: ["400", "500", "600"] }).fontFamily;
const serif = loadSerif("normal", { weights: ["500", "600"] }).fontFamily;
const mono = loadMono("normal", { weights: ["500", "600"] }).fontFamily;

const C = { bg: "#070a0f", ink: "#ece7dd", muted: "#9aa4b0", line: "#2a313b", cyan: "#7fd3e6", blue: "#7fa8e6", amber: "#e0a33a", green: "#5cc28a" };
const W = 1920, H = 1080;
type Rect = [number, number, number, number];
type Cam = { t: number; rect: Rect | null; fill?: number; dur?: number };
type Seg = { from: number; dur: number; src: number; rate: number };

const ease = Easing.bezier(0.45, 0, 0.2, 1);
const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;

// ---------------------------------------------------------------- camera
function camFor(rect: Rect | null, fill = 0.85) {
  if (!rect) return { s: 1, x: 0, y: 0 };
  let s = Math.min((W * fill) / rect[2], (H * fill) / rect[3], 2.6);
  s = Math.max(s, 1);
  let x = W / 2 - (rect[0] + rect[2] / 2) * s, y = H / 2 - (rect[1] + rect[3] / 2) * s;
  x = Math.min(0, Math.max(W - W * s, x));
  y = Math.min(0, Math.max(H - H * s, y));
  return { s, x, y };
}
function useCam(cams: Cam[], t: number) {
  let cur = camFor(null), prev = cur, k = 0, at = 0, dur = 0.6;
  for (const c of cams) {
    if (t < c.t) break;
    prev = cur;
    cur = camFor(c.rect, c.fill ?? 0.85);
    at = c.t;
    dur = c.dur ?? 0.6;
  }
  k = ease(Math.min(1, Math.max(0, (t - at) / dur)));
  const push = 1 + Math.min(0.025, (t - at) * 0.004); // a slow, gentle push while holding a framing
  const s = (prev.s + (cur.s - prev.s) * k) * push;
  const cx = W / 2, cy = H / 2;
  return { s, x: (prev.x + (cur.x - prev.x) * k) * push + cx * (1 - push), y: (prev.y + (cur.y - prev.y) * k) * push + cy * (1 - push) };
}

// ---------------------------------------------------------------- footage with camera, cursor, rings
const Footage: React.FC<{ segs: Seg[]; cams: Cam[]; rings: any[]; cursor: any[]; clicks: any[]; dim?: number }> = ({ segs, cams, rings, cursor, clicks, dim = 0 }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const cam = useCam(cams, t);
  const xf = (x: number) => cam.x + x * cam.s, yf = (y: number) => cam.y + y * cam.s;
  // cursor: latest move, eased; hidden before the first move of the scene
  let cx = -100, cy = -100, shown = 0;
  for (const m of cursor) {
    if (t < m.t) break;
    const k = ease(Math.min(1, (t - m.t) / Math.max(0.05, m.dur)));
    cx = m.x0 + (m.x - m.x0) * k; cy = m.y0 + (m.y - m.y0) * k; shown = 1;
  }
  return (
    <AbsoluteFill style={{ overflow: "hidden", background: C.bg }}>
      <AbsoluteFill style={{ transformOrigin: "0 0", transform: `translate(${cam.x}px, ${cam.y}px) scale(${cam.s})` }}>
        {segs.map((s, i) => (
          <Sequence key={i} name={`footage ${s.src.toFixed(1)}s`} from={Math.round(s.from * fps)} durationInFrames={Math.max(1, Math.round(s.dur * fps))} layout="none">
            <OffthreadVideo src={staticFile("footage.mp4")} trimBefore={Math.round(s.src * fps)} playbackRate={s.rate} muted style={{ width: W, height: H, position: "absolute" }} />
          </Sequence>
        ))}
      </AbsoluteFill>
      {dim > 0 && <AbsoluteFill style={{ background: `rgba(5,8,12,${dim})` }} />}
      {rings.map((r, i) => {
        const a = interpolate(t, [r.t, r.t + 0.3, r.t + r.dur - 0.3, r.t + r.dur], [0, 1, 1, 0], clamp);
        const pulse = r.pulse ? 1 + 0.04 * Math.sin((t - r.t) * 5) : 1;
        const pad = 10, x = xf(r.rect[0] - pad), y = yf(r.rect[1] - pad), w = (r.rect[2] + 2 * pad) * cam.s, h = (r.rect[3] + 2 * pad) * cam.s;
        return (
          <div key={i} style={{ position: "absolute", left: x, top: y, width: w, height: h, opacity: a, transform: `scale(${pulse})`,
            border: `4px solid ${r.color ?? C.amber}`, borderRadius: r.round ? "50%" : 14,
            boxShadow: `0 0 0 4000px rgba(4,6,10,${0.38 * a}), 0 0 36px ${r.color ?? C.amber}66` }} />
        );
      })}
      {clicks.map((c, i) => {
        const k = (t - c.t) / 0.45;
        if (k < 0 || k > 1) return null;
        return <div key={i} style={{ position: "absolute", left: xf(c.x) - 22, top: yf(c.y) - 22, width: 44, height: 44, borderRadius: 22,
          border: "3px solid rgba(255,255,255,.9)", opacity: 1 - k, transform: `scale(${0.6 + k * 1.6})` }} />;
      })}
      {shown ? (
        <svg style={{ position: "absolute", left: xf(cx) - 3, top: yf(cy) - 2, width: 30, height: 30, filter: "drop-shadow(0 2px 4px rgba(0,0,0,.6))" }} viewBox="0 0 24 24">
          <path d="M3 2 L3 19 L7.5 14.8 L10.4 21.5 L13.2 20.3 L10.4 13.8 L16.5 13.8 Z" fill="#fff" stroke="#111" strokeWidth="1.2" strokeLinejoin="round" />
        </svg>
      ) : null}
    </AbsoluteFill>
  );
};

// ---------------------------------------------------------------- typography
const Rise: React.FC<{ at: number; children: React.ReactNode; style?: React.CSSProperties }> = ({ at, children, style }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const k = spring({ frame: frame - at * fps, fps, config: { damping: 200 }, durationInFrames: 0.6 * fps });
  return <div style={{ opacity: k, transform: `translateY(${(1 - k) * 22}px)`, ...style }}>{children}</div>;
};

const Callout: React.FC<{ c: any }> = ({ c }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const a = interpolate(t, [c.t, c.t + 0.3, c.t + c.dur - 0.3, c.t + c.dur], [0, 1, 1, 0], clamp);
  const x = interpolate(t, [c.t, c.t + 0.35], [-24, 0], { ...clamp, easing: ease });
  const pos: React.CSSProperties = c.pos === "top" ? { left: 72, top: 64 } : { left: 72, bottom: 150 };
  return (
    <div style={{ position: "absolute", ...pos, opacity: a, transform: `translateX(${x}px)`, display: "flex", background: "rgba(8,11,16,.9)",
      border: `1px solid ${C.line}`, borderRadius: 14, overflow: "hidden", boxShadow: "0 18px 50px rgba(0,0,0,.45)" }}>
      <div style={{ width: 8, background: c.color ?? C.blue }} />
      <div style={{ padding: "16px 26px" }}>
        <div style={{ font: `600 34px ${sans}`, color: C.ink }}>{c.title}</div>
        {c.sub && <div style={{ font: `400 24px ${sans}`, color: C.muted, marginTop: 4 }}>{c.sub}</div>}
      </div>
    </div>
  );
};

const Captions: React.FC<{ chunks: { t: number; end: number; text: string }[] }> = ({ chunks }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const c = chunks.find((x) => t >= x.t && t < x.end);
  if (!c) return null;
  const a = interpolate(t, [c.t, c.t + 0.15], [0, 1], clamp);
  return (
    <div style={{ position: "absolute", left: 0, right: 0, bottom: 46, display: "flex", justifyContent: "center", opacity: a }}>
      <div style={{ maxWidth: 1400, font: `500 34px ${sans}`, color: "#fff", background: "rgba(4,6,10,.72)", padding: "8px 20px", borderRadius: 10, textAlign: "center" }}>{c.text}</div>
    </div>
  );
};

// ---------------------------------------------------------------- scene-specific layers
const Hook: React.FC<{ d: number; textAt: number }> = ({ d, textAt }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const s = interpolate(t, [0, d], [1.0, 1.14]);
  return (
    <AbsoluteFill style={{ background: C.bg, overflow: "hidden" }}>
      <Img src={staticFile("colon.png")} style={{ width: W, height: H, objectFit: "cover", transform: `scale(${s})`, transformOrigin: "62% 58%" }} />
      <AbsoluteFill style={{ background: "radial-gradient(1400px 800px at 50% 40%, transparent 30%, rgba(4,6,10,.55) 100%)" }} />
      <Rise at={textAt} style={{ position: "absolute", left: 90, bottom: 150, background: "rgba(6,9,13,.82)", padding: "26px 36px", borderRadius: 16 }}>
        <div style={{ font: `600 54px ${serif}`, color: C.ink }}>About 1 in 5 cells missed or mis-drawn.</div>
        <div style={{ font: `600 54px ${serif}`, color: C.amber }}>No warning.</div>
        <div style={{ font: `400 24px ${sans}`, color: C.muted, marginTop: 8 }}>The strongest general segmenter we tested (Cellpose-SAM) on LIVECell</div>
      </Rise>
    </AbsoluteFill>
  );
};

const Lockup: React.FC<{ at: number; until: number }> = ({ at, until }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const a = interpolate(t, [at, at + 0.4, until - 0.4, until], [0, 1, 1, 0], clamp);
  return (
    <div style={{ position: "absolute", left: 72, bottom: 140, opacity: a, background: "rgba(7,10,15,.9)", border: `1px solid ${C.line}`, borderRadius: 16, padding: "22px 30px", boxShadow: "0 20px 60px rgba(0,0,0,.5)" }}>
      <div style={{ font: `600 22px ${mono}`, color: C.amber, letterSpacing: ".05em" }}>BUILT DURING THE META AI DEVELOPER HACKATHON 2026</div>
      <div style={{ display: "flex", gap: 12, marginTop: 14 }}>
        {["Muse Spark", "SAM 3.1", "Muse Voice", "Muse Spark Open"].map((m, i) => (
          <Rise key={m} at={at + 0.3 + i * 0.18}><span style={{ font: `600 26px ${sans}`, color: C.blue, border: `1px solid ${C.line}`, borderRadius: 999, padding: "8px 18px" }}>{m}</span></Rise>
        ))}
      </div>
    </div>
  );
};

const Window: React.FC<{ children: React.ReactNode; x: number; y: number; w: number }> = ({ children, x, y, w }) => (
  <div style={{ position: "absolute", left: x, top: y, width: w, height: (w * 9) / 16, borderRadius: 18, overflow: "hidden", border: `1px solid ${C.line}`, boxShadow: "0 40px 120px rgba(0,0,0,.65)" }}>
    <div style={{ width: W, height: H, transform: `scale(${w / W})`, transformOrigin: "0 0" }}>{children}</div>
  </div>
);

const Backdrop: React.FC = () => {
  const frame = useCurrentFrame();
  const gx = 60 + 10 * Math.sin(frame / 90), gy = 30 + 8 * Math.cos(frame / 110);
  return <AbsoluteFill style={{ background: `radial-gradient(1200px 800px at ${gx}% ${gy}%, #132235 0%, ${C.bg} 62%)` }} />;
};

const Panel: React.FC<{ p: any }> = ({ p }) => (
  <div style={{ position: "absolute", left: p.x ?? 1290, top: 0, bottom: 0, width: p.w ?? 560, display: "flex", flexDirection: "column", justifyContent: "center", gap: 22 }}>
    <Rise at={p.t}><div style={{ font: `600 22px ${mono}`, color: C.amber, letterSpacing: ".06em" }}>{p.kicker}</div></Rise>
    <Rise at={p.t + 0.2}><div style={{ font: `600 48px/1.12 ${serif}`, color: C.ink }}>{p.title}</div></Rise>
    {p.lines.map((l: any, i: number) => (
      <Rise key={i} at={p.t + 0.5 + i * 0.35}>
        <div style={{ font: l.big ? `600 44px ${mono}` : `400 27px/1.4 ${sans}`, color: l.color ? (C as any)[l.color] : C.muted }}>{l.text}</div>
      </Rise>
    ))}
  </div>
);

const Truth: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const before = Math.round(interpolate(t, [0.4, 1.6], [0, 7], { ...clamp, easing: ease }));
  const after = spring({ frame: frame - 2.1 * fps, fps, config: { damping: 14 } });
  return (
    <AbsoluteFill style={{ justifyContent: "center", alignItems: "center" }}>
      <div style={{ background: "rgba(7,10,15,.92)", border: `1px solid ${C.line}`, borderRadius: 24, padding: "50px 70px", boxShadow: "0 40px 120px rgba(0,0,0,.6)", width: 1300 }}>
        <Rise at={0.1}><div style={{ font: `600 46px ${serif}`, color: C.ink }}>Against hand-measured ground truth</div></Rise>
        <div style={{ display: "flex", gap: 90, marginTop: 34, alignItems: "flex-end" }}>
          <Rise at={0.4}><div style={{ font: `600 150px ${serif}`, color: C.muted, lineHeight: 0.9 }}>{before}%</div>
            <div style={{ font: `400 26px ${sans}`, color: C.muted, marginTop: 12 }}>off before the checks · <span style={{ fontFamily: mono }}>8,240 px²</span></div></Rise>
          <div style={{ opacity: after, transform: `scale(${0.85 + 0.15 * after})`, transformOrigin: "left bottom" }}>
            <div style={{ font: `600 150px ${serif}`, color: C.green, lineHeight: 0.9 }}>&lt;1%</div>
            <div style={{ font: `400 26px ${sans}`, color: C.muted, marginTop: 12 }}>off after four checks · <span style={{ fontFamily: mono }}>8,772</span> vs truth <span style={{ fontFamily: mono }}>8,818 px²</span></div></div>
        </div>
      </div>
    </AbsoluteFill>
  );
};

const ROWS = [["Ask a question in plain words", "y", "p", "n", "y", "y"], ["Outlines cells and organoids", "y", "y", "y", "y", "y"],
  ["Checks every object it found", "y", "n", "n", "n", "p"], ["Calibrated 90% range on answers", "y", "n", "n", "n", "n"],
  ["Says when the conclusion could change", "y", "n", "n", "n", "n"], ["Asks you about the objects that matter", "y", "n", "n", "n", "n"]];
const Compare: React.FC = () => {
  const mark = (m: string) => m === "y" ? <span style={{ color: C.green, fontWeight: 600 }}>✓</span> : m === "p" ? <span style={{ color: C.amber }}>partial</span> : <span style={{ color: "#4d555f" }}>—</span>;
  return (
    <AbsoluteFill style={{ padding: "110px 140px" }}>
      <Rise at={0.1}><div style={{ font: `600 54px ${serif}`, color: C.ink }}>Tools that segment, and tools that answer.<br /><span style={{ color: C.cyan }}>Only one that doubts itself.</span></div></Rise>
      <table style={{ marginTop: 40, borderCollapse: "collapse", font: `400 27px ${sans}`, color: C.ink, width: 1640 }}>
        <thead><Rise at={0.6} style={{ display: "contents" }}><tr>{["", "VeriCell", "napari", "CellProfiler", "MorphoOrgaAgent", "GenCellAgent"].map((h, i) => (
          <th key={i} style={{ font: `600 24px ${sans}`, color: i === 1 ? C.ink : C.muted, padding: "12px 10px", textAlign: i ? "center" : "left", background: i === 1 ? "rgba(127,211,230,.1)" : "none", borderRadius: "12px 12px 0 0" }}>{h}</th>))}</tr></Rise></thead>
        <tbody>{ROWS.map((r, i) => (
          <tr key={i}>{r.map((c, j) => (
            <td key={j} style={{ padding: "14px 10px", borderTop: `1px solid ${C.line}`, textAlign: j ? "center" : "left", background: j === 1 ? "rgba(127,211,230,.06)" : "none" }}>
              <Rise at={1.0 + i * 0.9 + (j ? 0.15 : 0)}>{j ? mark(c) : c}</Rise></td>))}</tr>))}</tbody>
      </table>
      <Rise at={7.5}><div style={{ font: `400 19px ${mono}`, color: C.muted, marginTop: 22, maxWidth: 1640 }}>From each tool's documentation and papers, Nov 2026. napari: questions only via the napari-chatgpt (Omega) plugin. GenCellAgent: an evaluator its authors call "prone to overconfidence".</div></Rise>
    </AbsoluteFill>
  );
};

const BENCH: [string, number, number][] = [["Largest area", 16.8, 3.6], ["Area, tiers", 38.7, 28.1], ["Longest perimeter", 15.4, 7.3], ["Perimeter, tiers", 22.0, 16.5],
  ["Roughest edge", 46.0, 38.9], ["Roughness, tiers", 31.8, 23.8]];
const Proof: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  return (
    <AbsoluteFill style={{ padding: "100px 140px" }}>
      <Rise at={0.1}><div style={{ font: `600 54px ${serif}`, color: C.ink }}>Measured, not claimed.</div></Rise>
      <div style={{ display: "grid", gridTemplateColumns: "1.25fr 1fr", gap: 90, marginTop: 40 }}>
        <div>
          <Rise at={0.4}><div style={{ font: `500 25px ${sans}`, color: C.muted, marginBottom: 18 }}>Median error vs the published organoid agent · <b style={{ color: C.ink }}>16 / 16</b> question types better · 68 images</div></Rise>
          {BENCH.map(([q, pub, ours], i) => {
            const k = interpolate(t, [0.8 + i * 0.25, 1.8 + i * 0.25], [0, 1], { ...clamp, easing: ease });
            return (
              <div key={q} style={{ display: "grid", gridTemplateColumns: "250px 1fr 90px", alignItems: "center", gap: 14, marginBottom: 16, opacity: Math.min(1, k * 3) }}>
                <div style={{ font: `400 23px ${sans}`, color: C.ink }}>{q}</div>
                <div>
                  <div style={{ height: 10, borderRadius: 5, width: `${(pub / 50) * 100 * k}%`, background: "#3a424d" }} />
                  <div style={{ height: 10, borderRadius: 5, width: `${(ours / 50) * 100 * k}%`, background: C.cyan, marginTop: 6 }} />
                </div>
                <div style={{ font: `600 22px ${mono}`, color: C.cyan }}>{(ours * k).toFixed(1)}%</div>
              </div>
            );
          })}
          <Rise at={2.6}><div style={{ font: `400 20px ${sans}`, color: C.muted }}><span style={{ color: "#8a929c" }}>■</span> MorphoOrgaAgent (published) &nbsp; <span style={{ color: C.cyan }}>■</span> VeriCell</div></Rise>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 44 }}>
          <Rise at={5.5}><div style={{ font: `600 110px ${serif}`, lineHeight: 0.95 }}><span style={{ color: C.green }}>13</span><span style={{ color: C.muted, fontSize: 56 }}> vs </span><span style={{ color: C.amber }}>49</span><span style={{ fontSize: 56 }}>%</span></div>
            <div style={{ font: `400 25px/1.4 ${sans}`, color: C.muted, marginTop: 10 }}>of "largest" answers are wrong when VeriCell says the conclusion <b style={{ color: C.green }}>holds</b>, vs when it <b style={{ color: C.amber }}>warns</b></div></Rise>
          <Rise at={11.5}><div style={{ font: `600 110px ${serif}`, color: C.blue, lineHeight: 0.95 }}>16 / 16</div>
            <div style={{ font: `400 25px/1.4 ${sans}`, color: C.muted, marginTop: 10 }}>analysis plans correct on <b style={{ color: C.ink }}>Muse Spark Open</b>: the reasoning runs on open weights</div></Rise>
        </div>
      </div>
    </AbsoluteFill>
  );
};

const CloseLockup: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const dim = interpolate(frame / fps, [0.6, 1.4], [0, 0.6], { ...clamp, easing: ease });
  return (
  <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", background: `rgba(5,8,12,${dim})` }}>
    <Rise at={0.9} style={{ background: "rgba(7,10,15,.92)", border: `1px solid ${C.line}`, borderRadius: 20, padding: "36px 48px", maxWidth: 1240, boxShadow: "0 30px 90px rgba(0,0,0,.55)", textAlign: "center" }}>
      <div style={{ font: `600 46px/1.15 ${serif}`, color: C.ink }}>AI that knows when it might be wrong, and helps scientists find out what's true.</div>
      <div style={{ font: `500 25px ${mono}`, color: C.cyan, marginTop: 16 }}>pranavdaware1-1--vericell-web.us-east.modal.run</div>
      <div style={{ font: `500 22px ${mono}`, color: C.muted, marginTop: 6 }}>github.com/dawarepranav/VeriCell · research use only</div>
    </Rise>
  </AbsoluteFill>
  );
};

// ---------------------------------------------------------------- the film
export const Demo: React.FC<{ tl?: any; music?: string; musicGain?: number; musicFrom?: number; scored?: boolean }> = ({ tl, music = "music.mp3", musicGain = 1, musicFrom = 0, scored = false }) => {
  const { fps } = useVideoConfig();
  const T: any = tl ?? timeline;
  const total = T.total * fps;
  const speaking = (f: number) => T.scenes.some((s: any) => {
    const a = (s.start + s.vo_at) * fps, b = a + s.vo_len * fps;
    return f >= a - 6 && f <= b + 6; // duck just before the voice, release just after
  });
  const musicVolume = (f: number) => {
    const edge = Math.min(interpolate(f, [0, 1.2 * fps], [0, 1], clamp), interpolate(f, [total - 2.5 * fps, total - 0.2 * fps], [1, 0], clamp));
    const near = [-8, -4, 0, 4, 8].filter((d) => speaking(f + d)).length / 5; // smooth the duck over a few frames
    return (scored ? 1 : edge) * (0.34 - 0.2 * near) * musicGain; // a scored bed already carries its own fades
  };
  return (
    <AbsoluteFill style={{ background: C.bg, fontFamily: sans }}>
      {music !== "none" && <Audio name="Music" src={staticFile(`audio/${music}`)} volume={musicVolume} trimBefore={musicFrom * fps} />}
      {T.scenes.map((s: any) => {
        const from = Math.round(s.start * fps), dur = Math.round(s.dur * fps);
        const fadeIn = s.fade ?? 0.35;
        return (
          <Sequence key={s.id} name={s.id} from={from} durationInFrames={dur}>
            <SceneFade d={s.dur} fadeIn={fadeIn}>
              {s.kind === "hook" && <Hook d={s.dur} textAt={s.id === "hook" ? 1.0 : 0.15} />}
              {s.kind === "footage" && (s.window ? (
                <><Backdrop /><Window x={s.window.x} y={s.window.y} w={s.window.w}><Footage segs={s.segs} cams={s.cams} rings={s.rings} cursor={s.cursor} clicks={s.clicks} /></Window><Panel p={s.panel} /></>
              ) : (
                <Footage segs={s.segs} cams={s.cams} rings={s.rings} cursor={s.cursor} clicks={s.clicks} dim={s.dim ?? 0} />
              ))}
              {s.kind === "graphic" && <><AbsoluteFill style={{ filter: "blur(18px) brightness(.22) saturate(.6)" }}><Footage segs={s.segs} cams={[]} rings={[]} cursor={[]} clicks={[]} /></AbsoluteFill>
                {s.id === "truth" && <Truth />}{s.id === "compare" && <Compare />}{s.id === "proof" && <Proof />}</>}
              {s.lockup && <Lockup at={s.lockup[0]} until={s.lockup[1]} />}
              {s.id.endsWith("close") && <CloseLockup />}
              {(s.callouts ?? []).map((c: any, i: number) => <Callout key={i} c={c} />)}
            </SceneFade>
            <Sequence name={`VO ${s.id}`} from={Math.round(s.vo_at * fps)} layout="none">
              <Audio src={staticFile(`audio/${s.id}.mp3`)} />
            </Sequence>
            {(s.sfx ?? []).map((x: any, i: number) => (
              <Sequence key={i} name={`sfx ${x.src}`} from={Math.round(x.t * fps)} durationInFrames={Math.round(1.5 * fps)} layout="none">
                <Audio src={staticFile(`audio/sfx/${x.src}`)} volume={x.vol ?? 0.35} />
              </Sequence>
            ))}
            <Captions chunks={s.captions} />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};

const SceneFade: React.FC<{ d: number; fadeIn: number; children: React.ReactNode }> = ({ d, fadeIn, children }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const a = interpolate(t, [0, fadeIn], [0, 1], clamp); // scenes overlap: each dissolves in over the previous one
  return <AbsoluteFill style={{ opacity: a }}>{children}</AbsoluteFill>;
};
