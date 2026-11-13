import { useEffect, useRef, useState } from 'react';
import { useStore } from './store';
import type { ResolutionDistribution } from './types';

const PALETTE = ['#b49ddf', '#76c4bb', '#d5b27c', '#759edb', '#ee8d98', '#b4c67a'];
const color = (index: number) => PALETTE[index] || `hsl(${(265 + index * 137.5) % 360} 55% 68%)`;
type Point = { x: number; y: number; bin: number; curve: ResolutionDistribution };

function IsometricDistribution({ curves }: { curves: ResolutionDistribution[] }) {
  const canvas = useRef<HTMLCanvasElement>(null),
    points = useRef<Point[]>([]);
  const [hover, setHover] = useState<(Point & { left: number; top: number }) | null>(null);
  useEffect(() => {
    setHover(null);
    const element = canvas.current!;
    const draw = () => {
      const rect = element.getBoundingClientRect(),
        width = rect.width,
        height = rect.height;
      if (!width || !height) return;
      const ratio = window.devicePixelRatio || 1;
      element.width = Math.round(width * ratio);
      element.height = Math.round(height * ratio);
      const ctx = element.getContext('2d')!;
      ctx.scale(ratio, ratio);
      ctx.clearRect(0, 0, width, height);
      const left = 41,
        bottom = height - 44,
        depthWidth = width * 0.2,
        depthHeight = 54;
      const plotWidth = width - left - depthWidth - 16,
        plotHeight = height - 120;
      const ceiling = Math.max(1e-8, ...curves.flatMap((curve) => curve.probabilities)) * 1.1;
      const project = (t: number, depth: number, probability: number) => ({
        x: left + t * plotWidth + depth * depthWidth,
        y: bottom - depth * depthHeight - (probability / ceiling) * plotHeight,
      });
      const line = (a: { x: number; y: number }, b: { x: number; y: number }) => {
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      };
      ctx.font = '10px "Segoe UI", sans-serif';
      ctx.strokeStyle = '#3d3d3d';
      ctx.fillStyle = '#adadad';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const probability = (ceiling * i) / 4;
        line(project(0, 0, probability), project(0, 1, probability));
        line(project(0, 1, probability), project(1, 1, probability));
        const label = project(0, 0, probability);
        ctx.textAlign = 'right';
        ctx.fillText((probability * 100).toPrecision(2), label.x - 6, label.y + 3);
        const t = i / 4,
          tick = project(t, 0, 0);
        line(tick, project(t, 1, 0));
        ctx.textAlign = 'center';
        ctx.fillText(t.toFixed(2), tick.x, tick.y + 15);
      }
      ctx.textAlign = 'left';
      ctx.fillText('Probability / bin (%)', 4, 14);
      ctx.textAlign = 'center';
      ctx.fillText('Clean · timestep · Noise', left + plotWidth / 2, height - 6);
      const plotted: Point[] = [];
      // Draw the far resolution slices first, using one common probability scale.
      for (let index = curves.length - 1; index >= 0; index--) {
        const curve = curves[index],
          depth = curves.length > 1 ? index / (curves.length - 1) : 0.5;
        ctx.strokeStyle = '#3d3d3d';
        line(project(0, depth, 0), project(1, depth, 0));
        ctx.strokeStyle = color(index);
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        curve.probabilities.forEach((probability, bin) => {
          const point = project((bin + 0.5) / curve.probabilities.length, depth, probability);
          if (!bin) ctx.moveTo(point.x, point.y);
          else ctx.lineTo(point.x, point.y);
          plotted.push({ ...point, bin, curve });
        });
        ctx.stroke();
        ctx.lineWidth = 1;
        const tick = project(1, depth, 0);
        ctx.fillStyle = color(index);
        ctx.textAlign = 'right';
        ctx.fillText(`${curve.resolution}px`, tick.x, tick.y - 5);
      }
      points.current = plotted;
    };
    let frame = 0;
    const schedule = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(draw);
    };
    const observer = new ResizeObserver(schedule);
    observer.observe(element);
    schedule();
    return () => {
      observer.disconnect();
      cancelAnimationFrame(frame);
    };
  }, [curves]);
  return (
    <div className="distribution-isometric">
      <canvas
        ref={canvas}
        role="img"
        aria-label="Resolution timestep distributions (3D)"
        data-testid="timestep-3d"
        onPointerMove={(event) => {
          const rect = event.currentTarget.getBoundingClientRect(),
            x = event.clientX - rect.left,
            y = event.clientY - rect.top;
          let nearest: Point | null = null,
            distance = 144;
          for (const point of points.current) {
            const candidate = (point.x - x) ** 2 + (point.y - y) ** 2;
            if (candidate < distance) {
              nearest = point;
              distance = candidate;
            }
          }
          setHover(
            nearest
              ? { ...nearest, left: Math.max(2, Math.min(rect.width - 185, x + 10)), top: y + 12 }
              : null,
          );
        }}
        onPointerLeave={() => setHover(null)}
      />
      <div className="distribution-legend" aria-label="Resolution shifts">
        {curves.map((curve, index) => (
          <span key={curve.resolution} style={{ color: color(index) }}>
            <i style={{ background: color(index) }} />
            {curve.resolution}px · shift {curve.shift.toFixed(3)}
          </span>
        ))}
      </div>
      {hover && (
        <div className="chart-tooltip" role="tooltip" style={{ left: hover.left, top: hover.top }}>
          {hover.curve.resolution}px · shift {hover.curve.shift.toFixed(3)}
          <strong>
            t [{(hover.bin / hover.curve.probabilities.length).toFixed(2)},{' '}
            {((hover.bin + 1) / hover.curve.probabilities.length).toFixed(2)}
            {hover.bin === hover.curve.probabilities.length - 1 ? ']' : ')'}:{' '}
            {(hover.curve.probabilities[hover.bin] * 100).toPrecision(4)}%
          </strong>
        </div>
      )}
    </div>
  );
}

export default function TimestepPreview() {
  const state = useStore((value) => value.state);
  const distribution = state?.validation.distribution;
  const dynamic = state?.flat['flow.cabal_shift'] || state?.flat['flow.flux_shift'];
  if (state && dynamic && state.validation.resolutionDistributions.length) {
    return (
      <div className="distribution" aria-label="Timestep distribution">
        <IsometricDistribution curves={state.validation.resolutionDistributions} />
        <small>Preview before curriculum mapping</small>
      </div>
    );
  }
  if (!state || !distribution || dynamic)
    return (
      <div className="distribution muted">
        {state?.validation.timestepError || 'Fix the timestep settings to update the preview.'}
      </div>
    );
  const maximum = Math.max(...distribution.probabilities, 1e-8);
  return (
    <div className="distribution" aria-label="Timestep distribution">
      <>
        <div className="distribution-bars" data-testid="timestep-2d">
          {distribution.probabilities.map((probability, index) => (
            <div
              key={index}
              title={`${(index / distribution.probabilities.length).toFixed(2)}–${((index + 1) / distribution.probabilities.length).toFixed(2)}: ${(probability * 100).toFixed(2)}%`}
              style={{ height: (probability / maximum) * 100 + '%' }}
            />
          ))}
        </div>
        <div className="distribution-labels">
          <span>Clean · 0</span>
          <span>Median {distribution.median.toFixed(3)}</span>
          <span>Noise · 1</span>
        </div>
      </>
      <small>
        {`${(distribution.high_noise * 100).toFixed(1)}% at high noise · `}Preview before curriculum
        mapping
      </small>
    </div>
  );
}
