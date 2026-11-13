import { useEffect, useRef, useState } from 'react';
import type { Flat } from './types';
type Series = [string, (record: Flat) => number | undefined][];
type Point = { x: number; y: number; step: number; value: number; label: string };
type Geometry = {
  left: number;
  top: number;
  width: number;
  height: number;
  min: number;
  max: number;
  fullMin: number;
  fullMax: number;
  points: Point[];
};
const COLORS = ['#B49DDF', '#76C4BB', '#ADADAD', '#D5B27C'];
export default function Chart({
  title,
  records,
  series,
  format = (n: number) => n.toPrecision(3),
}: {
  title: string;
  records: Flat[];
  series: Series;
  format?: (n: number) => string;
}) {
  const ref = useRef<HTMLCanvasElement>(null),
    geometry = useRef<Geometry | null>(null),
    drag = useRef<{ x: number; range: [number, number] } | null>(null);
  const [view, setView] = useState<[number, number] | null>(null),
    [hover, setHover] = useState<(Point & { left: number; top: number }) | null>(null);
  const clamp = (start: number, end: number, g: Geometry): [number, number] => {
    const span = Math.min(end - start, g.fullMax - g.fullMin),
      left = Math.max(g.fullMin, Math.min(start, g.fullMax - span));
    return [left, left + span];
  };
  useEffect(() => {
    if (!records.length) {
      setView(null);
      setHover(null);
    }
  }, [records.length === 0]);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const draw = () => {
      const rect = canvas.getBoundingClientRect(),
        width = Math.max(100, rect.width),
        height = Math.max(120, rect.height),
        dpr = window.devicePixelRatio || 1;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      const c = canvas.getContext('2d')!;
      c.scale(dpr, dpr);
      c.clearRect(0, 0, width, height);
      const left = 55,
        top = 13,
        w = width - left - 12,
        h = height - top - 28;
      const fullMin = records[0]?.step ?? 0,
        fullMax = Math.max(fullMin + 1, records.at(-1)?.step ?? 1);
      const min = view ? Math.max(fullMin, view[0]) : fullMin,
        max = view ? Math.min(fullMax, view[1]) : fullMax;
      const visible = records.filter((r) => r.step >= min && r.step <= max);
      let low = Infinity,
        high = -Infinity;
      for (const record of visible)
        for (const [, get] of series) {
          const value = get(record);
          if (typeof value === 'number' && Number.isFinite(value)) {
            low = Math.min(low, value);
            high = Math.max(high, value);
          }
        }
      if (low === Infinity) {
        low = 0;
        high = 1;
      }
      const margin = (high - low) * 0.08 || Math.abs(high) * 0.05 || 0.1;
      low -= margin;
      high += margin;
      c.font = '10px "Segoe UI", sans-serif';
      c.lineWidth = 1;
      for (let i = 0; i < 4; i++) {
        const y = top + (h * i) / 3;
        c.strokeStyle = '#313131';
        c.beginPath();
        c.moveTo(left, y);
        c.lineTo(width - 12, y);
        c.stroke();
        c.fillStyle = '#ADADAD';
        c.textAlign = 'right';
        c.fillText(format(high - ((high - low) * i) / 3), left - 8, y + 3);
      }
      c.fillStyle = '#ADADAD';
      c.textAlign = 'left';
      c.fillText(Math.round(min).toString(), left, height - 8);
      c.textAlign = 'right';
      c.fillText(Math.round(max).toString(), width - 12, height - 8);
      const points: Point[] = [];
      c.save();
      c.beginPath();
      c.rect(left, top, w, h);
      c.clip();
      series.forEach(([label, get], index) => {
        c.strokeStyle = COLORS[index];
        c.lineWidth = index === 1 ? 1.8 : 1.2;
        c.setLineDash(index > 1 ? [4, 3] : []);
        c.beginPath();
        let drawing = false;
        const stride = Math.max(1, Math.floor(visible.length / Math.max(1, w)));
        for (let start = 0; start < visible.length; start += stride) {
          const end = Math.min(visible.length, start + stride),
            indices = new Set([start, end - 1]);
          let lowest = Infinity,
            highest = -Infinity,
            lowIndex = start,
            highIndex = start;
          for (let i = start; i < end; i++) {
            const value = get(visible[i]);
            if (typeof value !== 'number' || !Number.isFinite(value)) continue;
            if (value < lowest) {
              lowest = value;
              lowIndex = i;
            }
            if (value > highest) {
              highest = value;
              highIndex = i;
            }
          }
          indices.add(lowIndex);
          indices.add(highIndex);
          // Draw each bucket in chronological order, retaining its extrema.
          for (const i of [...indices].sort((a, b) => a - b)) {
            const record = visible[i],
              value = get(record);
            if (typeof value !== 'number' || !Number.isFinite(value)) {
              drawing = false;
              continue;
            }
            const x = left + ((record.step - min) / Math.max(1e-9, max - min)) * w,
              y = top + ((high - value) / (high - low)) * h;
            if (drawing) c.lineTo(x, y);
            else {
              c.moveTo(x, y);
              drawing = true;
            }
            points.push({ x, y, step: record.step, value, label });
          }
        }
        c.stroke();
        c.setLineDash([]);
      });
      c.restore();
      geometry.current = { left, top, width: w, height: h, min, max, fullMin, fullMax, points };
      if (!visible.length) {
        c.textAlign = 'center';
        c.fillStyle = '#ADADAD';
        c.fillText(
          records.length ? 'No points in this range' : 'Waiting for training metrics',
          left + w / 2,
          top + h / 2,
        );
      }
    };
    let frame = 0;
    const schedule = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(draw);
    };
    const observer = new ResizeObserver(schedule);
    observer.observe(canvas);
    schedule();
    const wheel = (event: WheelEvent) => {
      const g = geometry.current,
        rect = canvas.getBoundingClientRect();
      if (!g) return;
      const x = event.clientX - rect.left,
        y = event.clientY - rect.top;
      if (x < g.left || x > g.left + g.width || y < g.top || y > g.top + g.height) return;
      event.preventDefault();
      setHover(null);
      const relative = Math.max(0, Math.min(1, (x - g.left) / g.width)),
        center = g.min + (g.max - g.min) * relative;
      const span = Math.max(
        1,
        Math.min(
          g.fullMax - g.fullMin,
          (g.max - g.min) * Math.exp(Math.max(-4, Math.min(4, event.deltaY / 120)) * 0.12),
        ),
      );
      setView(clamp(center - span * relative, center + span * (1 - relative), g));
    };
    canvas.addEventListener('wheel', wheel, { passive: false });
    return () => {
      observer.disconnect();
      cancelAnimationFrame(frame);
      canvas.removeEventListener('wheel', wheel);
    };
  }, [records, series, view]);
  return (
    <section className="chart">
      <div className="chart-heading">
        <strong>{title}</strong>
        <div className="chart-legend">
          {series.map(([name], index) => (
            <span key={name}>
              <i style={{ background: COLORS[index] }} />
              {name}
            </span>
          ))}
          {view && (
            <button
              aria-label={'Reset ' + title + ' view'}
              onClick={() => {
                setView(null);
                setHover(null);
              }}
            >
              Reset view
            </button>
          )}
        </div>
      </div>
      <canvas
        ref={ref}
        role="img"
        aria-label={title + ' chart'}
        data-zoomed={!!view}
        title="Wheel to zoom; drag to pan; hover a point for its value."
        onPointerDown={(event) => {
          const g = geometry.current,
            rect = event.currentTarget.getBoundingClientRect();
          if (event.button !== 0 || !g || event.clientY - rect.top < g.top) return;
          drag.current = { x: event.clientX, range: [g.min, g.max] };
          event.currentTarget.setPointerCapture(event.pointerId);
          event.currentTarget.style.cursor = 'grabbing';
          setHover(null);
        }}
        onPointerMove={(event) => {
          const g = geometry.current;
          if (!g) return;
          if (drag.current) {
            const shift =
              (-(event.clientX - drag.current.x) / g.width) *
              (drag.current.range[1] - drag.current.range[0]);
            setView(clamp(drag.current.range[0] + shift, drag.current.range[1] + shift, g));
            return;
          }
          const rect = event.currentTarget.getBoundingClientRect(),
            x = event.clientX - rect.left,
            y = event.clientY - rect.top;
          let best: Point | undefined,
            distance = 144;
          for (const point of g.points) {
            const d = (point.x - x) ** 2 + (point.y - y) ** 2;
            if (d < distance) {
              best = point;
              distance = d;
            }
          }
          const parent = event.currentTarget.parentElement!.getBoundingClientRect();
          setHover(
            best
              ? {
                  ...best,
                  left: Math.max(4, Math.min(parent.width - 170, event.clientX - parent.left + 12)),
                  top: event.clientY - parent.top + 10,
                }
              : null,
          );
        }}
        onPointerUp={(event) => {
          drag.current = null;
          event.currentTarget.style.cursor = 'grab';
          if (event.currentTarget.hasPointerCapture(event.pointerId))
            event.currentTarget.releasePointerCapture(event.pointerId);
        }}
        onPointerCancel={() => {
          drag.current = null;
        }}
        onPointerLeave={() => {
          if (!drag.current) setHover(null);
        }}
      />
      {hover && (
        <div className="chart-tooltip" role="tooltip" style={{ left: hover.left, top: hover.top }}>
          {hover.label} · step {hover.step}
          <strong>{hover.value.toPrecision(6)}</strong>
        </div>
      )}
    </section>
  );
}
