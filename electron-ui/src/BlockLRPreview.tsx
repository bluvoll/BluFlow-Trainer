import { useEffect, useRef, useState } from 'react';
import type { BlockLRPreviewData } from './types';

const LEFT = 50,
  RIGHT = 14,
  TOP = 24,
  BOTTOM = 36;

function format(value: number, digits: number) {
  const rounded = value.toPrecision(digits);
  return Number.isFinite(Number(rounded)) ? String(Number(rounded)) : rounded;
}

function BlockLRPlot({ multipliers }: { multipliers: number[] }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [block, setBlock] = useState(0);
  const last = multipliers.length - 1;
  const readout = `Block ${block} · ${format(multipliers[block], 10)}×`;

  useEffect(() => {
    const element = canvas.current!;
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize({ width, height });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const { width, height } = size;
    const plotWidth = width - LEFT - RIGHT;
    const plotHeight = height - TOP - BOTTOM;
    if (plotWidth <= 0 || plotHeight <= 0) return;
    const element = canvas.current!;
    const ratio = window.devicePixelRatio || 1;
    element.width = Math.round(width * ratio);
    element.height = Math.round(height * ratio);
    const context = element.getContext('2d')!;
    context.scale(ratio, ratio);
    const styles = getComputedStyle(element);
    const accent = styles.getPropertyValue('--ui-accent');
    const ceiling = Math.max(1, ...multipliers);
    const x = (index: number) => LEFT + (index / last) * plotWidth;
    const y = (value: number) => TOP + (1 - value / ceiling) * plotHeight;

    context.font = '11px "Segoe UI", sans-serif';
    context.fillStyle = styles.getPropertyValue('--ui-muted');
    context.fillText('LR multiplier (×)', LEFT, 14);
    context.strokeStyle = styles.getPropertyValue('--ui-border');
    context.textAlign = 'right';
    for (const fraction of [0, 0.25, 0.5, 0.75, 1]) {
      const value = ceiling * fraction;
      const gridY = y(value);
      context.beginPath();
      context.moveTo(LEFT, gridY);
      context.lineTo(x(last), gridY);
      context.stroke();
      context.fillText(format(value, 2), LEFT - 7, gridY + 4);
    }
    context.textAlign = 'center';
    for (const index of [0, 10, 20, 30, last]) {
      context.fillText(String(index), x(index), height - BOTTOM + 15);
    }
    context.fillText('Joint block', LEFT + plotWidth / 2, height - 5);

    context.strokeStyle = accent;
    context.lineWidth = 1.6;
    context.beginPath();
    multipliers.forEach((value, index) => {
      if (index === 0) context.moveTo(x(index), y(value));
      else context.lineTo(x(index), y(value));
    });
    context.stroke();
    context.fillStyle = accent;
    context.beginPath();
    context.arc(x(block), y(multipliers[block]), 3.5, 0, 2 * Math.PI);
    context.fill();
    element.dataset.rendered = 'true';
  }, [block, last, multipliers, size]);

  const select = (index: number) => setBlock(Math.max(0, Math.min(index, last)));
  return (
    <>
      <header>
        <strong>Block LR by depth</strong>
        <output aria-hidden="true">{readout}</output>
      </header>
      <canvas
        ref={canvas}
        role="slider"
        tabIndex={0}
        aria-label="Inspect block LR by depth"
        aria-valuemin={0}
        aria-valuemax={last}
        aria-valuenow={block}
        aria-valuetext={readout}
        onPointerMove={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          const plotWidth = rect.width - LEFT - RIGHT;
          if (plotWidth <= 0) return;
          const fraction = (event.clientX - rect.left - LEFT) / plotWidth;
          select(Math.round(fraction * last));
        }}
        onKeyDown={(event) => {
          const next = {
            ArrowLeft: block - 1,
            ArrowDown: block - 1,
            ArrowRight: block + 1,
            ArrowUp: block + 1,
            Home: 0,
            End: last,
          }[event.key];
          if (typeof next !== 'number') return;
          event.preventDefault();
          select(next);
        }}
      />
    </>
  );
}

export function BlockLRPreview({ preview }: { preview: BlockLRPreviewData | null }) {
  return (
    <section className="block-lr-preview" aria-label="Block learning rate preview">
      {preview?.multipliers ? (
        <BlockLRPlot multipliers={preview.multipliers} />
      ) : (
        <>
          <header>
            <strong>Block LR by depth</strong>
          </header>
          <p role="status">{preview?.error || 'Updating preview…'}</p>
        </>
      )}
    </section>
  );
}
