import { useLayoutEffect, useRef, type ReactNode } from 'react';

// Multicol balancing can leave whole columns empty when cards have very different
// heights. Grid places every card into a fitting column; measured row spans keep
// the compact vertical packing without reparenting focused editors on resize.
export default function SettingsGrid({ children }: { children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const grid = ref.current!;
    const cards = new Set<HTMLElement>();
    let frame = 0;
    const measure = () => {
      const style = getComputedStyle(grid);
      const row = parseFloat(style.gridAutoRows);
      const gap = parseFloat(style.columnGap);
      for (const card of cards) {
        const span = Math.max(1, Math.ceil((card.getBoundingClientRect().height + gap) / row));
        const end = `span ${span}`;
        if (card.style.gridRowEnd !== end) card.style.gridRowEnd = end;
      }
    };
    const sizes = new ResizeObserver(() => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(measure);
    });
    const sync = () => {
      for (const card of cards) {
        if (card.parentElement !== grid) {
          sizes.unobserve(card);
          cards.delete(card);
        }
      }
      for (const card of grid.children) {
        if (card instanceof HTMLElement && !cards.has(card)) {
          cards.add(card);
          sizes.observe(card);
        }
      }
      measure();
    };
    const childrenChanged = new MutationObserver(sync);
    childrenChanged.observe(grid, { childList: true });
    sync();
    return () => {
      childrenChanged.disconnect();
      sizes.disconnect();
      cancelAnimationFrame(frame);
    };
  }, []);
  return (
    <div className="settings-groups" ref={ref}>
      {children}
    </div>
  );
}
