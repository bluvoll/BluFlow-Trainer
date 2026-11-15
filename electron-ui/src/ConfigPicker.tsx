import { useEffect, useRef, useState } from 'react';

export function ConfigPicker({
  presets,
  path,
  onSelect,
}: {
  presets: { name: string; path: string }[];
  path: string | null;
  onSelect: (path: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const search = useRef<HTMLInputElement>(null);
  const results = presets.filter((preset) =>
    preset.name.toLocaleLowerCase().includes(query.trim().toLocaleLowerCase()),
  );
  const current =
    presets.find((preset) => preset.path === path)?.name ||
    path?.split(/[\\/]/).pop() ||
    'Unsaved recipe';

  useEffect(() => {
    if (!open) return;
    search.current?.focus();
    const outside = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener('pointerdown', outside);
    return () => document.removeEventListener('pointerdown', outside);
  }, [open]);

  const close = () => {
    setOpen(false);
    trigger.current?.focus();
  };

  return (
    <div
      className="config-picker"
      ref={root}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) setOpen(false);
      }}
      onKeyDown={(event) => {
        if (event.key === 'Escape' && open) {
          event.preventDefault();
          event.stopPropagation();
          close();
        }
      }}
    >
      <button
        ref={trigger}
        className="config-picker-trigger"
        aria-label="Configuration preset"
        aria-haspopup="dialog"
        aria-expanded={open}
        title={path || current}
        onClick={() => {
          setQuery('');
          setOpen(!open);
        }}
      >
        <span>{current}</span>
        <span aria-hidden="true">⌄</span>
      </button>
      {open && (
        <div
          className="config-picker-popup"
          role="dialog"
          aria-label="Choose configuration"
          onKeyDown={(event) => {
            if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
            event.preventDefault();
            const items = Array.from(
              event.currentTarget.querySelectorAll<HTMLElement>('input, button'),
            );
            const index = items.indexOf(document.activeElement as HTMLElement);
            items[
              (index + (event.key === 'ArrowDown' ? 1 : items.length - 1)) % items.length
            ]?.focus();
          }}
        >
          <input
            ref={search}
            type="search"
            aria-label="Find configuration"
            placeholder="Find a config…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && results.length) {
                event.preventDefault();
                close();
                onSelect(results[0].path);
              }
            }}
          />
          <div className="config-picker-results">
            {results.map((preset) => (
              <button
                key={preset.path}
                title={preset.path}
                aria-current={preset.path === path ? 'true' : undefined}
                onClick={() => {
                  close();
                  onSelect(preset.path);
                }}
              >
                {preset.name}
              </button>
            ))}
            {!results.length && <p role="status">No matching configurations</p>}
          </div>
        </div>
      )}
    </div>
  );
}
