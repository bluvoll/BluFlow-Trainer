import { useState } from 'react';
import type { Flat } from './types';
import { flush, state as currentStore } from './store';

export function StageExposure({ flat, world }: { flat: Flat; world: number }) {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{ key: string; text: string } | null>(null);
  const key = JSON.stringify([flat, world]);
  return (
    <div className="stage-exposure">
      <button
        disabled={busy || !flat['resolution_schedule.enabled']}
        onClick={async () => {
          setBusy(true);
          let requestKey = key;
          try {
            await flush();
            const current = currentStore().state!;
            const snapshot = { flat: current.flat, world: current.selectedGpus.length };
            requestKey = JSON.stringify([snapshot.flat, snapshot.world]);
            const plan = await window.bluflow.request('preview.stages', snapshot);
            setResult({ key: requestKey, text: plan.text });
          } catch (exc) {
            setResult({ key: requestKey, text: String(exc) });
          } finally {
            setBusy(false);
          }
        }}
      >
        {busy ? 'Scanning dataset…' : 'Calculate dataset exposure'}
      </button>
      <p>
        Reads cached dataset entries, including repeats. Recalculate after changing files or
        settings.
      </p>
      {result &&
        (result.key === key ? (
          <pre style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{result.text}</pre>
        ) : (
          <p>Settings changed. Recalculate the exposure plan.</p>
        ))}
    </div>
  );
}
