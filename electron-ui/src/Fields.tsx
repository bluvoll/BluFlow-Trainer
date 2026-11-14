import { useEffect, useRef, useState } from 'react';
import type { Field, Flat } from './types';
import { edit, error, useStore } from './store';

export const editorKinds = [
  'TextEditor',
  'OptTextEditor',
  'PathEditor',
  'IntEditor',
  'OptIntEditor',
  'ImageTargetEditor',
  'FloatEditor',
  'SciEditor',
  'BoolEditor',
  'TrainComponentEditor',
  'ChoiceEditor',
  'OptimizerEditor',
  'NumListEditor',
  'IntListEditor',
  'FloatListEditor',
  'StrListEditor',
  'CheckSetEditor',
  'BucketMapEditor',
  'WeightsEditor',
  'CanvasListEditor',
  'CurriculumEditor',
  'SubsetEditor',
  'BlockLRRangesEditor',
  'ResolutionStagesEditor',
];
const numeric = (text: string, integer = false) =>
  text.trim() === ''
    ? null
    : (integer ? /^[+-]?\d+$/ : /^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i).test(text) &&
        Number.isFinite(Number(text))
      ? Number(text)
      : text;
const same = (a: any, b: any) => JSON.stringify(a) === JSON.stringify(b);

function ValueInput({
  value,
  onChange,
  format = String,
  parse = (v: string) => v,
  multiline = false,
  ...props
}: {
  value: any;
  onChange: (v: any) => void;
  format?: (v: any) => string;
  parse?: (v: string) => any;
  [key: string]: any;
}) {
  const formatValue = (v: any) => (v == null ? '' : format(v));
  const [text, setText] = useState(() => formatValue(value));
  const last = useRef(value),
    focused = useRef(false);
  useEffect(() => {
    if (!same(value, last.current)) {
      last.current = value;
      if (!focused.current) setText(formatValue(value));
    }
  }, [value]);
  const Input = multiline ? 'textarea' : 'input';
  return (
    <Input
      {...props}
      value={text}
      onFocus={() => (focused.current = true)}
      onBlur={() => {
        focused.current = false;
        setText(formatValue(value));
      }}
      onChange={(event) => {
        const text = event.target.value;
        setText(text);
        const parsed = parse(text);
        last.current = parsed;
        onChange(parsed);
      }}
    />
  );
}

export function PathInput({
  value,
  onChange,
  kind = 'file',
  placeholder = '',
  label = 'Path',
}: {
  value: any;
  onChange: (v: any) => void;
  kind?: string;
  placeholder?: string;
  label?: string;
}) {
  return (
    <div className="path-input">
      <ValueInput aria-label={label} value={value} placeholder={placeholder} onChange={onChange} />
      <button
        type="button"
        title={'Browse ' + label}
        aria-label={'Browse ' + label}
        onClick={() =>
          void window.bluflow
            .file({ kind: kind === 'folder' ? 'folder' : 'open', defaultPath: value || undefined })
            .then((path) => {
              if (path) onChange(path);
            })
            .catch(error)
        }
      >
        <FolderIcon />
      </button>
    </div>
  );
}
export function FolderIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      aria-hidden="true"
    >
      <path d="M3 7V5h6l2 2h10v13H3z" />
      <path d="M3 10h18" />
    </svg>
  );
}
export function CloseIcon() {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      aria-hidden="true"
    >
      <path d="m4 4 8 8m0-8-8 8" />
    </svg>
  );
}

const tableDrafts = new Map<string, { revision: number; rows: Flat[] }>();
function TargetInput({
  value,
  onChange,
  label = 'Image weighting',
}: {
  value: any;
  onChange: (value: any) => void;
  label?: string;
}) {
  const retained = useRef(1);
  if (typeof value === 'number') retained.current = value;
  return (
    <div className="image-weighting">
      <label className="check">
        <span className="weight-switch">
          <input
            type="checkbox"
            role="switch"
            aria-label={`${label}: use target images`}
            checked={value != null}
            onChange={(e) => onChange(e.target.checked ? retained.current : null)}
          />
          <span className="switch-track" aria-hidden="true" />
        </span>
        {value == null ? 'Repeats' : 'Target images'}
      </label>
      {value != null && (
        <ValueInput
          aria-label={`${label}: target image count`}
          type="number"
          min="0"
          step="1"
          value={value}
          parse={(text) => numeric(text, true) ?? ''}
          onChange={onChange}
        />
      )}
    </div>
  );
}

function TableEditor({
  field,
  value,
  onChange,
}: {
  field: Field;
  value: any;
  onChange: (v: any) => void;
}) {
  const { data, state, configRevision } = useStore((value) => ({
    data: value.data,
    state: value.state,
    configRevision: value.configRevision,
  }));
  const kind = field.kind,
    curriculum = kind === 'CurriculumEditor';
  const cacheKey = field.key;
  const [retained, setRetained] = useState<Flat[]>(() => {
    const draft = tableDrafts.get(cacheKey);
    return draft?.revision === configRevision ? draft.rows : [];
  });
  const rows: Flat[] = Array.isArray(value) && value.length ? value : curriculum ? retained : [];
  const enabled = !curriculum || (Array.isArray(value) && value.length > 0);
  const commit = (next: Flat[]) => {
    setRetained(next);
    tableDrafts.set(cacheKey, { revision: configRevision, rows: next });
    onChange(curriculum && !enabled ? [] : next);
  };
  const columns =
    kind === 'SubsetEditor'
      ? ['Folder', 'Image weighting', 'Texture crops']
      : curriculum
        ? ['Start %', 't min', 't max', 'Mode', 'LR ×']
        : kind === 'BlockLRRangesEditor'
          ? ['First block', 'Last block', 'LR ×']
          : ['Resolution', 'Share %', 'LR %'];
  const add = () => {
    const firstBlock = Array.from({ length: 42 }, (_, i) => i).find(
      (i) => !rows.some((row) => i >= row.start && i <= row.end),
    );
    if (kind === 'BlockLRRangesEditor' && firstBlock === undefined) return;
    const row =
      kind === 'SubsetEditor'
        ? { path: '', num_repeats: 1, texture: true }
        : curriculum
          ? {
              at: rows.length ? Math.min(0.99, Number(rows.at(-1)?.at || 0) + 0.1) : 0,
              t_range: [0, 1],
              mode: 'fullres',
              lr_mul: 1,
            }
          : kind === 'BlockLRRangesEditor'
            ? { start: firstBlock, end: firstBlock, lr_mul: 1 }
            : { resolution: 1024, share: 0.2, lr_mul: 1 };
    commit([...rows, row]);
  };
  const cell = (row: Flat, index: number, column: number) => {
    const set = (key: string, v: any) =>
      commit(rows.map((r, i) => (i === index ? { ...r, [key]: v } : r)));
    if (kind === 'SubsetEditor') {
      if (column === 0)
        return (
          <PathInput
            label={`Subset ${index + 1} folder`}
            value={row.path}
            kind="folder"
            onChange={(v) => set('path', v)}
          />
        );
      if (column === 1)
        return (
          <div>
            <TargetInput
              value={row.target_images}
              label={`Subset ${index + 1}`}
              onChange={(v) => set('target_images', v)}
            />
            {row.target_images == null && (
              <input
                aria-label={`Subset ${index + 1} repeats`}
                type="number"
                min="0"
                step="any"
                value={row.num_repeats ?? 1}
                onChange={(e) => set('num_repeats', numeric(e.target.value))}
              />
            )}
          </div>
        );
      return (
        <input
          aria-label={`Subset ${index + 1} texture crops`}
          type="checkbox"
          checked={row.texture !== false}
          onChange={(e) => set('texture', e.target.checked)}
        />
      );
    }
    if (curriculum) {
      if (column === 3)
        return (
          <select
            aria-label={`Phase ${index + 1} mode`}
            value={row.mode}
            onChange={(e) => set('mode', e.target.value)}
          >
            <option value="fullres">fullres</option>
            <option value="texture">texture</option>
          </select>
        );
      const v =
        column === 0
          ? Number(row.at) * 100
          : column === 4
            ? (row.lr_mul ?? 1)
            : (row.t_range?.[column - 1] ?? (column === 1 ? 0 : 1));
      return (
        <input
          aria-label={`Phase ${index + 1} ${columns[column]}`}
          type="number"
          step="any"
          value={v}
          onChange={(e) => {
            const n = numeric(e.target.value);
            if (column === 0) set('at', typeof n === 'number' ? n / 100 : n);
            else if (column === 4) set('lr_mul', n);
            else
              set(
                't_range',
                column === 1 ? [n, row.t_range?.[1] ?? 1] : [row.t_range?.[0] ?? 0, n],
              );
          }}
        />
      );
    }
    const key =
      kind === 'BlockLRRangesEditor'
        ? ['start', 'end', 'lr_mul'][column]
        : ['resolution', 'share', 'lr_mul'][column];
    const percent = kind === 'ResolutionStagesEditor' && column > 0;
    return (
      <input
        aria-label={`${curriculum ? 'Phase' : 'Row'} ${index + 1} ${columns[column]}`}
        type="number"
        step={column === 0 ? '1' : 'any'}
        value={percent ? (row[key] ?? 1) * 100 : (row[key] ?? 1)}
        onChange={(e) => {
          const n = numeric(e.target.value);
          set(key, percent && typeof n === 'number' ? n / 100 : n);
        }}
      />
    );
  };
  return (
    <div className={'table-editor ' + (kind === 'SubsetEditor' ? 'subset-editor' : '')}>
      {curriculum && (
        <label className="check">
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => {
              if (e.target.checked) {
                const next = retained.length
                  ? retained
                  : [{ at: 0, t_range: [0, 1], mode: 'fullres', lr_mul: 1 }];
                setRetained(next);
                onChange(next);
              } else {
                setRetained(rows);
                tableDrafts.set(cacheKey, { revision: configRevision, rows });
                onChange([]);
              }
            }}
          />
          Enable curriculum
        </label>
      )}
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              {columns.map((c) => (
                <th key={c}>{c}</th>
              ))}
              <th />
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={index}>
                {columns.map((_, column) => (
                  <td key={column}>{cell(row, index, column)}</td>
                ))}
                <td>
                  <button
                    aria-label={`Remove row ${index + 1}`}
                    title="Remove row"
                    onClick={() => commit(rows.filter((_, i) => i !== index))}
                  >
                    <CloseIcon />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {!rows.length && (
          <div className="empty-table">
            {kind === 'SubsetEditor'
              ? 'Use a single dataset folder, or add subsets.'
              : 'No rows configured.'}
          </div>
        )}
      </div>
      <div className="table-actions">
        <button onClick={add}>
          Add{' '}
          {kind === 'SubsetEditor'
            ? 'subset'
            : curriculum
              ? 'phase'
              : kind === 'BlockLRRangesEditor'
                ? 'range'
                : 'stage'}
        </button>
        {curriculum && (
          <button
            onClick={() => {
              const next = data?.curriculumPreset || [];
              setRetained(next);
              tableDrafts.set(cacheKey, { revision: configRevision, rows: next });
              onChange(next);
            }}
          >
            TrainTrain schedule
          </button>
        )}
        <button
          onClick={() => {
            setRetained([]);
            tableDrafts.delete(cacheKey);
            onChange([]);
          }}
        >
          Clear
        </button>
      </div>
      {kind === 'ResolutionStagesEditor' && state?.validation.stages && (
        <div className="stage-preview">
          {state.validation.stages.map((stage, i) => (
            <div key={i}>
              {stage.resolution}px · {stage.epochs ?? stage.steps}{' '}
              {stage.unit === 'epochs' ? 'epochs' : 'updates'} · effective batch{' '}
              {stage.effective_batch}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Editor({ field, value }: { field: Field; value: any }) {
  const { state } = useStore((value) => ({ state: value.state }));
  const set = (v: any) => edit(field.key, v),
    a = field.args,
    k = field.kwargs;
  const label = field.label;
  if (field.kind === 'ImageTargetEditor') return <TargetInput value={value} onChange={set} />;
  if (
    ['CurriculumEditor', 'SubsetEditor', 'BlockLRRangesEditor', 'ResolutionStagesEditor'].includes(
      field.kind,
    )
  )
    return <TableEditor field={field} value={value} onChange={set} />;
  if (field.kind === 'BoolEditor')
    return (
      <label className="check">
        <input
          aria-label={label}
          type="checkbox"
          checked={!!value}
          onChange={(e) => set(e.target.checked)}
        />
        {a[0] || label}
      </label>
    );
  if (field.kind === 'PathEditor')
    return <PathInput label={label} value={value} kind={a[0]} onChange={(v) => set(v || null)} />;
  if (field.kind === 'ChoiceEditor' || field.kind === 'OptimizerEditor') {
    const labels = a[0] || [],
      values: any[] = k.values ?? a[1] ?? labels;
    const index = values.findIndex((v) => same(v, value));
    return (
      <select
        aria-label={label}
        value={index < 0 ? 'unknown' : String(index)}
        onChange={(e) => set(values[Number(e.target.value)])}
      >
        {index < 0 && <option value="unknown">{String(value)} (not in this build)</option>}
        {labels.map((text: string, i: number) => (
          <option
            key={i}
            value={i}
            disabled={
              field.key === 'train.attention_backend' &&
              values[i] === 'torch_varlen' &&
              state?.platform === 'win32'
            }
          >
            {text}
          </option>
        ))}
      </select>
    );
  }
  if (field.kind === 'TrainComponentEditor')
    return (
      <div className="component-lr">
        <label className="check">
          <input
            type="checkbox"
            aria-label={a[0] || label}
            checked={value !== 0}
            onChange={(e) => set(e.target.checked ? null : 0)}
          />
          {a[0] || label}
        </label>
        <ValueInput
          aria-label="AdaLN learning rate"
          disabled={value === 0}
          value={value === 0 ? null : value}
          placeholder="Blank = main optimizer LR"
          onChange={set}
          parse={(s) => numeric(s)}
        />
      </div>
    );
  if (field.kind === 'WeightsEditor')
    return (
      <div className="weights">
        {(a[0] || []).map((key: string) => (
          <label key={key}>
            {key}
            <input
              aria-label={`${label} ${key}`}
              type="number"
              min="0"
              max="1000"
              value={value?.[key] ?? 0}
              onChange={(e) => set({ ...value, [key]: numeric(e.target.value, true) })}
            />
          </label>
        ))}
      </div>
    );
  if (field.kind === 'CheckSetEditor')
    return (
      <div className="check-set">
        {a[0].map((option: string) => (
          <label className="check" key={option} title={k.tooltips?.[option]}>
            <input
              aria-label={option}
              type="checkbox"
              checked={value?.includes(option) || false}
              onChange={(e) =>
                set(
                  e.target.checked
                    ? [...(value || []), option]
                    : (value || []).filter((v: string) => v !== option),
                )
              }
            />
            {option}
          </label>
        ))}
      </div>
    );
  if (field.kind === 'BucketMapEditor')
    return (
      <ValueInput
        aria-label={label}
        value={value}
        placeholder="4 or 768=16, 1024=12"
        format={(v) =>
          typeof v === 'object'
            ? Object.entries(v)
                .map(([k, v]) => `${k}=${v}`)
                .join(', ')
            : String(v)
        }
        parse={(s) => {
          if (!s.includes('=')) return numeric(s, true) ?? 1;
          const result: Flat = {};
          for (const item of s.split(',')) {
            const [key, val] = item.split('=');
            if (
              !/^\s*\d+\s*$/.test(key) ||
              val === undefined ||
              typeof numeric(val, true) !== 'number'
            )
              return s;
            result[Number(key)] = Number(val);
          }
          return result;
        }}
        onChange={set}
      />
    );
  if (
    [
      'IntListEditor',
      'FloatListEditor',
      'NumListEditor',
      'StrListEditor',
      'CanvasListEditor',
    ].includes(field.kind)
  ) {
    const canvas = field.kind === 'CanvasListEditor',
      strings = field.kind === 'StrListEditor';
    return (
      <ValueInput
        aria-label={label}
        value={value}
        multiline={strings || canvas}
        style={strings || canvas ? { minHeight: k.height || 54, resize: 'vertical' } : undefined}
        placeholder={a[0] || 'Comma-separated values'}
        format={(v) =>
          Array.isArray(v)
            ? v.map((v) => (Array.isArray(v) ? v.join('x') : v)).join(', ')
            : String(v)
        }
        parse={(s) => {
          if (!s.trim()) return strings || canvas ? [] : null;
          const pieces = s
            .replaceAll('\n', ',')
            .replaceAll(';', ',')
            .split(',')
            .map((p) => p.trim())
            .filter(Boolean);
          if (strings) return k.as_set || a[1] === true ? [...new Set(pieces)].sort() : pieces;
          if (canvas) {
            const values = pieces.map((p) => p.toLowerCase().split('x').map(Number));
            return values.every(
              (p) => p.length === 2 && p.every((n) => Number.isFinite(n) && n > 0),
            )
              ? values
              : s;
          }
          const values = pieces.map((v) => numeric(v, field.kind === 'IntListEditor'));
          return values.every((v) => typeof v === 'number') ? values : s;
        }}
        onChange={set}
      />
    );
  }
  if (['FloatEditor', 'IntEditor', 'OptIntEditor', 'SciEditor'].includes(field.kind)) {
    const integer = field.kind === 'IntEditor' || field.kind === 'OptIntEditor';
    return (
      <ValueInput
        aria-label={label}
        value={value}
        type={field.kind === 'SciEditor' ? 'text' : 'number'}
        inputMode="decimal"
        min={a[0] !== undefined && typeof a[0] === 'number' ? a[0] : undefined}
        max={a[1]}
        step={k.step ?? a[2] ?? (integer ? 1 : 'any')}
        placeholder={typeof a[0] === 'string' ? a[0] : k.optional ? 'Unset' : ''}
        parse={(s) => numeric(s, integer)}
        onChange={set}
      />
    );
  }
  if (['TextEditor', 'OptTextEditor'].includes(field.kind))
    return (
      <ValueInput
        aria-label={label}
        value={value}
        placeholder={a[0] || ''}
        parse={(s) =>
          field.kind === 'OptTextEditor' && !s.trim()
            ? null
            : k.strip === false || a[1] === false
              ? s
              : s.trim()
        }
        onChange={set}
      />
    );
  throw Error('Unimplemented editor: ' + field.kind);
}

export default function FieldRow({ field }: { field: Field }) {
  const { state, configRevision } = useStore((value) => ({
    state: value.state,
    configRevision: value.configRevision,
  }));
  if (!state) return null;
  const disabled = state.validation.enabled[field.key] === false;
  const full = [
    'CurriculumEditor',
    'SubsetEditor',
    'BlockLRRangesEditor',
    'ResolutionStagesEditor',
  ].includes(field.kind);
  if (
    state.flat['resolution_schedule.enabled'] &&
    state.flat['resolution_schedule.unit'] !== 'epochs' &&
    ['train.epochs', 'train.save_every_epochs'].includes(field.key)
  )
    return null;
  if (
    state.flat['resolution_schedule.enabled'] &&
    state.flat['resolution_schedule.unit'] === 'epochs' &&
    field.key === 'train.max_steps'
  )
    return null;
  if (field.key === 'dataset.num_repeats' && state.flat['dataset.target_images'] != null)
    return null;
  return (
    <fieldset
      className={'field-row ' + (full ? 'field-full ' : '') + (field.inline ? 'field-inline' : '')}
      disabled={disabled}
      title={field.tooltip}
      data-field={field.key}
    >
      {!field.inline && <label className="field-label">{field.label}</label>}
      <div className="field-control">
        <Editor key={configRevision} field={field} value={state.flat[field.key]} />
      </div>
    </fieldset>
  );
}
