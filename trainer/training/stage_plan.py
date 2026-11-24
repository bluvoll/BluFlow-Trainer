"""Resolve stage exposure once using the same rounding rules as the sampler."""
import math

from .resolution_schedule import stage_warmup_steps


def epoch_samples(entries, epochs):
    """Full traversals followed by a deterministic, shuffled partial traversal."""
    whole = math.floor(epochs)
    fraction = epochs - whole
    return [entries] * whole + ([max(1, math.ceil(fraction * entries - 1e-9))] if fraction else [])


def resolve_stage_plan(schedule, total, counts, batches, world=1, accumulation=1):
    if type(world) is not int or world < 1 or type(accumulation) is not int or accumulation < 1:
        raise ValueError('GPU count and accumulation must be positive integers')
    rows = []
    for stage, duration in zip(schedule.stages, schedule.budgets(total)):
        n, batch = counts[stage.resolution], batches[stage.resolution]
        if n <= 0 or type(batch) is not int or batch <= 0:
            raise ValueError('Every stage needs eligible samples and a positive batch size')
        global_batch = batch * world
        effective = global_batch * accumulation
        slices = epoch_samples(n, duration) if schedule.epoch_based else []
        microbatches = [math.ceil(size / global_batch) for size in slices]
        updates = (sum(math.ceil(m / accumulation) for m in microbatches)
                   if schedule.epoch_based else duration)
        requested = sum(slices) if schedule.epoch_based else updates * effective
        images = (sum(microbatches) * global_batch if world > 1 else requested) if schedule.epoch_based else requested
        notes = []
        if effective > n / 4:
            notes.append("Effective batch exceeds 25% of this tier's weighted dataset; epochs contain few updates.")
        if images > requested:
            notes.append(f'DDP fills the last batch: {images - requested} extra presentations ({(images / requested - 1):.1%}).')
        rows.append(dict(resolution=stage.resolution, unit=schedule.unit, epochs=duration if schedule.epoch_based else None,
                         steps=updates, batch=batch, effective_batch=effective, entries=n,
                         images=images, equivalent_epochs=images / n, padding=images - requested,
                         lr_mul=stage.lr_mul, warnings=notes, epoch_samples=slices,
                         epoch_microbatches=microbatches))
    if rows:
        low = min(r['equivalent_epochs'] for r in rows)
        for row in rows:
            if row['equivalent_epochs'] >= 2 * low:
                row['warnings'].append('This stage has at least twice the dataset exposure of the least-exposed stage.')
    return rows


def format_stage_plan(rows, restart_schedule=None):
    lines = ['Staged exposure plan (dataset entries include repeats/weighting):']
    header = 'px | epochs requested | batch/GPU | effective batch | entries | presentations | passes | updates | LR x'
    if restart_schedule is not None:
        lines.append(f'{restart_schedule.kind} restarts at each resolution with up to {restart_schedule.warmup_steps} warmup updates.')
        header += ' | warmup updates'
    lines.append(header)
    for r in rows:
        epochs = f"{r['epochs']:g}" if r['epochs'] is not None else '-'
        lines.append(f"{r['resolution']} | {epochs} | {r['batch']} | {r['effective_batch']} | {r['entries']} | "
                     f"{r['images']} | {r['equivalent_epochs']:.3f} | {r['steps']} | {r['lr_mul']:g}")
        if restart_schedule is not None:
            lines[-1] += f" | {stage_warmup_steps(restart_schedule.warmup_steps, r['steps'])}"
        lines.extend(f"  Note: {note}" for note in r['warnings'])
    lines.append(f"Total: {sum(r['steps'] for r in rows)} planned updates, {sum(r['images'] for r in rows)} presentations.")
    lines.append('Step mode counts successful updates; skipped attempts add exposure. Epoch modes do not extend on skipped updates. '
                 'Increasing GPU count increases exposure in step mode; epoch modes change update counts and DDP padding.')
    return '\n'.join(lines)
