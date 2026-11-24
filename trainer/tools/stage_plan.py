"""Read-only exposure preview: python -m trainer.tools.stage_plan CONFIG --gpus 2."""
import argparse
from dataclasses import replace
import warnings

from ..training.stage_plan import resolve_stage_plan, format_stage_plan


def inspect_plan(cfg, world=1):
    from ..data.dataset import MageFlowDataset
    if not cfg.resolution_schedule.enabled:
        raise ValueError('Enable staged resolutions to calculate an exposure plan')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        dataset = MageFlowDataset(replace(cfg.dataset, tier_collapse='repeat'))
    counts = {s.resolution: len(dataset.epoch_indices(0, s.resolution))
              for s in cfg.resolution_schedule.stages}
    rows = resolve_stage_plan(cfg.resolution_schedule,
                              cfg.train.epochs if cfg.resolution_schedule.epoch_based else cfg.train.max_steps,
                              counts, cfg.train.batch_size, world, cfg.train.gradient_accumulation_steps)
    notes = list(dict.fromkeys(str(w.message) for w in caught))
    restart_schedule = cfg.schedule if cfg.resolution_schedule.restart_lr_each_stage else None
    text = format_stage_plan(rows, restart_schedule) + ''.join(f'\nDataset: {n}' for n in notes)
    return dict(stages=rows, warnings=notes, text=text)


def main():
    from ..training.config import load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--gpus', type=int, default=1, help='Planned number of training GPUs; no GPU is used')
    args = parser.parse_args()
    print(inspect_plan(load_config(args.config), args.gpus)['text'])


if __name__ == '__main__':
    main()
