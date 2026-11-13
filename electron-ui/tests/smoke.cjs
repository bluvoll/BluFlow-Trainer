const { _electron: electron } = require('playwright');
const { verifyStartup } = require('./startup.cjs');
const { spawnSync } = require('node:child_process');
const fs = require('node:fs'),
  path = require('node:path'),
  assert = require('node:assert/strict');
const root = path.resolve(__dirname, '../..');
const artifacts = path.join(root, '.artifacts/electron-smoke');
fs.mkdirSync(artifacts, { recursive: true });
const profile = fs.mkdtempSync(path.join(artifacts, 'profile-'));
const configs = path.join(profile, 'configs');
fs.mkdirSync(configs);
fs.writeFileSync(
  path.join(configs, 'smoke.toml'),
  `[train]\nmodel_family="moss_image"\nmodel_path="unused-model"\nrun_name="electron-smoke"\noutput_dir="${profile.replaceAll('\\', '/')}/output"\n[dataset]\npath="${profile.replaceAll('\\', '/')}/images"\n[adapter]\nkind="lora"\n`,
);
fs.mkdirSync(path.join(profile, 'images'));
const python =
  process.env.BLUFLOW_PYTHON ||
  path.join(root, 'venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const fixture = spawnSync(
  python,
  [
    '-c',
    `
import sys
from pathlib import Path
import torch
from safetensors.torch import save_file
from trainer.gui import bridge
from trainer.training.config import load_config
from trainer.modeling.checkpoint_metadata import encode_settings
folder=Path(sys.argv[1])
config=folder/'utility-recipe.toml'
bridge.write_toml(config,bridge.defaults()|{'dataset.path':str(folder/'images'),'train.model_family':'moss_image','adapter.kind':'lora'})
tensors={}
for p in ('q','k','v'):
    prefix='diffusion_model.layers.0.proj.'+p
    tensors[prefix+'.lora_down.weight']=torch.arange(8,dtype=torch.float32).reshape(2,4)
    tensors[prefix+'.lora_up.weight']=torch.arange(8,dtype=torch.float32).reshape(4,2)
    tensors[prefix+'.alpha']=torch.tensor(1.)
save_file(tensors,folder/'utility-adapter.safetensors',metadata={'training_config':encode_settings(load_config(config))})
`,
    profile,
  ],
  { cwd: root, encoding: 'utf8', windowsHide: true },
);
assert.equal(fixture.status, 0, fixture.stderr);
let app;
const electronEnv = { ...process.env };
delete electronEnv.ELECTRON_RUN_AS_NODE;
(async () => {
  await verifyStartup();
  app = await electron.launch({
    args: [
      path.join(root, 'electron-ui'),
      '--test',
      '--no-update-check',
      '--no-dependency-check',
      ...(process.platform === 'linux' && process.env.CI ? ['--no-sandbox'] : []),
    ],
    cwd: root,
    env: {
      ...electronEnv,
      BLUFLOW_PYTHON:
        process.env.BLUFLOW_PYTHON ||
        path.join(root, 'venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python'),
      BLUFLOW_TEST_PROFILE: profile,
      BLUFLOW_TEST_PREFS: path.join(profile, 'gui.json'),
      BLUFLOW_TEST_CONFIG_DIR: configs,
    },
    timeout: 60000,
  });
  const page = await app.firstWindow();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.getByTestId('app').waitFor({ timeout: 60000 });
  await page.locator('[data-field="train.run_name"] input').waitFor();
  await page.screenshot({ path: path.join(artifacts, 'workspace.png') });
  const fields = await page.evaluate(
    async () => Object.keys((await window.bluflow.request('bootstrap')).fields).length,
  );
  assert.equal(fields, 201);
  const qkvSelect = page.locator('[data-field="train.moss_qkv_format"] select');
  assert.equal(
    await qkvSelect.inputValue(),
    '0',
    'Keyword-based choice values preserve the auto default',
  );
  await qkvSelect.selectOption({ label: 'separated' });
  await page.waitForFunction(
    async () =>
      (await window.bluflow.request('state')).flat['train.moss_qkv_format'] === 'separated',
  );
  await qkvSelect.selectOption('0');
  assert.ok(
    await page.evaluate(async () =>
      Array.isArray(
        (await window.bluflow.request('bootstrap')).state.flat['dataset.caption.protected_tags'],
      ),
    ),
  );
  await page.locator('[data-field="train.run_name"] input').fill('electron-smoke-edited');
  await page.getByRole('button', { name: 'Save', exact: true }).click();
  await page.waitForFunction(
    async () =>
      !!(await window.bluflow.request('state')).path?.endsWith('electron-smoke-edited.toml'),
  );
  assert.ok(
    fs.existsSync(path.join(configs, 'smoke.toml')),
    'Renaming must preserve the original config',
  );
  assert.ok(fs.existsSync(path.join(configs, 'electron-smoke-edited.toml')));
  await page.getByRole('button', { name: 'Save & add to queue', exact: true }).click();
  await page.waitForFunction(
    async () => (await window.bluflow.request('state')).queue.length === 1,
  );
  await page.getByRole('button', { name: /Dataset.*sections/ }).click();
  await page.locator('[data-field="dataset.subsets"]').waitFor();
  await page.getByRole('button', { name: 'Add subset', exact: true }).click();
  await page.getByLabel('Subset 1 folder', { exact: true }).fill('日本語/dataset');
  await page.waitForFunction(
    async () =>
      (await window.bluflow.request('state')).flat['dataset.subsets']?.[0]?.path ===
      '日本語/dataset',
  );
  await page.waitForFunction(
    () => document.querySelector('[data-field="dataset.path"]')?.disabled === true,
  );
  await page.getByLabel('Search settings', { exact: true }).fill('text cache batch');
  await page.locator('[data-field="train.text_cache_batch_size"]').waitFor();
  await page.getByRole('button', { name: 'Clear search', exact: true }).click();
  await page.getByRole('button', { name: 'Preferences', exact: true }).first().click();
  await page.getByRole('dialog', { name: 'Preferences', exact: true }).waitFor();
  await page.screenshot({ path: path.join(artifacts, 'preferences.png') });
  await page
    .getByRole('checkbox', { name: 'Enable advanced experimental settings', exact: true })
    .check();
  await page.getByRole('button', { name: 'Save preferences', exact: true }).click();
  const renderedFields = new Set();
  for (const category of ['Dataset', 'Training', 'Optimizer', 'Method', 'Experimental']) {
    await page.getByRole('button', { name: new RegExp('^' + category + '.*sections') }).click();
    await page.waitForFunction(
      (category) => document.querySelector('.content-heading h2')?.textContent === category,
      category,
    );
    for (const key of await page
      .locator('[data-field]')
      .evaluateAll((nodes) => nodes.map((n) => n.dataset.field)))
      renderedFields.add(key);
  }
  assert.equal(renderedFields.size, fields, 'Every Python field has a rendered editor');
  // Uneven sections must fill the available width, including when one is collapsed.
  await page.getByRole('button', { name: /^Method.*sections/ }).click();
  await page.getByRole('button', { name: 'Expand all', exact: true }).click();
  await page.setViewportSize({ width: 3840, height: 1500 });
  await page.waitForFunction(() => {
    const cards = [...document.querySelectorAll('.settings-group')];
    return (
      cards.length === 6 &&
      new Set(cards.map((card) => Math.round(card.getBoundingClientRect().x))).size === 6
    );
  });
  await page.getByRole('button', { name: 'Full Finetuning', exact: true }).click();
  const packed = async () => {
    await page.waitForFunction(() => {
      const grid = document.querySelector('.settings-groups');
      const cards = [...grid.children].map((card) => card.getBoundingClientRect());
      const overlaps = cards.some((a, i) =>
        cards
          .slice(i + 1)
          .some(
            (b) =>
              Math.min(a.right, b.right) - Math.max(a.left, b.left) > 1 &&
              Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > 1,
          ),
      );
      return (
        !overlaps &&
        Math.abs(
          Math.max(...cards.map((card) => card.right)) - grid.getBoundingClientRect().right,
        ) < 2
      );
    });
  };
  await packed();
  await page.screenshot({ path: path.join(artifacts, 'settings-wide.png') });
  const rankInput = page.locator('[data-field="adapter.rank"] input');
  await rankInput.focus();
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.waitForFunction(() => {
    const columns = new Set(
      [...document.querySelectorAll('.settings-group')].map((card) =>
        Math.round(card.getBoundingClientRect().x),
      ),
    ).size;
    return columns >= 2 && columns < 6;
  });
  await packed();
  assert.ok(
    await rankInput.evaluate((input) => input === document.activeElement),
    'Resizing preserves editor focus',
  );
  const columns = await page
    .locator('.settings-group')
    .evaluateAll(
      (cards) => new Set(cards.map((card) => Math.round(card.getBoundingClientRect().x))).size,
    );
  assert.ok(columns >= 2 && columns < 6, 'Sections reflow into fewer columns in a narrower panel');
  await page.screenshot({ path: path.join(artifacts, 'settings-reflow.png') });
  await page.getByRole('button', { name: /^Dataset.*sections/ }).click();
  await page.locator('[data-field="dataset.resolutions"] input').fill('512, 768, 1024');
  await page.waitForFunction(
    async () => (await window.bluflow.request('state')).flat['dataset.resolutions']?.length === 3,
  );
  await page.getByRole('button', { name: /^Method.*sections/ }).click();
  await page.locator('[data-field="flow.logit_mean"] input').fill('-0.2');
  await page.locator('[data-field="flow.sigmoid_scale"] input').fill('1.5');
  await page.locator('[data-field="flow.normal_low_snr"] input').check();
  await page.getByTestId('timestep-2d').waitFor();
  await page.locator('[data-field="flow.cabal_shift"] input').check();
  await page.getByTestId('timestep-3d').waitFor();
  await page
    .getByLabel('Resolution shifts', { exact: true })
    .getByText('512px · shift 2.750', { exact: true })
    .waitFor();
  await page
    .getByLabel('Resolution shifts', { exact: true })
    .getByText('1024px · shift 6.000', { exact: true })
    .waitFor();
  await page.locator('.distribution').scrollIntoViewIfNeeded();
  const hoverPoint = await page.getByTestId('timestep-3d').evaluate((canvas) => {
    const rect = canvas.getBoundingClientRect(),
      ctx = canvas.getContext('2d');
    const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    const sx = canvas.width / rect.width,
      sy = canvas.height / rect.height;
    for (let y = Math.ceil(40 * sy); y < canvas.height - 65 * sy; y++) {
      for (let x = Math.ceil(75 * sx); x < canvas.width * 0.65; x++) {
        const i = (y * canvas.width + x) * 4;
        if (
          Math.abs(pixels[i] - 180) +
            Math.abs(pixels[i + 1] - 157) +
            Math.abs(pixels[i + 2] - 223) <
          10
        )
          return { x: rect.left + x / sx, y: rect.top + y / sy };
      }
    }
    return null;
  });
  assert.ok(hoverPoint, 'The 512px distribution curve is painted');
  await page.mouse.move(hoverPoint.x, hoverPoint.y);
  await page
    .getByRole('tooltip')
    .getByText(/512px · shift 2.750/)
    .waitFor();
  await page.screenshot({ path: path.join(artifacts, 'timesteps-cabal.png') });
  await page
    .locator('.distribution')
    .screenshot({ path: path.join(artifacts, 'timesteps-cabal-detail.png') });
  await page.mouse.move(0, 0);
  await page.setViewportSize({ width: 1080, height: 900 });
  await page.waitForFunction(() => {
    const canvas = document.querySelector('[data-testid="timestep-3d"]');
    return (
      canvas &&
      Math.abs(canvas.width - Math.round(canvas.getBoundingClientRect().width * devicePixelRatio)) <
        2
    );
  });
  await page.locator('.distribution').scrollIntoViewIfNeeded();
  await page
    .locator('.distribution')
    .screenshot({ path: path.join(artifacts, 'timesteps-compact.png') });
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.locator('[data-field="flow.cabal_shift_offset"] input').fill('-10');
  await page
    .locator('.distribution')
    .getByText(/Cabal shift must be positive/)
    .waitFor();
  assert.equal(
    await page.getByTestId('timestep-3d').count(),
    0,
    'Invalid settings clear the old curves',
  );
  await page.locator('[data-field="flow.cabal_shift_offset"] input').fill('0.25');
  await page
    .getByLabel('Resolution shifts', { exact: true })
    .getByText('1024px · shift 6.250', { exact: true })
    .waitFor();
  await page.locator('[data-field="flow.flux_shift"] input').check();
  await page.waitForFunction(async () => {
    const state = await window.bluflow.request('state');
    return (
      state.flat['flow.flux_shift'] &&
      !state.flat['flow.cabal_shift'] &&
      state.flat['flow.shift'] === null
    );
  });
  await page
    .getByLabel('Resolution shifts', { exact: true })
    .getByText('1024px · shift 3.158', { exact: true })
    .waitFor();
  await page.locator('.distribution').scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(artifacts, 'timesteps-flux.png') });
  await page.getByRole('button', { name: /^Dataset.*sections/ }).click();
  await page.locator('[data-field="dataset.resolutions"] input').fill('768');
  await page.waitForFunction(
    async () =>
      (await window.bluflow.request('state')).validation.resolutionDistributions.length === 1,
  );
  await page.getByRole('button', { name: /^Method.*sections/ }).click();
  await page.waitForFunction(
    () => document.querySelectorAll('[aria-label="Resolution shifts"] > span').length === 1,
  );
  assert.equal(
    await page.getByLabel('Resolution shifts', { exact: true }).locator('span').count(),
    1,
  );
  await page.locator('[data-field="flow.flux_shift"] input').uncheck();
  await page.getByTestId('timestep-2d').waitFor();
  await page
    .locator('[data-field="flow.timestep_sample_method"] select')
    .selectOption({ label: 'uniform' });
  await page.waitForFunction(
    async () => (await window.bluflow.request('state')).flat['flow.sigmoid_scale'] === 1,
  );
  assert.ok(await page.locator('[data-field="flow.normal_low_snr"] input').isEnabled());
  await page
    .locator('[data-field="flow.timestep_sample_method"] select')
    .selectOption({ label: 'moss' });
  await page.waitForFunction(
    () => document.querySelector('[data-field="flow.cabal_shift"]')?.disabled === true,
  );
  assert.ok(await page.locator('[data-field="flow.normal_low_snr"] input').isEnabled());
  assert.ok(await page.locator('[data-field="flow.logit_mean"] input').isDisabled());
  await page.getByRole('button', { name: 'Save', exact: true }).click();
  assert.match(
    fs.readFileSync(path.join(configs, 'electron-smoke-edited.toml'), 'utf8'),
    /normal_low_snr = true/,
  );
  await page.getByText('Utilities', { exact: true }).click();
  await page.getByRole('button', { name: 'Resize LoRA rank (SVD)', exact: true }).click();
  await page.getByRole('dialog', { name: 'Resize LoRA rank', exact: true }).waitFor();
  assert.equal(
    await page.getByLabel('Frobenius norm retention', { exact: true }).inputValue(),
    '0.99',
  );
  await page.screenshot({ path: path.join(artifacts, 'rank-resize.png') });
  await page
    .getByLabel('Utility source', { exact: true })
    .fill(path.join(profile, 'utility-adapter.safetensors'));
  await page
    .getByLabel('Utility destination', { exact: true })
    .fill(path.join(profile, 'resized.safetensors'));
  await page.getByRole('button', { name: 'Write output', exact: true }).click();
  await page
    .locator('.tool-result')
    .getByText('Output written', { exact: true })
    .waitFor({ timeout: 30000 });
  assert.ok(fs.existsSync(path.join(profile, 'resized.safetensors')));
  await page.getByRole('button', { name: 'Close', exact: true }).click();
  for (const [name, target] of [
    ['Edit LoRA alpha / strength', 'alpha.safetensors'],
    ['Convert MOSS Q/K/V', 'fused.safetensors'],
    ['Recover training TOML', 'recovered.toml'],
  ]) {
    await page.getByText('Utilities', { exact: true }).click();
    await page.getByRole('button', { name, exact: true }).click();
    await page
      .getByLabel('Utility source', { exact: true })
      .fill(path.join(profile, 'utility-adapter.safetensors'));
    await page.getByLabel('Utility destination', { exact: true }).fill(path.join(profile, target));
    // Typing in a dialog must preserve focus across controlled-input updates.
    if (name === 'Edit LoRA alpha / strength') {
      await page.getByRole('button', { name: 'Inspect module groups', exact: true }).click();
      await page
        .getByText('3 modules · 0 DoRA tensors', { exact: true })
        .waitFor({ timeout: 30000 });
    }
    await page
      .getByRole('button', {
        name: name === 'Recover training TOML' ? 'Recover TOML' : 'Write output',
        exact: true,
      })
      .click();
    await page
      .locator('.tool-result')
      .getByText('Output written', { exact: true })
      .waitFor({ timeout: 30000 });
    assert.ok(fs.existsSync(path.join(profile, target)));
    await page.getByRole('button', { name: 'Close', exact: true }).click();
  }
  await page.getByText('Panels', { exact: true }).click();
  await page
    .locator('.panel-menu-row')
    .filter({ has: page.getByRole('button', { name: 'Live Metrics', exact: true }) })
    .getByRole('button', { name: 'Live Metrics', exact: true })
    .click();
  await page.getByTestId('metrics').waitFor({ state: 'visible' });
  const metrics = {
    records: Array.from({ length: 80 }, (_, i) => ({
      step: i + 1,
      total: 80,
      loss: 0.15 - 0.001 * i,
      ema: 0.16 - 0.001 * i,
      mse: 0.14 - 0.001 * i,
      hf: 0.01,
      lr: 0.0001,
      sit: 1.8,
      peak: [8.1],
    })),
    latest: { step: 80, total: 80, loss: 0.071, lr: 0.0001, sit: 1.8, peak: [8.1] },
    reports: ['mode  MOSS LoRA', 'dataset  80 cached samples'],
    startedAt: null,
    elapsed: 144000,
  };
  await app.evaluate(({ BrowserWindow }, metrics) => {
    BrowserWindow.getAllWindows()[0].webContents.send('bluflow:event', {
      event: 'metrics_reset',
      metrics,
    });
  }, metrics);
  await page.getByTestId('elapsed').getByText('00:02:24', { exact: true }).waitFor();
  const memoryChart = page.getByRole('img', {
    name: 'Peak allocated VRAM · GiB chart',
    exact: true,
  });
  const box = await memoryChart.boundingBox();
  assert.ok(box, 'Memory chart is visible');
  // A constant memory line sits at the vertical midpoint of the plot area.
  await page.mouse.move(box.x + box.width * 0.55, box.y + box.height / 2 - 7.5);
  await page
    .getByRole('tooltip')
    .getByText(/rank 0 · step/)
    .waitFor();
  await page.mouse.wheel(0, -240);
  await page.waitForFunction(
    () =>
      document.querySelector('canvas[aria-label="Peak allocated VRAM · GiB chart"]')?.dataset
        .zoomed === 'true',
  );
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.65, box.y + box.height / 2 - 7.5, { steps: 5 });
  await page.mouse.up();
  await page
    .getByRole('button', { name: 'Reset Peak allocated VRAM · GiB view', exact: true })
    .click();
  await page.waitForFunction(
    () =>
      document.querySelector('canvas[aria-label="Peak allocated VRAM · GiB chart"]')?.dataset
        .zoomed === 'false',
  );
  await page.screenshot({ path: path.join(artifacts, 'metrics.png') });
  await app.evaluate(
    async ({ app }, { metrics, target }) => {
      await app.__bluflowCaptureMetrics({ metrics, path: target });
    },
    { metrics, target: path.join(profile, 'metrics-final.png') },
  );
  assert.ok(
    fs.statSync(path.join(profile, 'metrics-final.png')).size > 5000,
    'Offscreen final metrics PNG',
  );
  fs.copyFileSync(
    path.join(profile, 'metrics-final.png'),
    path.join(artifacts, 'metrics-final.png'),
  );
  await page.getByText('Panels', { exact: true }).click();
  await page
    .locator('.panel-menu-row')
    .filter({ has: page.getByRole('button', { name: 'Console', exact: true }) })
    .getByRole('button', { name: 'Console', exact: true })
    .click();
  await page.getByRole('log', { name: 'Process console' }).waitFor({ state: 'visible' });
  await page.getByRole('checkbox', { name: 'Follow output', exact: true }).uncheck();
  await page.getByLabel('Console history position', { exact: true }).waitFor();
  assert.match(
    await page.getByRole('log', { name: 'Process console' }).innerText(),
    /Saved final metrics screenshot/,
  );
  await page.getByRole('button', { name: 'Clear console', exact: true }).click();
  await page.waitForFunction(
    async () => (await window.bluflow.request('console.read')).total === 0,
  );
  await page
    .getByRole('log', { name: 'Process console' })
    .getByText('Process output appears here.', { exact: true })
    .waitFor();
  await page.setViewportSize({ width: 1080, height: 760 });
  await page.screenshot({ path: path.join(artifacts, 'compact.png') });
  await page.getByText('Panels', { exact: true }).click();
  await page.getByTitle('Float Run Queue', { exact: true }).click();
  await page.waitForFunction(
    async () => ((await window.bluflow.preferences()).layout?.floatingGroups?.length || 0) > 0,
  );
  await page.getByRole('button', { name: 'Reset workspace layout', exact: true }).click();
  await page.waitForFunction(
    async () => !(await window.bluflow.preferences()).layout?.floatingGroups?.length,
  );
  assert.deepEqual(errors, [], 'Renderer errors');
  const closingCapture = path.join(profile, 'metrics-on-close.png');
  await app.evaluate(
    ({ app }, { metrics, target }) => {
      void app.__bluflowCaptureMetrics({ metrics, path: target });
    },
    { metrics, target: closingCapture },
  );
  await app.close();
  app = undefined;
  assert.ok(
    fs.statSync(closingCapture).size > 5000,
    'Closing waits for pending final metrics export',
  );
  console.log(
    'Electron smoke passed: all 201 editors, sampling controls and resolution previews, config save, immutable queue, subsets, search, preferences, four utility outputs, graph hover/zoom/pan/reset, console history/clear, offscreen PNG, compact layout and floating-panel persistence.',
  );
})()
  .catch(async (error) => {
    if (app) {
      const pages = await app.windows();
      if (pages[0])
        await pages[0].screenshot({ path: path.join(artifacts, 'failure.png') }).catch(() => {});
    }
    console.error(error);
    process.exitCode = 1;
  })
  .finally(async () => {
    if (app) await app.close();
  });
