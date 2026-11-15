const { test } = require('node:test');
const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const { captureMetricsPng } = require('../electron/metrics-capture.cjs');

const frame = (pixels) => ({
  toBitmap: () => Buffer.from(pixels),
  toPNG: () => Buffer.from('rendered PNG'),
});
const blank = frame([16, 17, 22, 255, 16, 17, 22, 255]);
const rendered = frame([16, 17, 22, 255, 200, 210, 220, 255]);

test('background-only Windows frames are retried until metrics paint', async () => {
  const contents = new EventEmitter();
  let paints = 0;
  contents.invalidate = () => contents.emit('paint', {}, {}, ++paints < 3 ? blank : rendered);
  assert.equal((await captureMetricsPng(contents)).toString(), 'rendered PNG');
  assert.equal(paints, 3);
  assert.equal(contents.listenerCount('paint'), 0);
  assert.equal(contents.listenerCount('destroyed'), 0);
});

test('a renderer that stays blank fails instead of exporting a blank PNG', async () => {
  const contents = new EventEmitter();
  contents.invalidate = () => contents.emit('paint', {}, {}, blank);
  await assert.rejects(captureMetricsPng(contents, 25), /did not paint content/);
  assert.equal(contents.listenerCount('paint'), 0);
});

test('closing the capture window rejects and removes paint listeners', async () => {
  const contents = new EventEmitter();
  contents.invalidate = () => contents.emit('destroyed');
  await assert.rejects(captureMetricsPng(contents), /closed before capture/);
  assert.equal(contents.listenerCount('paint'), 0);
});

const plots = {
  width: 100,
  height: 100,
  charts: [0, 1, 2, 3].map((i) => ({ x: i * 20, y: 10, width: 20, height: 70 })),
};
const withGraphs = (count) => ({
  ...rendered,
  getSize: () => ({ width: 100, height: 100 }),
  crop: (rect) => (Math.floor(rect.x / 20) < count ? rendered : blank),
});

test('headings or three graphs are insufficient: all four plots must paint', async () => {
  const contents = new EventEmitter();
  let paints = 0;
  contents.invalidate = () => contents.emit('paint', {}, {}, withGraphs(Math.min(4, paints++)));
  assert.equal((await captureMetricsPng(contents, 1000, plots)).toString(), 'rendered PNG');
  assert.equal(paints, 5);
});

test('a permanently missing graph fails instead of saving an incomplete screenshot', async () => {
  const contents = new EventEmitter();
  contents.invalidate = () => contents.emit('paint', {}, {}, withGraphs(3));
  await assert.rejects(captureMetricsPng(contents, 25, plots), /every graph/);
});
