// A ready renderer can still have a background-only compositor frame on Windows.
// Metrics always contain text, even when no graph samples have been recorded.
function hasContent(image) {
  const pixels = image.toBitmap();
  for (let i = 4; i < pixels.length; i += 4) {
    if (pixels[i] !== pixels[0] || pixels[i + 1] !== pixels[1] ||
        pixels[i + 2] !== pixels[2]) return true;
  }
  return false;
}

function captureMetricsPng(contents, timeoutMs = 15000) {
  return new Promise((resolve, reject) => {
    let repaint;
    const cleanup = () => {
      clearTimeout(timeout);
      clearTimeout(repaint);
      contents.removeListener('paint', paint);
      contents.removeListener('destroyed', destroyed);
    };
    const fail = (error) => { cleanup(); reject(error); };
    const destroyed = () => fail(Error('Metrics window closed before capture'));
    const paint = (_event, _dirty, image) => {
      try {
        if (hasContent(image)) {
          const png = image.toPNG();
          cleanup();
          resolve(png);
        } else {
          // Give the compositor a frame interval before requesting another paint.
          clearTimeout(repaint);
          repaint = setTimeout(() => contents.invalidate(), 16);
        }
      } catch (error) { fail(error); }
    };
    const timeout = setTimeout(() => fail(Error('Metrics capture did not paint content')), timeoutMs);
    contents.on('paint', paint);
    contents.once('destroyed', destroyed);
    try { contents.invalidate(); } catch (error) { fail(error); }
  });
}

module.exports = { captureMetricsPng };
