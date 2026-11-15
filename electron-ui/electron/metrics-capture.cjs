// A ready renderer can still have a background-only compositor frame on Windows.
// Metrics always contain text, even when no graph samples have been recorded.
function hasContent(image) {
  const pixels = image.toBitmap();
  for (let i = 4; i < pixels.length; i += 4) {
    if (pixels[i] !== pixels[0] || pixels[i + 1] !== pixels[1] || pixels[i + 2] !== pixels[2])
      return true;
  }
  return false;
}

function graphsPainted(image, plots) {
  if (!plots) return true;
  if (plots.charts?.length !== 4 || !(plots.width > 0) || !(plots.height > 0)) return false;
  const size = image.getSize();
  const sx = size.width / plots.width,
    sy = size.height / plots.height;
  return plots.charts.every((r) => {
    // Stay inside the canvas: section borders/headings must not satisfy this test.
    const rect = {
      x: Math.ceil((r.x + 4) * sx),
      y: Math.ceil((r.y + 4) * sy),
      width: Math.floor((r.width - 8) * sx),
      height: Math.floor((r.height - 8) * sy),
    };
    return (
      rect.width > 0 &&
      rect.height > 0 &&
      rect.x >= 0 &&
      rect.y >= 0 &&
      rect.x + rect.width <= size.width &&
      rect.y + rect.height <= size.height &&
      hasContent(image.crop(rect))
    );
  });
}

function captureMetricsPng(contents, timeoutMs = 15000, plots) {
  return new Promise((resolve, reject) => {
    let repaint;
    const cleanup = () => {
      clearTimeout(timeout);
      clearTimeout(repaint);
      contents.removeListener('paint', paint);
      contents.removeListener('destroyed', destroyed);
    };
    const fail = (error) => {
      cleanup();
      reject(error);
    };
    const destroyed = () => fail(Error('Metrics window closed before capture'));
    const paint = (_event, _dirty, image) => {
      try {
        if (hasContent(image) && graphsPainted(image, plots)) {
          const png = image.toPNG();
          cleanup();
          resolve(png);
        } else {
          // Give the compositor a frame interval before requesting another paint.
          clearTimeout(repaint);
          repaint = setTimeout(() => contents.invalidate(), 16);
        }
      } catch (error) {
        fail(error);
      }
    };
    const timeout = setTimeout(
      () => fail(Error('Metrics capture did not paint content in every graph')),
      timeoutMs,
    );
    contents.on('paint', paint);
    contents.once('destroyed', destroyed);
    try {
      contents.invalidate();
    } catch (error) {
      fail(error);
    }
  });
}

module.exports = { captureMetricsPng };
