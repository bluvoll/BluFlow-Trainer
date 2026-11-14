// Shared setup/launch supervisor for both platforms. It never installs Python packages at setup.
const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const root = path.resolve(__dirname, ".."),
  ui = path.join(root, "electron-ui");
const python =
  process.env.BLUFLOW_PYTHON ||
  path.join(
    root,
    "venv",
    process.platform === "win32" ? "Scripts/python.exe" : "bin/python",
  );
function run(command, args, cwd = root) {
  const result = spawnSync(command, args, {
    cwd,
    stdio: "inherit",
    windowsHide: true,
  });
  if (result.error) throw result.error;
  if (result.status !== 0)
    throw Error(command + " exited with code " + result.status);
}
function npm(args) {
  const executable =
    process.env.npm_execpath ||
    path.join(
      path.dirname(process.execPath),
      "node_modules/npm/bin/npm-cli.js",
    );
  if (fs.existsSync(executable))
    run(process.execPath, [executable, ...args], ui);
  else if (process.platform === "win32")
    run(process.env.ComSpec || "cmd.exe", ["/d", "/c", "npm", ...args], ui);
  else run("npm", args, ui);
}
function setup() {
  if (!fs.existsSync(python))
    throw Error(
      "Trainer environment is missing. Run " +
        (process.platform === "win32" ? "install.bat" : "./install.sh") +
        " first, or set BLUFLOW_PYTHON.",
    );
  const [major, minor] = process.versions.node.split(".").map(Number);
  if (major < 22 || (major === 22 && minor < 12))
    throw Error("Node.js 22.12 or newer is required.");
  run(python, ["-m", "trainer.electron_backend", "--check"]);
  npm(["ci"]);
  installRuntime();
  npm(["run", "build"]);
}
function installRuntime() {
  run(
    process.execPath,
    [path.join(ui, "node_modules/electron/install.js")],
    ui,
  );
}
try {
  const [action, ...args] = process.argv.slice(2);
  if (action === "setup") {
    setup();
    process.exit(0);
  }
  if (action !== "start")
    throw Error("Usage: node scripts/electron.cjs setup|start [GUI options]");
  if (!fs.existsSync(python))
    throw Error("Run the trainer installer first, then setup-electron.");
  const stamp = path.join(ui, "dist/build-stamp.json");
  if (
    !fs.existsSync(stamp) ||
    JSON.parse(fs.readFileSync(stamp, "utf8")).fingerprint !==
      require("../electron-ui/scripts/stamp.cjs").fingerprint()
  )
    throw Error(
      "Electron UI is missing or out of date. Run setup-electron" +
        (process.platform === "win32" ? ".bat" : ".sh") +
        ".",
    );
  let restarts = 0;
  while (true) {
    const runtimePath = path.join(ui, "node_modules/electron/path.txt");
    if (
      !fs.existsSync(runtimePath) ||
      !fs.existsSync(
        path.join(
          ui,
          "node_modules/electron/dist",
          fs.readFileSync(runtimePath, "utf8").trim(),
        ),
      )
    )
      throw Error("Electron runtime is missing. Run setup-electron first.");
    const electron = require(path.join(ui, "node_modules/electron"));
    const electronEnv = { ...process.env, BLUFLOW_PYTHON: python };
    delete electronEnv.ELECTRON_RUN_AS_NODE;
    const result = spawnSync(electron, [ui, ...args], {
      cwd: root,
      stdio: "inherit",
      // This is the interactive app. STARTUPINFO's hidden-window flag on Windows
      // can suppress BrowserWindow.show(), even after the renderer is ready.
      windowsHide: false,
      env: electronEnv,
    });
    if (result.error) throw result.error;
    if (result.status === null)
      throw Error(
        "Electron stopped unexpectedly" +
          (result.signal ? " (" + result.signal + ")" : ""),
      );
    if (![75, 76].includes(result.status)) {
      process.exit(result.status || 0);
    }
    if (++restarts > 2)
      throw Error("Repeated restart requests. Run setup-electron manually.");
    if (result.status === 75)
      run(python, ["-m", "trainer.tools.update_dependencies", "--root", root]);
    npm(["ci"]);
    installRuntime();
    npm(["run", "build"]);
    if (!args.includes("--no-update-check")) args.push("--no-update-check");
  }
} catch (error) {
  console.error("\nBluFlow Electron: " + error.message);
  process.exit(1);
}
