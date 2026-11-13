const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('bluflow', {
  request: (method, params = {}) => ipcRenderer.invoke('bluflow:request', method, params),
  file: (options) => ipcRenderer.invoke('bluflow:file', options),
  preferences: (patch) => ipcRenderer.invoke('bluflow:presentation', patch),
  restart: (options) => ipcRenderer.invoke('bluflow:restart', options),
  snapshot: () => ipcRenderer.invoke('bluflow:snapshot'),
  ready: () => ipcRenderer.send('bluflow:ready'),
  onEvent: (callback) => {
    const listener = (_, event) => callback(event);
    ipcRenderer.on('bluflow:event', listener);
    return () => ipcRenderer.removeListener('bluflow:event', listener);
  },
});
