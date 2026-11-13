import { createContext, useContext } from 'react';
export type Actions = {
  search: string;
  setSearch: (value: string) => void;
  ask: (title: string, message: string, confirm?: string) => Promise<boolean>;
  openPreferences: () => void;
  openUtility: (kind: string) => void;
  reveal: (panel: string) => void;
  save: () => Promise<void>;
  start: (action?: string) => Promise<void>;
  addCurrent: () => Promise<void>;
};
export const ActionsContext = createContext<Actions>(null!);
export const useActions = () => useContext(ActionsContext);
