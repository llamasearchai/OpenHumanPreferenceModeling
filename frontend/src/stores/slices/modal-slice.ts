import { StateCreator } from 'zustand';

export interface ModalState {
  id: string;
  isOpen: boolean;
  data?: unknown;
}

export interface ModalSlice {
  modals: Record<string, ModalState>;
  openModal: (id: string, data?: unknown) => void;
  closeModal: (id: string) => void;
  toggleModal: (id: string) => void;
  closeAllModals: () => void;
  getModalData: <T = unknown>(id: string) => T | undefined;
}

export const createModalSlice: StateCreator<ModalSlice> = (set, get) => ({
  modals: {},
  openModal: (id, data) =>
    set((state) => {
      const next: ModalState =
        data === undefined ? { id, isOpen: true } : { id, isOpen: true, data };
      return { modals: { ...state.modals, [id]: next } };
    }),
  closeModal: (id) =>
    set((state) => {
      const current = state.modals[id];
      if (!current) return {};
      return { modals: { ...state.modals, [id]: { ...current, isOpen: false } } };
    }),
  toggleModal: (id) =>
    set((state) => {
      const current = state.modals[id];
      if (!current) {
        return { modals: { ...state.modals, [id]: { id, isOpen: true } } };
      }
      return { modals: { ...state.modals, [id]: { ...current, isOpen: !current.isOpen } } };
    }),
  closeAllModals: () =>
    set((state) => {
      const next: Record<string, ModalState> = {};
      for (const [id, modal] of Object.entries(state.modals)) {
        next[id] = { ...modal, isOpen: false };
      }
      return { modals: next };
    }),
  getModalData: <T = unknown>(id: string) => {
    return get().modals[id]?.data as T | undefined;
  },
});
