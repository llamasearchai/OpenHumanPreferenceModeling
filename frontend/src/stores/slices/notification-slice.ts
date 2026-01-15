import { StateCreator } from 'zustand';

export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  title: string;
  message?: string | undefined;
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
  createdAt: number;
}

export interface NotificationSlice {
  notifications: Notification[];
  maxNotifications: number;
  addNotification: (notification: Omit<Notification, 'id' | 'createdAt'>) => string;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
}

export const createNotificationSlice: StateCreator<NotificationSlice> = (set) => ({
  notifications: [],
  maxNotifications: 5,
  addNotification: (notification) => {
    const id = `notification-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const createdAt = Date.now();

    set((state) => {
      const newNotification: Notification = { ...notification, id, createdAt };
      const next = [newNotification, ...state.notifications].slice(0, state.maxNotifications);
      return { notifications: next };
    });

    return id;
  },
  removeNotification: (id) =>
    set((state) => ({
      notifications: state.notifications.filter((n) => n.id !== id),
    })),
  clearNotifications: () => set({ notifications: [] }),
});
