import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.tsx';
import './index.css';
import { ThemeProvider } from './contexts/ThemeProvider.tsx';

// #region agent log
window.addEventListener('error', (event) => {
  fetch('http://127.0.0.1:7259/ingest/44e72182-20fc-4ac5-ace5-6d05735c6915',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.tsx:window-error',message:'Unhandled window error',data:{message:event.message,filename:event.filename,lineno:event.lineno,colno:event.colno},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
});

window.addEventListener('unhandledrejection', (event) => {
  fetch('http://127.0.0.1:7259/ingest/44e72182-20fc-4ac5-ace5-6d05735c6915',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.tsx:unhandledrejection',message:'Unhandled promise rejection',data:{reason:String(event.reason)},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
});
// #endregion

async function enableMocking() {
  if (import.meta.env.VITE_USE_MOCKS !== 'true') {
    return;
  }
  const { worker } = await import('./mocks/browser');
  await worker.start({ onUnhandledRequest: 'bypass' });
}

const rootEl = document.getElementById('root');
// #region agent log
fetch('http://127.0.0.1:7259/ingest/44e72182-20fc-4ac5-ace5-6d05735c6915',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.tsx:root-check',message:'Root element check',data:{rootExists:!!rootEl},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
// #endregion
if (!rootEl) {
  throw new Error('Root element not found');
}

const root = ReactDOM.createRoot(rootEl);
root.render(
  <React.StrictMode>
    <ThemeProvider>
      <App />
    </ThemeProvider>
  </React.StrictMode>,
);
// #region agent log
fetch('http://127.0.0.1:7259/ingest/44e72182-20fc-4ac5-ace5-6d05735c6915',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'main.tsx:render-called',message:'React render invoked',data:{},timestamp:Date.now(),sessionId:'debug-session',runId:'run1',hypothesisId:'A'})}).catch(()=>{});
// #endregion

enableMocking().catch((err) => {
  console.warn('MSW failed to start; continuing without request mocking.', err);
});
