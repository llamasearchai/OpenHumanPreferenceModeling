import React, { useEffect, useState } from 'react';

const SystemOverview: React.FC = () => {
    const [metrics, setMetrics] = useState<any[]>([]);

    useEffect(() => {
        // Poll metrics
        const interval = setInterval(() => {
            fetch('/api/metrics?name=encoder_latency_seconds&limit=5')
                .then(res => res.json())
                .then(data => setMetrics(data))
                .catch(err => console.error(err));
        }, 2000);
        return () => clearInterval(interval);
    }, []);

    // Also fetch error count
    const [errorCount, setErrorCount] = useState<any[]>([]);
     useEffect(() => {
        const interval = setInterval(() => {
            fetch('/api/metrics?name=error_count&limit=5')
                .then(res => res.json())
                .then(data => setErrorCount(data))
                .catch(err => console.error(err));
        }, 2000);
        return () => clearInterval(interval);
    }, []);

    return (
        <div>
            <h2>System Health</h2>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
                <div style={{ border: '1px solid #ddd', padding: 20 }}>
                     <h3>Encoder Latency (Last 5)</h3>
                     {metrics.map((m, i) => (
                         <div key={i}>{new Date(m.timestamp).toLocaleTimeString()}: {(m.value * 1000).toFixed(0)} ms</div>
                     ))}
                </div>
                <div style={{ border: '1px solid #ddd', padding: 20 }}>
                     <h3>API Error Rate</h3>
                      {errorCount.map((m, i) => (
                         <div key={i}>{new Date(m.timestamp).toLocaleTimeString()}: {m.value > 0 ? "ERROR" : "OK"}</div>
                     ))}
                </div>
            </div>
        </div>
    );
};

export default SystemOverview;
