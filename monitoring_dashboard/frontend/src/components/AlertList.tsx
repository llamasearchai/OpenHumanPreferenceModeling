import React, { useEffect, useState } from 'react';

const AlertList: React.FC = () => {
    const [alerts, setAlerts] = useState<any[]>([]);

    const fetchAlerts = () => {
        fetch('/api/alerts')
            .then(res => res.json())
            .then(data => setAlerts(data))
            .catch(err => console.error(err));
    };

    useEffect(() => {
        fetchAlerts();
        const interval = setInterval(fetchAlerts, 2000);
        return () => clearInterval(interval);
    }, []);

    const acknowledge = (id: string) => {
        fetch(`/api/alerts/${id}/ack`, { method: 'POST' })
            .then(() => fetchAlerts());
    }

    return (
        <div>
            <h2>Active Alerts</h2>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                    <tr style={{ textAlign: 'left', background: '#f0f0f0' }}>
                        <th style={{ padding: 10 }}>Severity</th>
                        <th>Rule</th>
                        <th>Message</th>
                        <th>Time</th>
                        <th>Status</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
                    {alerts.length === 0 && <tr><td colSpan={6} style={{padding: 20, textAlign: 'center'}}>No active alerts</td></tr>}
                    {alerts.map(a => (
                        <tr key={a.id} style={{ borderBottom: '1px solid #eee' }}>
                            <td style={{ padding: 10, color: a.severity === 'critical' ? 'red' : 'orange' }}>{a.severity.toUpperCase()}</td>
                            <td>{a.rule_name}</td>
                            <td>{a.message}</td>
                            <td>{new Date(a.timestamp).toLocaleTimeString()}</td>
                            <td>{a.status}</td>
                            <td>
                                {a.status === 'firing' && (
                                    <button onClick={() => acknowledge(a.id)}>Ack</button>
                                )}
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

export default AlertList;
