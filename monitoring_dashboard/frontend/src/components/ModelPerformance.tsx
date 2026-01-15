import React, { useEffect, useState } from 'react';

const ModelPerformance: React.FC = () => {
    const [accuracy, setAccuracy] = useState<any[]>([]);

    useEffect(() => {
        const interval = setInterval(() => {
            fetch('/api/metrics?name=model_accuracy&limit=10')
                .then(res => res.json())
                .then(data => setAccuracy(data))
                .catch(err => console.error(err));
        }, 2000);
        return () => clearInterval(interval);
    }, []);

    // Mock Reliability Diagram data
    const reliability = [
        { bin: 0.1, acc: 0.12 },
        { bin: 0.2, acc: 0.21 },
        { bin: 0.3, acc: 0.28 },
        { bin: 0.8, acc: 0.79 },
        { bin: 0.9, acc: 0.88 },
    ];

    return (
        <div>
            <h2>ML Model Verification</h2>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
                <div style={{ border: '1px solid #ddd', padding: 20 }}>
                    <h3>DPO Model Accuracy (Real-time)</h3>
                    {accuracy.map((m, i) => (
                         <div key={i}>{new Date(m.timestamp).toLocaleTimeString()}: {(m.value * 100).toFixed(1)}%</div>
                     ))}
                </div>
                 <div style={{ border: '1px solid #ddd', padding: 20 }}>
                    <h3>Calibration (Reliability Diagram)</h3>
                    <div style={{ height: 200, position: 'relative', borderLeft: '1px solid black', borderBottom: '1px solid black' }}>
                        {/* Simple CSS bar chart */}
                        {reliability.map((r, i) => (
                            <div key={i} style={{
                                position: 'absolute',
                                left: `${r.bin * 100}%`,
                                bottom: 0,
                                height: `${r.acc * 100}%`,
                                width: '10%',
                                background: 'blue',
                                opacity: 0.6
                            }}></div>
                        ))}
                         {/* Perfect calibration line */}
                         <div style={{
                             position: 'absolute',
                             left: 0, bottom: 0, width: '100%', height: '100%',
                             borderTop: '1px dashed red',
                             transform: 'rotate(-45deg)',
                             transformOrigin: 'bottom left'
                         }}></div>
                    </div>
                    <p>Bars should align with diagonal (red)</p>
                </div>
            </div>
        </div>
    );
};

export default ModelPerformance;
