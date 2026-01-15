import React, { useState, useEffect } from 'react';
import SystemOverview from './components/SystemOverview';
import AlertList from './components/AlertList';
import ModelPerformance from './components/ModelPerformance';

const Dashboard: React.FC = () => {
    const [activeTab, setActiveTab] = useState("overview");

    return (
        <div style={{ padding: 20 }}>
            <h1>Evaluation & Monitoring Dashboard</h1>
            
            <div style={{ marginBottom: 20, borderBottom: '1px solid #ccc' }}>
                <button onClick={() => setActiveTab("overview")} style={btnStyle(activeTab === "overview")}>System Overview</button>
                <button onClick={() => setActiveTab("ml")} style={btnStyle(activeTab === "ml")}>ML Performance</button>
                <button onClick={() => setActiveTab("alerts")} style={btnStyle(activeTab === "alerts")}>Alerts</button>
            </div>

            {activeTab === "overview" && <SystemOverview />}
            {activeTab === "ml" && <ModelPerformance />}
            {activeTab === "alerts" && <AlertList />}
        </div>
    );
};

const btnStyle = (isActive: boolean) => ({
    padding: '10px 20px',
    background: isActive ? '#007bff' : 'transparent',
    color: isActive ? 'white' : 'black',
    border: 'none',
    cursor: 'pointer',
    marginRight: 10
});

export default Dashboard;
