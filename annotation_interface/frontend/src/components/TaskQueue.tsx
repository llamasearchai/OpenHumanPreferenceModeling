import React, { useEffect, useState } from 'react';
import { Task } from '../types';

interface Props {
    onTaskFetch: (task: Task) => void;
    annotatorId: string;
}

const TaskQueue: React.FC<Props> = ({ onTaskFetch, annotatorId }) => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        fetchNextTask();
    }, []);

    const fetchNextTask = async () => {
        setLoading(true);
        setError(null);
        try {
            const res = await fetch(`/api/tasks/next?annotator_id=${annotatorId}`);
            if (res.ok) {
                const task = await res.json();
                onTaskFetch(task);
            } else {
                setError("No tasks available");
            }
        } catch (err) {
            setError("Network error");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="task-queue">
            {loading && <div>Loading next task...</div>}
            {error && (
                <div className="error">
                    {error}
                    <button onClick={fetchNextTask}>Retry</button>
                </div>
            )}
        </div>
    );
};

export default TaskQueue;
