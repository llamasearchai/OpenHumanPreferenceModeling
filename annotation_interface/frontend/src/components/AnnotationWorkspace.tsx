import React, { useState, useEffect } from 'react';
import { Task, Annotation } from '../types';
import TaskQueue from './TaskQueue';

const AnnotationWorkspace: React.FC = () => {
    const [currentTask, setCurrentTask] = useState<Task | null>(null);
    const [startTime, setStartTime] = useState<number>(0);
    const [annotatorId, setAnnotatorId] = useState<string>("user_123"); // Mock ID for now

    const handleTaskLoaded = (task: Task) => {
        setCurrentTask(task);
        setStartTime(Date.now());
    };

    const submitAnnotation = async (responseData: any, confidence: number) => {
        if (!currentTask) return;

        const timeSpent = (Date.now() - startTime) / 1000;
        const annotation: Annotation = {
            taskId: currentTask.id,
            annotatorId: annotatorId,
            annotationType: currentTask.type,
            responseData: responseData,
            timeSpentSeconds: timeSpent,
            confidence: confidence
        };

        try {
            const res = await fetch('/api/annotations', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(annotation)
            });
            
            if (res.ok) {
                console.log("Annotation submitted");
                setCurrentTask(null); // Clear to trigger next fetch
            } else {
                console.error("Failed to submit");
            }
        } catch (err) {
            console.error(err);
        }
    };

    return (
        <div className="workspace">
            <header>
                <h1>OpenHuman Annotation</h1>
                <div className="stats">Annotator: {annotatorId}</div>
            </header>
            
            <main>
                {!currentTask && (
                    <TaskQueue onTaskFetch={handleTaskLoaded} annotatorId={annotatorId} />
                )}
                
                {currentTask && currentTask.type === 'pairwise' && (
                    <div className="pairwise-task">
                        <h2>{currentTask.content.prompt}</h2>
                        <div className="comparison">
                            <div className="card">
                                <h3>Response A</h3>
                                <p>{currentTask.content.response_a}</p>
                                <button onClick={() => submitAnnotation({winner: 'A'}, 5)}>Select A</button>
                            </div>
                            <div className="card">
                                <h3>Response B</h3>
                                <p>{currentTask.content.response_b}</p>
                                <button onClick={() => submitAnnotation({winner: 'B'}, 5)}>Select B</button>
                            </div>
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
};

export default AnnotationWorkspace;
