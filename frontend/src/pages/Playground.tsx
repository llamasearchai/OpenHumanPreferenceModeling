
import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { api } from '@/lib/api-client';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell
} from 'recharts';
import { Zap, Activity, Target, Download } from 'lucide-react';
import { ExportButton } from '@/components/widgets/ExportButton';

interface PredictionResult {
  probabilities: number[];
  action_index: number;
  confidence: number;
}

export default function PlaygroundPage() {
  const [vector, setVector] = useState<number[]>([0.5, 0.5, 0.5, 0.5, 0.5]);

  const predictMutation = useMutation({
    mutationFn: async (vec: number[]) => {
      const res = await api.post<PredictionResult>('/api/predict', { state_vector: vec });
      if (!res.success) throw new Error(res.error?.detail || 'Prediction failed');
      return res.data;
    },
  });

  const handleSliderChange = (index: number, value: number[]) => {
    const newVector = [...vector];
    newVector[index] = value[0] ?? 0;
    setVector(newVector);
  };

  const chartData = predictMutation.data?.probabilities.map((prob: number, idx: number) => ({
    name: `Action ${String.fromCharCode(65 + idx)}`,
    probability: prob,
    isSelected: idx === predictMutation.data?.action_index,
  })) || [];

  return (
    <div className="container mx-auto p-6 space-y-8">
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold tracking-tight">Model Playground</h1>
        <p className="text-muted-foreground">
          Interactively test the policy model by adjusting the state vector.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Input Section */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-primary" />
              State Vector Input
            </CardTitle>
            <CardDescription>
              Adjust the 5-dimensional input latent state vector (0.0 - 1.0).
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {vector.map((val, idx) => (
              <div key={idx} className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="font-medium text-muted-foreground">Dimension {idx + 1}</span>
                  <span className="font-mono">{val.toFixed(2)}</span>
                </div>
                <Slider
                  value={[val]}
                  min={0}
                  max={1}
                  step={0.01}
                  onValueChange={(v) => handleSliderChange(idx, v)}
                />
              </div>
            ))}

            <Button 
              className="w-full mt-4" 
              size="lg"
              onClick={() => predictMutation.mutate(vector)}
              disabled={predictMutation.isPending}
            >
              {predictMutation.isPending ? (
                <>
                  <Zap className="mr-2 h-4 w-4 animate-spin" />
                  Running Inference...
                </>
              ) : (
                <>
                  <Zap className="mr-2 h-4 w-4" />
                  Run Prediction
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Output Section */}
        <Card className="flex flex-col">
           <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-5 w-5 text-primary" />
              Prediction Results
            </CardTitle>
            <CardDescription>
              Model confidence and action probability distribution.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex-1 min-h-[300px] flex flex-col">
            {!predictMutation.data ? (
              <div className="flex-1 flex items-center justify-center text-muted-foreground border-2 border-dashed rounded-lg m-4">
                Run a prediction to see results
              </div>
            ) : (
              <div className="space-y-6 animate-in fade-in duration-500">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-secondary/20 p-4 rounded-lg">
                    <div className="text-sm text-muted-foreground">Selected Action</div>
                    <div className="text-2xl font-bold text-primary">
                      Action {String.fromCharCode(65 + predictMutation.data.action_index)}
                    </div>
                  </div>
                  <div className="bg-secondary/20 p-4 rounded-lg">
                    <div className="text-sm text-muted-foreground">Model Confidence</div>
                    <div className="text-2xl font-bold text-primary">
                      {(predictMutation.data.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                <div className="h-[250px] w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="probability" radius={[4, 4, 0, 0]}>
                        {chartData.map((entry, index: number) => (
                          <Cell 
                            key={`cell-${index}`} 
                            fill={entry.isSelected ? 'hsl(var(--primary))' : 'hsl(var(--muted-foreground))'} 
                            opacity={entry.isSelected ? 1 : 0.5}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
