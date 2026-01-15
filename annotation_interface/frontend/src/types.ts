export interface Task {
  id: string;
  type: "pairwise" | "ranking" | "critique" | "likert";
  content: any;
  priority: number;
}

export interface Annotation {
  taskId: string;
  annotatorId: string;
  annotationType: string;
  responseData: any;
  timeSpentSeconds: number;
  confidence: number;
}
