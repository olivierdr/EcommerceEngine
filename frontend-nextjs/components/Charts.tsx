'use client';

import { PredictionResult } from '@/app/types';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';

interface ChartsProps {
  results: PredictionResult[];
}

export const ConfidenceDistribution = ({ results }: ChartsProps) => {
  if (results.length === 0) return null;

  // Create bins for confidence distribution
  const bins = [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0];
  const distribution = bins.slice(0, -1).map((min, idx) => {
    const max = bins[idx + 1];
    const count = results.filter(r => r.confidence >= min && r.confidence < max).length;
    return {
      range: `${(min * 100).toFixed(0)}-${(max * 100).toFixed(0)}%`,
      count,
      min,
      max,
    };
  });

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Confidence Distribution</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={distribution}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="range" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="count" fill="#3b82f6">
            {distribution.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={
                  entry.min >= 0.7
                    ? '#10b981'
                    : entry.min >= 0.5
                    ? '#f59e0b'
                    : '#ef4444'
                }
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export const CategoryAccuracy = ({ results }: ChartsProps) => {
  if (results.length === 0) return null;

  // Calculate accuracy per category
  const categoryStats: { [key: string]: { correct: number; total: number; name: string } } = {};
  
  results.forEach(result => {
    if (result.category_id && result.predicted_category_id) {
      const catId = result.category_id;
      if (!categoryStats[catId]) {
        categoryStats[catId] = {
          correct: 0,
          total: 0,
          name: result.category_name || catId,
        };
      }
      categoryStats[catId].total++;
      if (result.is_correct) {
        categoryStats[catId].correct++;
      }
    }
  });

  const categoryData = Object.entries(categoryStats)
    .map(([id, stats]) => ({
      category: stats.name.length > 20 ? stats.name.substring(0, 20) + '...' : stats.name,
      accuracy: stats.total > 0 ? (stats.correct / stats.total) * 100 : 0,
      total: stats.total,
    }))
    .sort((a, b) => b.total - a.total)
    .slice(0, 10); // Top 10 categories

  if (categoryData.length === 0) return null;

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Accuracy by Category (Top 10)</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={categoryData} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" domain={[0, 100]} />
          <YAxis dataKey="category" type="category" width={150} />
          <Tooltip formatter={(value: number) => `${value.toFixed(1)}%`} />
          <Bar dataKey="accuracy" fill="#8b5cf6">
            {categoryData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={
                  entry.accuracy >= 80
                    ? '#10b981'
                    : entry.accuracy >= 60
                    ? '#f59e0b'
                    : '#ef4444'
                }
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export const LatencyChart = ({ results }: ChartsProps) => {
  if (results.length === 0) return null;

  const latencyData = results.map((result, index) => ({
    index: index + 1,
    latency: result.latency_ms,
  }));

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Latency per Request</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={latencyData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="index" label={{ value: 'Request #', position: 'insideBottom', offset: -5 }} />
          <YAxis label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft' }} />
          <Tooltip formatter={(value: number) => `${value.toFixed(0)} ms`} />
          <Bar dataKey="latency" fill="#06b6d4">
            {latencyData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={
                  entry.latency < 200
                    ? '#10b981'
                    : entry.latency < 500
                    ? '#f59e0b'
                    : '#ef4444'
                }
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export const CorrectnessChart = ({ results }: ChartsProps) => {
  if (results.length === 0) return null;

  const correctCount = results.filter(r => r.is_correct === true).length;
  const incorrectCount = results.filter(r => r.is_correct === false).length;
  const unknownCount = results.filter(r => r.is_correct === undefined).length;

  const data = [
    { name: 'Correct', value: correctCount, color: '#10b981' },
    { name: 'Incorrect', value: incorrectCount, color: '#ef4444' },
    { name: 'Unknown', value: unknownCount, color: '#9ca3af' },
  ].filter(item => item.value > 0);

  if (data.length === 0) return null;

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Prediction Correctness</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="value">
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

