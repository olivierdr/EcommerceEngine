'use client';

import { Stats } from '@/app/types';

interface StatsCardsProps {
  stats: Stats;
}

export const StatsCards = ({ stats }: StatsCardsProps) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="text-sm font-medium text-gray-500">Products Tested</div>
        <div className="mt-2 text-3xl font-bold text-gray-900">{stats.total_tested}</div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="text-sm font-medium text-gray-500">Accuracy</div>
        <div className="mt-2 text-3xl font-bold text-gray-900">
          {stats.total_tested > 0 ? (stats.accuracy * 100).toFixed(1) : '0.0'}%
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="text-sm font-medium text-gray-500">Avg Confidence</div>
        <div className="mt-2 text-3xl font-bold text-gray-900">
          {(stats.avg_confidence * 100).toFixed(1)}%
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="text-sm font-medium text-gray-500">Avg Latency</div>
        <div className="mt-2 text-3xl font-bold text-gray-900">
          {stats.avg_latency_ms.toFixed(0)} ms
        </div>
      </div>
    </div>
  );
};

