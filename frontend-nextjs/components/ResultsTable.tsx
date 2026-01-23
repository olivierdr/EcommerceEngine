'use client';

import { PredictionResult } from '@/app/types';

interface ResultsTableProps {
  results: PredictionResult[];
}

export const ResultsTable = ({ results }: ResultsTableProps) => {
  if (results.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6 text-center text-gray-500">
        No results yet. Test some products to see predictions.
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Results</h2>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Title
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                True Category
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Predicted Category
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Confidence
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Correct
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Latency (ms)
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {results.map((result, idx) => (
              <tr key={idx} className={result.is_correct === false ? 'bg-red-50' : result.is_correct === true ? 'bg-green-50' : ''}>
                <td className="px-4 py-3 text-sm">
                  <div className="max-w-xs truncate font-medium text-gray-900" title={result.title || 'No title'}>
                    {result.title || <span className="text-gray-400 italic">No title</span>}
                  </div>
                </td>
                <td className="px-4 py-3 text-sm">
                  {result.category_name ? (
                    <span className="text-gray-900 font-medium">{result.category_name}</span>
                  ) : (
                    <span className="text-gray-400 italic">Manual test (no true category)</span>
                  )}
                </td>
                <td className="px-4 py-3 text-sm font-medium text-gray-900">
                  {result.predicted_category_name}
                </td>
                <td className="px-4 py-3 text-sm">
                  <div className="flex items-center">
                    <span className="mr-2">{(result.confidence * 100).toFixed(1)}%</span>
                    <div className="w-24 bg-gray-200 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          result.confidence > 0.7
                            ? 'bg-green-500'
                            : result.confidence > 0.5
                            ? 'bg-yellow-500'
                            : 'bg-red-500'
                        }`}
                        style={{ width: `${result.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                </td>
                <td className="px-4 py-3 text-sm">
                  {result.is_correct === undefined ? (
                    <span className="text-gray-400">-</span>
                  ) : result.is_correct ? (
                    <span className="text-green-600 font-semibold">✓</span>
                  ) : (
                    <span className="text-red-600 font-semibold">✗</span>
                  )}
                </td>
                <td className="px-4 py-3 text-sm text-gray-600">
                  {result.latency_ms.toFixed(0)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

