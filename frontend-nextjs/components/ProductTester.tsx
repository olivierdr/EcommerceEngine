'use client';

import { useState } from 'react';
import { Product, PredictionResult } from '@/app/types';

interface ProductTesterProps {
  apiUrl: string;
  onResults: (results: PredictionResult[]) => void;
  onStats: (stats: { total_tested: number; accuracy: number; avg_confidence: number; avg_latency_ms: number }) => void;
}

export const ProductTester = ({ apiUrl, onResults, onStats }: ProductTesterProps) => {
  const [loading, setLoading] = useState(false);
  const [testProducts, setTestProducts] = useState<Product[]>([]);
  const [manualTitle, setManualTitle] = useState('');
  const [manualDescription, setManualDescription] = useState('');

  const loadTestSet = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${apiUrl}/testset`);
      if (!response.ok) throw new Error('Failed to load testset');
      
      const text = await response.text();
      const lines = text.split('\n').filter(line => line.trim());
      const headers = lines[0].split(',');
      
      const products: Product[] = [];
      for (let i = 1; i < lines.length && products.length < 100; i++) {
        const values = lines[i].split(',');
        if (values.length >= headers.length) {
          const product: Product = {};
          headers.forEach((header, idx) => {
            const key = header.trim().toLowerCase();
            if (key === 'product_id') product.product_id = values[idx]?.trim();
            if (key === 'title') product.title = values[idx]?.trim() || '';
            if (key === 'description') product.description = values[idx]?.trim();
            if (key === 'category_id') product.category_id = values[idx]?.trim();
            if (key === 'category_name') product.category_name = values[idx]?.trim();
            if (key === 'category_path') product.category_path = values[idx]?.trim();
          });
          if (product.title) {
            products.push(product);
          }
        }
      }
      
      setTestProducts(products);
    } catch (error) {
      console.error('Error loading testset:', error);
      alert('Failed to load testset');
    } finally {
      setLoading(false);
    }
  };

  const testRandomProducts = async (count: number = 10) => {
    if (testProducts.length === 0) {
      alert('Please load testset first');
      return;
    }

    setLoading(true);
    const selected = testProducts
      .sort(() => Math.random() - 0.5)
      .slice(0, count);

    const results: PredictionResult[] = [];
    let totalLatency = 0;
    let correctCount = 0;

    for (const product of selected) {
      try {
        const startTime = performance.now();
        const response = await fetch(`${apiUrl}/classify`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            title: product.title,
            description: product.description || '',
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        const latency = performance.now() - startTime;
        totalLatency += latency;

        const isCorrect = product.category_id && data.category_id === product.category_id;
        if (isCorrect) correctCount++;

        results.push({
          ...product,
          predicted_category_id: data.category_id,
          predicted_category_name: data.category_name,
          predicted_category_path: data.category_path,
          confidence: data.confidence,
          is_correct: isCorrect,
          latency_ms: Math.round(latency),
        });
      } catch (error) {
        console.error('Error classifying product:', error);
        results.push({
          ...product,
          predicted_category_id: 'ERROR',
          predicted_category_name: 'Error',
          predicted_category_path: 'N/A',
          confidence: 0,
          is_correct: false,
          latency_ms: 0,
        });
      }
    }

    setLoading(false);
    onResults(results);

    const stats = {
      total_tested: results.length,
      accuracy: results.length > 0 ? correctCount / results.length : 0,
      avg_confidence: results.length > 0
        ? results.reduce((sum, r) => sum + r.confidence, 0) / results.length
        : 0,
      avg_latency_ms: results.length > 0 ? totalLatency / results.length : 0,
    };
    onStats(stats);
  };

  const testManualProduct = async () => {
    if (!manualTitle.trim()) {
      alert('Please enter a product title');
      return;
    }

    setLoading(true);
    try {
      const startTime = performance.now();
      const response = await fetch(`${apiUrl}/classify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: manualTitle,
          description: manualDescription || '',
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      const latency = performance.now() - startTime;

      const result: PredictionResult = {
        title: manualTitle,
        description: manualDescription,
        predicted_category_id: data.category_id,
        predicted_category_name: data.category_name,
        predicted_category_path: data.category_path,
        confidence: data.confidence,
        latency_ms: Math.round(latency),
      };

      onResults([result]);
      onStats({
        total_tested: 1,
        accuracy: 0,
        avg_confidence: data.confidence,
        avg_latency_ms: latency,
      });

      setManualTitle('');
      setManualDescription('');
    } catch (error) {
      console.error('Error classifying product:', error);
      alert('Failed to classify product');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Test Products</h2>

      {/* Load Testset */}
      <div className="flex gap-4 items-end">
        <button
          onClick={loadTestSet}
          disabled={loading}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Loading...' : 'Load Testset'}
        </button>
        <button
          onClick={() => testRandomProducts(10)}
          disabled={loading || testProducts.length === 0}
          className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? 'Testing...' : 'Test 10 Random Products'}
        </button>
        <span className="text-sm text-gray-600">
          {testProducts.length > 0 && `${testProducts.length} products loaded`}
        </span>
      </div>

      {/* Manual Test */}
      <div className="border-t pt-6 space-y-4">
        <h3 className="text-lg font-semibold text-gray-700">Manual Test</h3>
        <div className="space-y-3">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Product Title *
            </label>
            <input
              type="text"
              value={manualTitle}
              onChange={(e) => setManualTitle(e.target.value)}
              placeholder="Enter product title..."
              className="w-full px-3 py-2 bg-gray-50 border-2 border-gray-400 text-gray-900 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 placeholder-gray-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Description (optional)
            </label>
            <textarea
              value={manualDescription}
              onChange={(e) => setManualDescription(e.target.value)}
              placeholder="Enter product description..."
              rows={3}
              className="w-full px-3 py-2 bg-gray-50 border-2 border-gray-400 text-gray-900 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 placeholder-gray-500"
            />
          </div>
          <button
            onClick={testManualProduct}
            disabled={loading || !manualTitle.trim()}
            className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Classifying...' : 'Classify Product'}
          </button>
        </div>
      </div>
    </div>
  );
};

