'use client';

import { useState } from 'react';
import { Product, PredictionResult } from '@/app/types';

/** Parse CSV with quoted fields (commas and newlines inside quotes). Returns [headers, ...rows]. */
function parseCSV(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = '';
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    const next = text[i + 1];
    if (inQuotes) {
      if (c === '"' && next === '"') {
        field += '"';
        i++;
      } else if (c === '"') {
        inQuotes = false;
      } else {
        field += c;
      }
    } else {
      if (c === '"') {
        inQuotes = true;
      } else if (c === ',') {
        row.push(field.trim());
        field = '';
      } else if (c === '\n' || c === '\r') {
        if (c === '\r' && next === '\n') i++;
        row.push(field.trim());
        field = '';
        if (row.some((cell) => cell.length > 0)) rows.push(row);
        row = [];
      } else {
        field += c;
      }
    }
  }
  if (field || row.length > 0) {
    row.push(field.trim());
    if (row.some((cell) => cell.length > 0)) rows.push(row);
  }
  return rows;
}

interface ProductTesterProps {
  apiUrl: string;
  onResults: (results: PredictionResult[]) => void;
  onStats: (stats: { total_tested: number; accuracy: number; avg_confidence: number; avg_latency_ms: number }) => void;
}

type LoadStatus = null | 'loading' | { count: number } | 'empty' | { error: string };

export const ProductTester = ({ apiUrl, onResults, onStats }: ProductTesterProps) => {
  const baseUrl = (apiUrl ?? '').replace(/\/$/, '') || 'http://localhost:8000';
  const [loading, setLoading] = useState(false);
  const [loadStatus, setLoadStatus] = useState<LoadStatus>(null);
  const [testProducts, setTestProducts] = useState<Product[]>([]);
  const [manualTitle, setManualTitle] = useState('');
  const [manualDescription, setManualDescription] = useState('');

  const loadTestSet = async () => {
    const url = `${baseUrl}/testset`;
    setLoadStatus('loading');
    setLoading(true);
    try {
      const response = await fetch(url);
      if (!response.ok) {
        const body = (await response.text()).slice(0, 200);
        throw new Error(`Testset: ${response.status} ${body || ''}`);
      }
      const text = await response.text();
      const parsed = parseCSV(text);
      if (parsed.length < 2) {
        setTestProducts([]);
        setLoadStatus('empty');
        return;
      }
      const headers = parsed[0].map((h) => h.trim().toLowerCase());
      const products: Product[] = [];
      for (let i = 1; i < parsed.length && products.length < 100; i++) {
        const values = parsed[i];
        const row: Partial<Product> = {};
        headers.forEach((key, idx) => {
          const v = values[idx]?.trim() ?? '';
          if (key === 'product_id') row.product_id = v;
          else if (key === 'title') row.title = v;
          else if (key === 'description') row.description = v;
          else if (key === 'category_id') row.category_id = v;
          else if (key === 'category_name') row.category_name = v;
          else if (key === 'category_path') row.category_path = v;
        });
        if (row.title) products.push(row as Product);
      }
      setTestProducts(products);
      setLoadStatus(products.length > 0 ? { count: products.length } : 'empty');
    } catch (error) {
      const isNetwork =
        error instanceof TypeError || (error instanceof Error && /fetch|network/i.test(error.message));
      const msg = isNetwork
        ? `Impossible de joindre l'API à ${url}. Vérifiez que l'API tourne (ex. ./start_local.sh).`
        : (error instanceof Error ? error.message : 'Échec chargement testset');
      console.error('Error loading testset:', error);
      setLoadStatus({ error: msg });
      alert(msg);
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
        const response = await fetch(`${baseUrl}/classify`, {
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

        const isCorrect = Boolean(product.category_id && data.category_id === product.category_id);
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
      const response = await fetch(`${baseUrl}/classify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: manualTitle,
          description: manualDescription || '',
        }),
      });

      if (!response.ok) {
        let detail = '';
        try {
          const b = await response.json() as { detail?: string };
          detail = b.detail ?? '';
        } catch {
          detail = (await response.text()).slice(0, 100);
        }
        throw new Error(`HTTP ${response.status}${detail ? `: ${detail}` : ''}`);
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
      const msg = error instanceof Error ? error.message : 'Failed to classify product';
      alert(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 space-y-6">
      <h2 className="text-2xl font-bold text-gray-800">Test Products</h2>

      {/* Load Testset */}
      <div className="space-y-2">
        <div className="flex flex-wrap gap-4 items-center">
          <button
            type="button"
            onClick={loadTestSet}
            disabled={loading}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Loading...' : 'Load Testset'}
          </button>
          <button
            type="button"
            onClick={() => testRandomProducts(10)}
            disabled={loading || testProducts.length === 0}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Testing...' : 'Test 10 Random Products'}
          </button>
          {loadStatus === 'loading' && (
            <span className="text-sm text-blue-600" role="status">Loading testset...</span>
          )}
          {loadStatus && loadStatus !== 'loading' && (
            <span
              className={`text-sm ${
                typeof loadStatus === 'object' && 'error' in loadStatus
                  ? 'text-red-600'
                  : loadStatus === 'empty'
                    ? 'text-amber-600'
                    : 'text-green-700'
              }`}
            >
              {loadStatus === 'empty'
                ? 'No products in testset'
                : typeof loadStatus === 'object' && 'count' in loadStatus
                  ? `${loadStatus.count} products loaded`
                  : typeof loadStatus === 'object' && 'error' in loadStatus
                    ? loadStatus.error
                    : null}
            </span>
          )}
        </div>
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

