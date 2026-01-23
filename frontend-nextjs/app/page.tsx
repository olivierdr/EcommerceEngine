'use client';

import { useState, useEffect } from 'react';
import { ProductTester } from '@/components/ProductTester';
import { ResultsTable } from '@/components/ResultsTable';
import { StatsCards } from '@/components/StatsCards';
import { ConfidenceDistribution, CategoryAccuracy, LatencyChart, CorrectnessChart } from '@/components/Charts';
import { PredictionResult, Stats } from './types';

const DEFAULT_API_URL = 'http://localhost:8000';
const STORAGE_KEY_RESULTS = 'ecommerce-classification-results';
const STORAGE_KEY_STATS = 'ecommerce-classification-stats';
const STORAGE_KEY_API_URL = 'ecommerce-classification-api-url';

export default function Home() {
  // Load from localStorage on mount
  const [apiUrl, setApiUrl] = useState(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem(STORAGE_KEY_API_URL) || DEFAULT_API_URL;
    }
    return DEFAULT_API_URL;
  });
  
  const [results, setResults] = useState<PredictionResult[]>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem(STORAGE_KEY_RESULTS);
      if (saved) {
        try {
          return JSON.parse(saved);
        } catch (e) {
          return [];
        }
      }
    }
    return [];
  });
  
  const [stats, setStats] = useState<Stats>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem(STORAGE_KEY_STATS);
      if (saved) {
        try {
          return JSON.parse(saved);
        } catch (e) {
          return {
            total_tested: 0,
            accuracy: 0,
            avg_confidence: 0,
            avg_latency_ms: 0,
          };
        }
      }
    }
    return {
      total_tested: 0,
      accuracy: 0,
      avg_confidence: 0,
      avg_latency_ms: 0,
    };
  });

  // Save to localStorage whenever results or stats change
  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem(STORAGE_KEY_RESULTS, JSON.stringify(results));
    }
  }, [results]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem(STORAGE_KEY_STATS, JSON.stringify(stats));
    }
  }, [stats]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem(STORAGE_KEY_API_URL, apiUrl);
    }
  }, [apiUrl]);

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="container mx-auto px-4 py-8">
        <header className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            Ecommerce Classification API
          </h1>
          <p className="text-gray-600">
            Interactive interface for testing e-commerce product classification
          </p>
        </header>

        {/* API URL Configuration */}
        <div className="mb-6 bg-white rounded-lg shadow-md p-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            API URL
          </label>
          <input
            type="text"
            value={apiUrl}
            onChange={(e) => setApiUrl(e.target.value)}
            placeholder="http://localhost:8000"
            className="w-full md:w-96 px-3 py-2 bg-gray-50 border-2 border-gray-400 text-gray-900 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 placeholder-gray-500"
          />
        </div>

        {/* Stats Cards */}
        {stats.total_tested > 0 && (
          <div className="mb-6">
            <div className="flex justify-between items-center mb-4">
              <div></div>
              <button
                onClick={() => {
                  setResults([]);
                  setStats({
                    total_tested: 0,
                    accuracy: 0,
                    avg_confidence: 0,
                    avg_latency_ms: 0,
                  });
                  if (typeof window !== 'undefined') {
                    localStorage.removeItem(STORAGE_KEY_RESULTS);
                    localStorage.removeItem(STORAGE_KEY_STATS);
                  }
                }}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 text-sm"
              >
                Clear Results
              </button>
            </div>
            <StatsCards stats={stats} />
          </div>
        )}

        {/* Product Tester */}
        <div className="mb-6">
          <ProductTester
            apiUrl={apiUrl}
            onResults={setResults}
            onStats={setStats}
          />
        </div>

        {/* Charts */}
        {results.length > 0 && (
          <div className="mb-6 grid grid-cols-1 lg:grid-cols-2 gap-6">
            <ConfidenceDistribution results={results} />
            <CorrectnessChart results={results} />
            <CategoryAccuracy results={results} />
            <LatencyChart results={results} />
          </div>
        )}

        {/* Results Table */}
        <div>
          <ResultsTable results={results} />
        </div>
      </div>
    </div>
  );
}

