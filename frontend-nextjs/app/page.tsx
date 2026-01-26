'use client';

import { useState, useEffect } from 'react';
import { ProductTester } from '@/components/ProductTester';
import { ResultsTable } from '@/components/ResultsTable';
import { StatsCards } from '@/components/StatsCards';
import { ConfidenceDistribution, CategoryAccuracy, LatencyChart, CorrectnessChart } from '@/components/Charts';
import { PredictionResult, Stats } from './types';
import { DEFAULT_API_URL } from './config';

const DEFAULT_STATS: Stats = {
  total_tested: 0,
  accuracy: 0,
  avg_confidence: 0,
  avg_latency_ms: 0,
};

const STORAGE_KEY_RESULTS = 'ecommerce-classification-results';
const STORAGE_KEY_STATS = 'ecommerce-classification-stats';

export default function Home() {
  const [results, setResults] = useState<PredictionResult[]>([]);
  const [stats, setStats] = useState<Stats>(DEFAULT_STATS);
  const [categoryNames, setCategoryNames] = useState<Record<string, string>>({});

  // Charger les noms de catégories (id → nom) pour affichage lisible
  useEffect(() => {
    const base = (DEFAULT_API_URL || '').replace(/\/$/, '') || 'http://localhost:8000';
    fetch(`${base}/category-names`)
      .then((r) => (r.ok ? r.json() : {}))
      .then((data: Record<string, { name?: string }>) => {
        const map: Record<string, string> = {};
        if (data && typeof data === 'object') {
          for (const [id, info] of Object.entries(data)) {
            map[id] = info?.name ?? id;
          }
        }
        setCategoryNames(map);
      })
      .catch(() => {});
  }, []);

  // Charger depuis localStorage après hydratation (évite Hydration failed)
  useEffect(() => {
    const savedResults = localStorage.getItem(STORAGE_KEY_RESULTS);
    if (savedResults) {
      try {
        setResults(JSON.parse(savedResults));
      } catch {
        // ignore
      }
    }
    const savedStats = localStorage.getItem(STORAGE_KEY_STATS);
    if (savedStats) {
      try {
        setStats(JSON.parse(savedStats));
      } catch {
        // ignore
      }
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY_RESULTS, JSON.stringify(results));
  }, [results]);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY_STATS, JSON.stringify(stats));
  }, [stats]);

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

        {/* Stats Cards */}
        {stats.total_tested > 0 && (
          <div className="mb-6">
            <div className="flex justify-between items-center mb-4">
              <div></div>
              <button
                onClick={() => {
                  setResults([]);
                  setStats(DEFAULT_STATS);
                  localStorage.removeItem(STORAGE_KEY_RESULTS);
                  localStorage.removeItem(STORAGE_KEY_STATS);
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
            apiUrl={DEFAULT_API_URL}
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
          <ResultsTable results={results} categoryNames={categoryNames} />
        </div>
      </div>
    </div>
  );
}

