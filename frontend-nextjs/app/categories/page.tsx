'use client';

import { useState, useEffect } from 'react';
import { CategoryData } from '../types';
import { DEFAULT_API_URL } from '../config';

export default function CategoriesPage() {
  const [categories, setCategories] = useState<CategoryData>({});
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    loadCategories();
  }, []);

  const loadCategories = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${DEFAULT_API_URL}/category-names`);
      if (!response.ok) throw new Error('Failed to load categories');
      
      const data = await response.json();
      setCategories(data);
    } catch (error) {
      console.error('Error loading categories:', error);
      alert('Failed to load categories');
    } finally {
      setLoading(false);
    }
  };

  const filteredCategories = Object.entries(categories).filter(([id, info]) => {
    if (!searchTerm) return true;
    const search = searchTerm.toLowerCase();
    return (
      info.name.toLowerCase().includes(search) ||
      id.toLowerCase().includes(search) ||
      info.example_titles.some(title => title.toLowerCase().includes(search))
    );
  });

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="container mx-auto px-4 py-8">
        <header className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Categories</h1>
          <p className="text-gray-600">
            Browse all product categories with examples
          </p>
        </header>

        {/* Search */}
        <div className="mb-6 bg-white rounded-lg shadow-md p-4">
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search categories by name, ID, or example..."
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Categories Grid */}
        {loading ? (
          <div className="text-center py-12 text-gray-500">Loading categories...</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredCategories.map(([categoryId, info]) => (
              <div
                key={categoryId}
                className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow"
              >
                <div className="mb-3">
                  <div className="text-xs font-mono text-gray-500 mb-1">{categoryId}</div>
                  <h3 className="text-lg font-semibold text-gray-900">{info.name}</h3>
                </div>
                
                <div className="mt-4">
                  <div className="text-sm font-medium text-gray-700 mb-2">
                    Examples ({info.example_titles.length}):
                  </div>
                  <ul className="space-y-1">
                    {info.example_titles.slice(0, 5).map((title, idx) => (
                      <li
                        key={idx}
                        className="text-sm text-gray-600 truncate"
                        title={title}
                      >
                        • {title}
                      </li>
                    ))}
                    {info.example_titles.length > 5 && (
                      <li className="text-xs text-gray-400">
                        +{info.example_titles.length - 5} more
                      </li>
                    )}
                  </ul>
                </div>
              </div>
            ))}
          </div>
        )}

        {!loading && filteredCategories.length === 0 && (
          <div className="text-center py-12 text-gray-500">
            No categories found matching your search.
          </div>
        )}

        {!loading && filteredCategories.length > 0 && (
          <div className="mt-6 text-center text-sm text-gray-500">
            Showing {filteredCategories.length} of {Object.keys(categories).length} categories
          </div>
        )}
      </div>
    </div>
  );
}

