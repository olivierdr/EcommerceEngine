'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export const Navigation = () => {
  const pathname = usePathname();

  return (
    <nav className="bg-white border-b border-gray-200 shadow-sm" aria-label="Main">
      <div className="container mx-auto px-4">
        <div className="flex gap-1" role="tablist">
          <Link
            href="/"
            role="tab"
            aria-selected={pathname === '/'}
            className={`py-4 px-4 border-b-2 font-medium transition-colors ${
              pathname === '/'
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-gray-600 hover:text-gray-900 hover:border-gray-300'
            }`}
          >
            Home
          </Link>
          <Link
            href="/categories"
            role="tab"
            aria-selected={pathname === '/categories'}
            className={`py-4 px-4 border-b-2 font-medium transition-colors ${
              pathname === '/categories'
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-gray-600 hover:text-gray-900 hover:border-gray-300'
            }`}
          >
            Categories
          </Link>
        </div>
      </div>
    </nav>
  );
};


