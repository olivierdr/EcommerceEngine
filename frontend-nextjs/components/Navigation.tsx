'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export const Navigation = () => {
  const pathname = usePathname();

  return (
    <nav className="bg-white shadow-md mb-8">
      <div className="container mx-auto px-4">
        <div className="flex space-x-6">
          <Link
            href="/"
            className={`py-4 px-2 border-b-2 ${
              pathname === '/'
                ? 'border-blue-600 text-blue-600 font-semibold'
                : 'border-transparent text-gray-600 hover:text-gray-900 hover:border-gray-300'
            }`}
          >
            Home
          </Link>
          <Link
            href="/categories"
            className={`py-4 px-2 border-b-2 ${
              pathname === '/categories'
                ? 'border-blue-600 text-blue-600 font-semibold'
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


