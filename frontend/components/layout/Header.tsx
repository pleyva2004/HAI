'use client';

import Link from 'next/link';
import { BookOpen, Search } from 'lucide-react';

export function Header() {
  return (
    <header className="border-b bg-white sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4 flex justify-between items-center">
        <Link href="/" className="flex items-center gap-2">
          <BookOpen className="w-6 h-6 text-blue-600" />
          <span className="text-xl font-bold">SAT Question Generator</span>
        </Link>

        <nav className="flex gap-4">
          <Link
            href="/"
            className="px-4 py-2 rounded hover:bg-gray-100 transition-colors"
          >
            Generate
          </Link>
          <Link
            href="/search"
            className="px-4 py-2 rounded hover:bg-gray-100 transition-colors flex items-center gap-2"
          >
            <Search className="w-4 h-4" />
            Search Bank
          </Link>
        </nav>
      </div>
    </header>
  );
}
