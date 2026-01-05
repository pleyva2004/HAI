'use client';

import Header from '@/components/Header';
import ChatInterface from '@/components/ChatInterface';

export default function Home() {
  return (
    <main className="min-h-screen relative">
      <Header />
      <ChatInterface />
    </main>
  );
}
