'use client';

import { useState } from 'react';
import Header from '@/components/Header';
import ChatInterface from '@/components/ChatInterface';
import Sidebar from '@/components/Sidebar';

const MOCK_STUDENTS = [
  { id: '1', name: 'Alex Johnson', grade: '11th Grade', subject: 'Math' },
  { id: '2', name: 'Sarah Williams', grade: '12th Grade', subject: 'Reading' },
  { id: '3', name: 'Michael Chen', grade: '10th Grade', subject: 'Math' },
  { id: '4', name: 'Emma Davis', grade: '11th Grade', subject: 'Writing' },
];

export default function Home() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [selectedStudent, setSelectedStudent] = useState<{ id: string; name: string; grade: string; subject: string } | null>(null);

  return (
    <main className="flex h-screen bg-white dark:bg-black overflow-hidden relative">
      <Sidebar
        isOpen={isSidebarOpen}
        onToggle={() => setIsSidebarOpen(!isSidebarOpen)}
        students={MOCK_STUDENTS}
        selectedStudent={selectedStudent}
        onSelectStudent={setSelectedStudent}
      />

      <div className={`flex-1 flex flex-col h-full bg-white dark:bg-black transition-all duration-300 relative ${isSidebarOpen ? 'md:ml-64' : 'md:ml-[72px]'}`}>
        <Header
          onToggleSidebar={() => setIsSidebarOpen(true)}
          isSidebarOpen={isSidebarOpen}
        />
        <div className="flex-1 overflow-hidden relative">
          <ChatInterface selectedStudent={selectedStudent} />
        </div>
      </div>
    </main>
  );
}
