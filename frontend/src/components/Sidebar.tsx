import { useState } from 'react';

type Student = {
    id: string;
    name: string;
    grade: string;
    subject: string;
};

interface SidebarProps {
    isOpen: boolean;
    onToggle: () => void;
    students: Student[];
    selectedStudent: Student | null;
    onSelectStudent: (student: Student | null) => void;
}

export default function Sidebar({ isOpen, onToggle, students, selectedStudent, onSelectStudent }: SidebarProps) {
    return (
        <>
            {/* Mobile Overlay */}
            <div
                className={`fixed inset-0 bg-black/50 z-40 transition-opacity duration-300 md:hidden ${isOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'}`}
                onClick={onToggle}
            />

            {/* Sidebar Container */}
            <div
                className={`fixed top-0 left-0 h-full bg-[#FAFAFA] dark:bg-[#000000] border-r border-[#E5E5E5] dark:border-[#333] z-50 transition-all duration-300 ease-in-out
                    ${isOpen ? 'w-64 translate-x-0' : 'w-64 -translate-x-full md:translate-x-0 md:w-[72px]'} flex flex-col`}
            >
                {/* Header */}
                {/* Header */}
                <div className={`h-16 flex items-center px-4 border-b border-[#E5E5E5] dark:border-[#333] ${isOpen ? 'justify-end' : 'justify-center'}`}>
                    {isOpen ? (
                        <>
                            {/* Mobile Close Button */}
                            <button
                                onClick={onToggle}
                                className="md:hidden p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                            >
                                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <line x1="18" y1="6" x2="6" y2="18"></line>
                                    <line x1="6" y1="6" x2="18" y2="18"></line>
                                </svg>
                            </button>

                            {/* Desktop Collapse Button */}
                            <button
                                onClick={onToggle}
                                className="hidden md:block p-1.5 rounded-md hover:bg-gray-200 dark:hover:bg-gray-800 text-gray-500 transition-colors"
                                title="Collapse Sidebar"
                            >
                                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                                    <line x1="9" y1="3" x2="9" y2="21"></line>
                                </svg>
                            </button>
                        </>
                    ) : (
                        // Collapsed State - Show Expand Button (replacing Logo)
                        <button
                            onClick={onToggle}
                            className="p-2 rounded-md hover:bg-gray-200 dark:hover:bg-gray-800 text-gray-500 transition-colors"
                            title="Expand Sidebar"
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                                <line x1="9" y1="3" x2="9" y2="21"></line>
                            </svg>
                        </button>
                    )}
                </div>

                {/* New Session Button */}
                <div className="p-4">
                    <button
                        onClick={() => onSelectStudent(null)}
                        className={`w-full flex items-center gap-3 bg-white dark:bg-[#1C1C1E] border border-[#E5E5E5] dark:border-[#333] hover:border-indigo-500 dark:hover:border-indigo-500 rounded-xl shadow-sm hover:shadow-md transition-all group ${isOpen ? 'px-4 py-3' : 'p-2 justify-center'}`}
                        title="New Session"
                    >
                        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-indigo-50 dark:bg-indigo-900/30 flex items-center justify-center text-indigo-600 dark:text-indigo-400 group-hover:scale-110 transition-transform">
                            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <line x1="12" y1="5" x2="12" y2="19"></line>
                                <line x1="5" y1="12" x2="19" y2="12"></line>
                            </svg>
                        </div>
                        <div className={`text-left overflow-hidden transition-all duration-200 ${!isOpen ? 'w-0 opacity-0 hidden' : 'w-auto opacity-100 flex-1'}`}>
                            <div className="text-sm font-medium text-[#1D1D1F] dark:text-white whitespace-nowrap">New Session</div>
                            <div className="text-xs text-gray-500 dark:text-gray-400 whitespace-nowrap">General Practice</div>
                        </div>
                    </button>
                </div>

                {/* List Header */}
                <div className={`px-4 pb-2 ${!isOpen ? 'hidden' : 'block'}`}>
                    <h3 className="text-xs font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-wider whitespace-nowrap">
                        My Students
                    </h3>
                </div>

                {/* Students List */}
                <div className={`flex-1 overflow-y-auto px-2 space-y-1 ${!isOpen ? 'scrollbar-hide' : ''}`}>
                    {students.map((student) => (
                        <button
                            key={student.id}
                            onClick={() => onSelectStudent(student)}
                            className={`w-full flex items-center gap-3 px-3 py-3 rounded-lg transition-all group ${selectedStudent?.id === student.id
                                ? 'bg-indigo-50 dark:bg-indigo-900/20 text-indigo-900 dark:text-indigo-100'
                                : 'hover:bg-gray-100 dark:hover:bg-[#1C1C1E] text-gray-700 dark:text-gray-300'
                                } ${!isOpen ? 'justify-center px-0' : ''}`}
                            title={student.name}
                        >
                            <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium border ${selectedStudent?.id === student.id
                                ? 'bg-indigo-100 dark:bg-indigo-800 border-indigo-200 dark:border-indigo-700'
                                : 'bg-gray-100 dark:bg-gray-800 border-gray-200 dark:border-gray-700'
                                }`}>
                                {student.name.split(' ').map(n => n[0]).join('').slice(0, 2)}
                            </div>
                            <div className={`flex-1 min-w-0 text-left overflow-hidden transition-all duration-200 ${!isOpen ? 'w-0 opacity-0 hidden' : 'w-auto opacity-100'}`}>
                                <div className="text-sm font-medium truncate">
                                    {student.name}
                                </div>
                                <div className={`text-xs truncate ${selectedStudent?.id === student.id
                                    ? 'text-indigo-600 dark:text-indigo-300'
                                    : 'text-gray-500'
                                    }`}>
                                    {student.grade} • {student.subject}
                                </div>
                            </div>
                        </button>
                    ))}
                </div>

                {/* Footer User Profile */}
                <div className="p-4 border-t border-[#E5E5E5] dark:border-[#333]">
                    <div className={`flex items-center gap-3 ${isOpen ? 'px-2' : 'justify-center'}`}>
                        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-pink-500 to-orange-400 flex items-center justify-center text-white text-xs font-bold">
                            T
                        </div>
                        <div className={`flex-1 min-w-0 overflow-hidden transition-all duration-200 ${!isOpen ? 'w-0 opacity-0 hidden' : 'w-auto opacity-100'}`}>
                            <div className="text-sm font-medium text-[#1D1D1F] dark:text-white truncate">Tutor Account</div>
                            <div className="text-xs text-gray-500 dark:text-gray-400 truncate">Pro Plan</div>
                        </div>
                        {isOpen && (
                            <button className="text-gray-400 hover:text-[#1D1D1F] dark:hover:text-white transition-colors">
                                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <circle cx="12" cy="12" r="3"></circle>
                                    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                                </svg>
                            </button>
                        )}
                    </div>
                </div>
            </div>
        </>
    );
}

