import { useState } from 'react';

interface GenerationSettingsProps {
    isOpen: boolean;
    onClose: () => void;
    settings: {
        section: 'Math' | 'Reading and Writing';
        difficulty: 'Easy' | 'Medium' | 'Hard';
    };
    onSettingsChange: (settings: {
        section: 'Math' | 'Reading and Writing';
        difficulty: 'Easy' | 'Medium' | 'Hard';
    }) => void;
}

export default function GenerationSettings({ isOpen, onClose, settings, onSettingsChange }: GenerationSettingsProps) {
    if (!isOpen) return null;

    const sections: ('Math' | 'Reading and Writing')[] = ['Math', 'Reading and Writing'];
    const difficulties: ('Easy' | 'Medium' | 'Hard')[] = ['Easy', 'Medium', 'Hard'];

    return (
        <div className="absolute bottom-full mb-4 left-0 w-full md:w-[320px] z-50 animate-in fade-in slide-in-from-bottom-4 duration-200">
            <div className="bg-[#1c1c1e]/95 dark:bg-[#1c1c1e]/95 backdrop-blur-xl border border-white/10 rounded-2xl p-4 shadow-2xl ring-1 ring-black/5">
                <div className="flex justify-between items-center mb-4">
                    <h3 className="text-sm font-semibold text-white/90">Generation Settings</h3>
                    <button
                        onClick={onClose}
                        className="p-1 rounded-full hover:bg-white/10 text-gray-400 transition-colors"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                    </button>
                </div>

                <div className="space-y-4">
                    {/* Section Selector */}
                    <div className="space-y-2">
                        <label className="text-xs font-medium text-gray-400 uppercase tracking-wider">Section</label>
                        <div className="grid grid-cols-2 gap-2">
                            {sections.map((s) => (
                                <button
                                    key={s}
                                    onClick={() => onSettingsChange({ ...settings, section: s })}
                                    className={`
                                        px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200
                                        ${settings.section === s
                                            ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20'
                                            : 'bg-white/5 text-gray-400 hover:bg-white/10 hover:text-gray-200'}
                                    `}
                                >
                                    {s === 'Reading and Writing' ? 'Reading' : s}
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Difficulty Selector */}
                    <div className="space-y-2">
                        <label className="text-xs font-medium text-gray-400 uppercase tracking-wider">Difficulty</label>
                        <div className="grid grid-cols-3 gap-2">
                            {difficulties.map((d) => (
                                <button
                                    key={d}
                                    onClick={() => onSettingsChange({ ...settings, difficulty: d })}
                                    className={`
                                        px-3 py-2 rounded-lg text-sm font-medium transition-all duration-200
                                        ${settings.difficulty === d
                                            ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20'
                                            : 'bg-white/5 text-gray-400 hover:bg-white/10 hover:text-gray-200'}
                                    `}
                                >
                                    {d}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
