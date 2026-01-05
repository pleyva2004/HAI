import { Question } from '@/types';
import { useState } from 'react';

interface QuestionCardProps {
    question: Question;
    className?: string;
}

export default function QuestionCard({ question, className = '' }: QuestionCardProps) {
    const [showExplanation, setShowExplanation] = useState(false);
    const [copied, setCopied] = useState(false);

    const handleCopy = () => {
        let text = `Question: ${question.text}\n`;
        if (question.equation) text += `Equation: ${question.equation}\n`;
        if (question.table) {
            text += '\nTable:\n';
            if (question.table.headers.length > 0) {
                text += question.table.headers.join('\t') + '\n';
            }
            question.table.rows.forEach(row => {
                text += row.join('\t') + '\n';
            });
        }
        text += '\nChoices:\n';
        Object.entries(question.answer_choices).forEach(([key, value]) => {
            text += `${key}) ${value}\n`;
        });
        if (question.correct_answer) {
            text += `\nCorrect Answer: ${question.correct_answer}`;
        }

        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className={`glass-panel my-6 transition-all duration-300 rounded-2xl ${className}`}>
            {/* Header / Actions - Clean Minimalist */}
            <div className="px-6 py-4 border-b border-black/5 dark:border-white/10 flex justify-between items-center">
                <span className="text-[11px] font-semibold text-blue-500 dark:text-blue-400 uppercase tracking-widest">
                    Generated Question
                </span>
                <button
                    onClick={handleCopy}
                    className="text-[11px] font-medium text-gray-400 dark:text-gray-500 hover:text-blue-500 dark:hover:text-blue-400 flex items-center gap-1.5 transition-colors px-2 py-1 rounded-md hover:bg-blue-50/50 dark:hover:bg-blue-500/10"
                >
                    {copied ? (
                        <>
                            <span className="text-green-500 dark:text-green-400">Copied</span>
                        </>
                    ) : (
                        'Copy text'
                    )}
                </button>
            </div>

            <div className="p-6 sm:p-8">
                {/* Question Text */}
                <p className="text-[17px] leading-relaxed text-[#1D1D1F] dark:text-[#E0E0E0] mb-6 font-normal">
                    {question.text}
                </p>

                {/* Equation (if present) */}
                {question.equation && (
                    <div className="my-6 p-4 bg-gray-50/80 dark:bg-white/5 rounded-xl text-center font-mono text-[15px] border border-black/5 dark:border-white/10 shadow-sm text-gray-800 dark:text-gray-200">
                        {question.equation}
                    </div>
                )}

                {/* Table (if present) */}
                {question.table && question.table.headers && question.table.headers.length > 0 && (
                    <div className="my-6 overflow-x-auto">
                        <div className="inline-block min-w-full rounded-xl border border-black/5 dark:border-white/10 shadow-sm bg-white/40 dark:bg-white/5">
                            <table className="min-w-full divide-y divide-black/5 dark:divide-white/10">
                                <thead>
                                    <tr className="bg-gray-50/80 dark:bg-white/10">
                                        {question.table.headers.map((header, index) => (
                                            <th
                                                key={index}
                                                className="px-4 py-3 text-left text-[13px] font-semibold text-gray-900 dark:text-gray-100 uppercase tracking-wider"
                                            >
                                                {header}
                                            </th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-black/5 dark:divide-white/10">
                                    {question.table.rows.map((row, rowIndex) => (
                                        <tr
                                            key={rowIndex}
                                            className="bg-white/40 dark:bg-white/5 hover:bg-white/60 dark:hover:bg-white/10 transition-colors"
                                        >
                                            {row.map((cell, cellIndex) => (
                                                <td
                                                    key={cellIndex}
                                                    className="px-4 py-3 text-[14px] text-gray-800 dark:text-gray-200"
                                                >
                                                    {cell}
                                                </td>
                                            ))}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}

                {/* Visual Description (if present) */}
                {question.visual && (
                    <div className="my-4 p-4 bg-blue-50/40 dark:bg-blue-500/10 text-blue-900/80 dark:text-blue-200 rounded-xl text-sm border border-blue-100/50 dark:border-blue-500/20 backdrop-blur-sm">
                        <span className="font-semibold block mb-1 text-blue-600 dark:text-blue-400">Visual Context</span>
                        {question.visual}
                    </div>
                )}

                {/* Answer Choices */}
                <div className="space-y-3 mt-8">
                    {Object.entries(question.answer_choices).map(([key, value]) => (
                        <div
                            key={key}
                            className={`group flex items-start gap-4 p-3.5 rounded-xl border transition-all duration-200 cursor-default ${showExplanation && question.correct_answer === key
                                ? 'bg-green-50/80 dark:bg-green-500/20 border-green-200 dark:border-green-500/30 shadow-sm'
                                : 'bg-white/40 dark:bg-white/5 border-black/5 dark:border-white/10 hover:bg-white/60 dark:hover:bg-white/10 hover:border-black/10 dark:hover:border-white/20'
                                }`}
                        >
                            <span className={`
                                flex-shrink-0 w-7 h-7 flex items-center justify-center rounded-full text-[13px] font-semibold transition-colors
                                ${showExplanation && question.correct_answer === key
                                    ? 'bg-green-500 text-white shadow-md shadow-green-200 dark:shadow-green-500/30'
                                    : 'bg-gray-100/80 dark:bg-white/10 text-gray-500 dark:text-gray-400 group-hover:bg-blue-500 dark:group-hover:bg-blue-500 group-hover:text-white group-hover:shadow-md group-hover:shadow-blue-200 dark:group-hover:shadow-blue-500/30'}
                            `}>
                                {key}
                            </span>
                            <span className={`pt-0.5 text-[15px] ${showExplanation && question.correct_answer === key
                                ? 'text-green-900 dark:text-green-200 font-medium'
                                : 'text-gray-700 dark:text-gray-300'
                                }`}>
                                {value}
                            </span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Explanation Toggle */}
            {question.explanation && (
                <div className="border-t border-black/5 dark:border-white/10 bg-gray-50/30 dark:bg-white/5">
                    <button
                        onClick={() => setShowExplanation(!showExplanation)}
                        className="w-full px-6 py-4 text-left text-[13px] font-medium text-gray-500 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 hover:bg-black/[0.02] dark:hover:bg-white/10 flex justify-between items-center transition-colors"
                    >
                        <span>{showExplanation ? 'Hide Answer & Explanation' : 'Reveal Answer & Explanation'}</span>
                        <span className="text-gray-400 dark:text-gray-500 bg-black/5 dark:bg-white/10 w-5 h-5 flex items-center justify-center rounded-full text-xs">
                            {showExplanation ? '−' : '+'}
                        </span>
                    </button>

                    {showExplanation && (
                        <div className="px-6 pb-8 pt-2 animate-in slide-in-from-top-2 fade-in duration-300">
                            <div className="mb-4 flex items-center gap-2">
                                <div className="w-1.5 h-1.5 rounded-full bg-green-500" />
                                <span className="font-semibold text-green-700 dark:text-green-400 text-sm">Correct: {question.correct_answer}</span>
                            </div>
                            <div className="prose prose-sm prose-blue max-w-none text-gray-600 dark:text-gray-300 leading-relaxed">
                                {question.explanation}
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
