import { Question } from '@/types';
import { useState } from 'react';
import { BlockMath } from 'react-katex';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import remarkBreaks from 'remark-breaks';
import rehypeKatex from 'rehype-katex';

interface QuestionCardProps {
    question: Question;
    attemptCount?: number;
    iterationCount?: number;
    className?: string;
}

export default function QuestionCard({ question, attemptCount, iterationCount, className = '' }: QuestionCardProps) {
    const [showExplanation, setShowExplanation] = useState(false);
    const [copied, setCopied] = useState(false);
    const [selectedAnswer, setSelectedAnswer] = useState<string | null>(null);
    const [isCorrect, setIsCorrect] = useState<boolean | null>(null);

    const handleAnswerSelect = (key: string) => {
        // If already correctly answered, do nothing
        if (isCorrect) return;

        const correct = key === question.correct_answer;
        setSelectedAnswer(key);
        setIsCorrect(correct);

        if (correct) {
            // Auto reveal explanation after success
            setTimeout(() => {
                setShowExplanation(true);
            }, 1000);
        }
    };

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
            if (question.explanation) {
                text += `\nExplanation: ${question.explanation}`;
            }
        }

        navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <>
            <div className={`glass-panel my-6 transition-all duration-300 rounded-2xl ${className}`}>
                {/* Header / Actions - Clean Minimalist */}
                <div className="px-6 py-4 border-b border-black/5 dark:border-white/10 flex justify-between items-center">
                    <div className="flex items-center gap-4">
                        <span className="text-[11px] font-semibold text-orange-500 dark:text-orange-400 uppercase tracking-widest">
                            Generated Question
                        </span>
                    </div>

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
                    <div className="text-[17px] leading-relaxed text-[#1D1D1F] dark:text-[#E0E0E0] mb-6 font-normal [&>p]:m-0">
                        <ReactMarkdown
                            remarkPlugins={[remarkMath, remarkBreaks]}
                            rehypePlugins={[rehypeKatex]}
                        >
                            {question.text}
                        </ReactMarkdown>
                    </div>

                    {/* Equation (if present) */}
                    {question.equation && (
                        <div className="my-6">
                            <BlockMath>{question.equation}</BlockMath>
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
                        {Object.entries(question.answer_choices).map(([key, value]) => {
                            const isSelected = selectedAnswer === key;
                            const isAnswerCorrect = key === question.correct_answer;

                            // Determine styles based on state
                            let containerClasses = "bg-white/40 dark:bg-white/5 border-black/5 dark:border-white/10 hover:bg-white/60 dark:hover:bg-white/10 hover:border-black/10 dark:hover:border-white/20";
                            let circleClasses = "bg-gray-100/80 dark:bg-white/10 text-gray-500 dark:text-gray-400 group-hover:bg-[#D4A338] dark:group-hover:bg-[#D4A338] group-hover:text-white group-hover:shadow-md group-hover:shadow-[#D4A338]/40 dark:group-hover:shadow-[#D4A338]/30";
                            let textClasses = "text-gray-700 dark:text-gray-300";

                            if (isSelected) {
                                if (isAnswerCorrect) {
                                    // Correct State
                                    containerClasses = "bg-green-500/10 border-green-500 shadow-[0_0_15px_rgba(34,197,94,0.2)]";
                                    circleClasses = "bg-green-500 text-white shadow-lg shadow-green-500/40 transform scale-110";
                                    textClasses = "text-green-700 dark:text-green-400 font-medium";
                                } else {
                                    // Incorrect State
                                    containerClasses = "bg-red-500/10 border-red-500 animate-[shake_0.4s_ease-in-out]";
                                    circleClasses = "bg-red-500 text-white";
                                    textClasses = "text-red-700 dark:text-red-400";
                                }
                            } else if (isCorrect) {
                                // Dim other options when correct answer is found
                                containerClasses = "opacity-50 grayscale cursor-not-allowed border-transparent";
                            }

                            return (
                                <div
                                    key={key}
                                    onClick={() => handleAnswerSelect(key)}
                                    className={`group flex items-start gap-4 p-3.5 rounded-xl border transition-all duration-200 cursor-pointer ${containerClasses}`}
                                >
                                    <span className={`
                                        flex-shrink-0 w-7 h-7 flex items-center justify-center rounded-full text-[13px] font-semibold transition-all duration-300
                                        ${circleClasses}
                                    `}>
                                        {key}
                                    </span>
                                    <div className={`pt-0.5 text-[15px] transition-colors ${textClasses} [&>p]:m-0 break-words`}>
                                        <ReactMarkdown
                                            remarkPlugins={[remarkMath, remarkBreaks]}
                                            rehypePlugins={[rehypeKatex]}
                                        >
                                            {value}
                                        </ReactMarkdown>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            </div>

            {/* Footer with Metadata and Actions */}
            <div className="flex items-center gap-6 mt-2 pl-2">
                {attemptCount !== undefined && (
                    <div className="flex items-center gap-2 text-[10px] text-gray-400 font-medium uppercase tracking-wider">
                        <span className="w-1.5 h-1.5 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.4)]"></span>
                        {iterationCount && iterationCount > 0
                            ? `Revised • Iteration ${iterationCount} • Attempt ${attemptCount}`
                            : `Generated • Attempt ${attemptCount}`}
                    </div>
                )}

                {/* Answer Toggle Trigger (Moved to Footer) */}
                {(question.correct_answer || question.explanation) && (
                    <button
                        onClick={() => setShowExplanation(true)}
                        className="text-[10px] font-semibold text-orange-500 hover:text-gray-400 dark:hover:text-gray-500 transition-colors duration-300 uppercase tracking-widest cursor-pointer flex items-center gap-2"
                    >
                        Answer
                    </button>
                )}
            </div>

            {/* Answer Modal */}
            {showExplanation && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
                    {/* Backdrop */}
                    <div
                        className="absolute inset-0 bg-black/40 backdrop-blur-sm animate-in fade-in duration-200"
                        onClick={() => setShowExplanation(false)}
                    />

                    {/* Modal Content */}
                    <div className="relative w-full max-w-2xl bg-white dark:bg-[#1c1c1e] rounded-2xl shadow-2xl overflow-hidden animate-in zoom-in-95 duration-200 border border-white/10">
                        {/* Header */}
                        <div className="px-8 py-6 border-b border-black/5 dark:border-white/10 bg-gray-50/50 dark:bg-white/5 flex justify-between items-center">
                            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                                Solution
                            </h3>
                            <button
                                onClick={() => setShowExplanation(false)}
                                className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors"
                            >
                                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <line x1="18" y1="6" x2="6" y2="18"></line>
                                    <line x1="6" y1="6" x2="18" y2="18"></line>
                                </svg>
                            </button>
                        </div>

                        {/* Body */}
                        <div className="p-8">
                            <div className="flex flex-col gap-6">
                                {/* Correct Answer Section */}
                                <div className="flex items-center gap-3 p-4 bg-green-50 dark:bg-green-500/10 border border-green-100 dark:border-green-500/20 rounded-xl">
                                    <div className="w-10 h-10 rounded-full bg-green-500 flex items-center justify-center text-white font-bold text-lg shadow-lg shadow-green-500/30">
                                        {question.correct_answer}
                                    </div>
                                    <div>
                                        <div className="text-xs font-semibold text-green-600 dark:text-green-400 uppercase tracking-wider mb-0.5">
                                            Correct Answer
                                        </div>
                                        <div className="text-green-900 dark:text-green-100 font-medium">
                                            Option {question.correct_answer} is correct
                                        </div>
                                    </div>
                                </div>

                                {/* Explanation Section */}
                                <div>
                                    <h4 className="text-sm font-semibold text-gray-900 dark:text-white uppercase tracking-wider mb-3 flex items-center gap-2">
                                        <span className="w-1 h-4 bg-orange-500 rounded-full"></span>
                                        Explanation
                                    </h4>
                                    <div className="bg-gray-50/50 dark:bg-white/5 p-5 rounded-xl border border-black/5 dark:border-white/5 prose prose-sm prose-blue max-w-none dark:prose-invert text-gray-600 dark:text-gray-300 leading-relaxed">
                                        <ReactMarkdown
                                            remarkPlugins={[remarkMath, remarkBreaks]}
                                            rehypePlugins={[rehypeKatex]}
                                        >
                                            {question.explanation || "No explanation provided."}
                                        </ReactMarkdown>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
}
