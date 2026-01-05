import { Message } from '@/types';
import Image from 'next/image';
import QuestionCard from './QuestionCard';

interface MessageBubbleProps {
    message: Message;
}

export default function MessageBubble({ message }: MessageBubbleProps) {
    const isUser = message.role === 'user';

    return (
        <div className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'} mb-8 animate-in fade-in slide-in-from-bottom-2 duration-500`}>
            <div className={`flex flex-col max-w-[85%] sm:max-w-[75%] ${isUser ? 'items-end' : 'items-start'}`}>

                {/* User Image Attachment - Rendered OUTSIDE the colored bubble for a clean look */}
                {message.image && (
                    <div className="mb-2 rounded-2xl overflow-hidden border border-white/10 shadow-sm">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                            src={`data:image/jpeg;base64,${message.image}`}
                            alt="Uploaded context"
                            className="max-h-80 h-auto w-auto object-contain bg-black/5 dark:bg-white/5"
                        />
                    </div>
                )}

                {/* Message Content Bubble (Text Only) */}
                {(message.content || message.isLoading) && (
                    <div
                        className={`px-5 py-3.5 shadow-sm transition-all ${isUser
                            ? 'bg-[#007AFF] text-white shadow-blue-500/25 rounded-[20px] rounded-tr-sm'
                            : 'bg-white/80 dark:bg-[#1c1c1e] backdrop-blur-md border border-white/50 dark:border-white/10 text-[#1D1D1F] dark:text-[#E0E0E0] rounded-[20px] rounded-tl-sm shadow-[0_2px_8px_rgba(0,0,0,0.04)]'
                            }`}
                    >
                        {/* Loading State */}
                        {message.isLoading && (
                            <div className="flex space-x-1.5 py-1 px-1 items-center">
                                <div className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-[bounce_1s_infinite_-0.3s]"></div>
                                <div className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-[bounce_1s_infinite_-0.15s]"></div>
                                <div className="w-2 h-2 bg-gray-400 dark:bg-gray-500 rounded-full animate-[bounce_1s_infinite]"></div>
                            </div>
                        )}

                        {/* Text Content */}
                        {message.content && (
                            <div className={`whitespace-pre-wrap leading-relaxed text-[16px] font-normal`}>
                                {message.content}
                            </div>
                        )}
                    </div>
                )}

                {/* Generated Question Card */}
                {message.question && (
                    <div className="w-full mt-4 animate-in fade-in zoom-in-95 duration-500 delay-150">
                        <div className="transform transition-all hover:scale-[1.005] duration-300">
                            <QuestionCard question={message.question.question} />
                        </div>
                        <div className="flex items-center gap-2 text-[10px] text-gray-400 mt-2 pl-2 font-medium uppercase tracking-wider">
                            <span className="w-1.5 h-1.5 rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.4)]"></span>
                            Generated • Attempt {message.question.metadata.generate_attempts}
                        </div>
                    </div>
                )}

                {/* Error Message */}
                {message.error && (
                    <div className="mt-2 flex items-center gap-2 text-xs text-red-600 bg-red-50/50 px-3 py-2 rounded-lg border border-red-100 animate-in fade-in slide-in-from-top-1">
                        <div className="w-1.5 h-1.5 rounded-full bg-red-500"></div>
                        Failed to generate. Please try again or check your connection.
                    </div>
                )}

            </div>
        </div>
    );
}
