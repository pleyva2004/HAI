import { useState, useRef, FormEvent, useEffect } from 'react';
import { api } from '@/services/api';
import GenerationSettings from './GenerationSettings';

interface ChatInputProps {
    onSendMessage: (text: string, image?: string, settings?: {
        section: 'Math' | 'Reading and Writing';
        difficulty: 'Easy' | 'Medium' | 'Hard';
        provideAnswer: boolean;
        fileOut: boolean;
    }) => void;
    isLoading: boolean;
}

export default function ChatInput({ onSendMessage, isLoading }: ChatInputProps) {
    const [input, setInput] = useState('');
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [isUploading, setIsUploading] = useState(false);

    // Settings State
    const [showSettings, setShowSettings] = useState(false);
    const [settings, setSettings] = useState<{
        section: 'Math' | 'Reading and Writing';
        difficulty: 'Easy' | 'Medium' | 'Hard';
        provideAnswer: boolean;
        fileOut: boolean;
    }>({
        section: 'Math',
        difficulty: 'Medium',
        provideAnswer: true,
        fileOut: false
    });

    const fileInputRef = useRef<HTMLInputElement>(null);
    const settingsRef = useRef<HTMLDivElement>(null);
    const settingsButtonRef = useRef<HTMLButtonElement>(null);

    // Close settings when clicking outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (
                showSettings &&
                settingsRef.current &&
                settingsButtonRef.current &&
                !settingsRef.current.contains(event.target as Node) &&
                !settingsButtonRef.current.contains(event.target as Node)
            ) {
                setShowSettings(false);
            }
        };

        if (showSettings) {
            document.addEventListener('mousedown', handleClickOutside);
        }

        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
        };
    }, [showSettings]);

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        if ((!input.trim() && !selectedFile) || isLoading || isUploading) return;

        let base64Image: string | undefined = undefined;

        // Handle image upload if selected
        if (selectedFile) {
            try {
                setIsUploading(true);
                const result = await api.uploadScreenshot(selectedFile);
                base64Image = result.image;
            } catch (error) {
                console.error('Upload failed:', error);
                alert('Failed to upload image');
                setIsUploading(false);
                return;
            }
        }

        onSendMessage(input, base64Image, settings);

        // Reset state
        setInput('');
        setSelectedFile(null);
        setPreview(null);
        setIsUploading(false);
        // Do NOT reset settings, user might want to keep them
    };

    const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            if (file.size > 5 * 1024 * 1024) { // 5MB limit
                alert('File size too large. Please select an image under 5MB.');
                return;
            }
            setSelectedFile(file);
            const objectUrl = URL.createObjectURL(file);
            setPreview(objectUrl);
        }
    };

    const clearFile = () => {
        setSelectedFile(null);
        setPreview(null);
        if (fileInputRef.current) fileInputRef.current.value = '';
    };

    return (
        <div className="w-full px-4 z-40 pb-6 pt-2">
            <div className="container max-w-3xl mx-auto">

                {/* Image Preview - Floating above input */}
                {preview && (
                    <div className="mb-3 relative inline-block animate-in fade-in slide-in-from-bottom-2 duration-300">
                        <div className="relative h-20 w-20 rounded-2xl overflow-hidden border border-white/40 shadow-lg ring-1 ring-black/5 group">
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img src={preview} alt="Preview" className="h-full w-full object-cover" />
                            <div className="absolute inset-0 bg-black/10 group-hover:bg-black/20 transition-colors" />
                            <button
                                onClick={clearFile}
                                type="button"
                                className="absolute top-1 right-1 bg-black/50 hover:bg-black/70 text-white rounded-full p-1 opacity-0 group-hover:opacity-100 transition-opacity backdrop-blur-md"
                            >
                                <svg suppressHydrationWarning xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                            </button>
                        </div>
                    </div>
                )}

                <form
                    onSubmit={handleSubmit}
                    className="relative flex items-center gap-2 px-2 py-2 rounded-full bg-white dark:bg-[#151516] border border-gray-200 dark:border-white/10 shadow-xl shadow-black/5 transition-all duration-300"
                >
                    {/* Settings Popover */}
                    {showSettings && (
                        <div ref={settingsRef} className="absolute bottom-full left-0 mb-2 z-50">
                            <GenerationSettings
                                isOpen={showSettings}
                                onClose={() => setShowSettings(false)}
                                settings={settings}
                                onSettingsChange={setSettings}
                            />
                        </div>
                    )}
                    {/* Settings Toggle */}
                    <button
                        ref={settingsButtonRef}
                        type="button"
                        onClick={() => setShowSettings(!showSettings)}
                        className={`p-2 transition-colors duration-200 ${showSettings ? 'text-orange-500' : 'text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300'}`}
                        title="Generation Settings"
                    >
                        <svg suppressHydrationWarning xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M20 7h-9"></path><path d="M14 17H5"></path><circle cx="17" cy="17" r="3"></circle><circle cx="7" cy="7" r="3"></circle>
                        </svg>
                    </button>

                    {/* File Upload Button */}
                    <button
                        type="button"
                        onClick={() => fileInputRef.current?.click()}
                        disabled={isLoading || isUploading}
                        className="p-2 text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300 transition-colors"
                        title="Upload Image"
                    >
                        <svg suppressHydrationWarning
                            xmlns="http://www.w3.org/2000/svg"
                            width="24"
                            height="24"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="1.5"
                            strokeLinecap="round"
                            strokeLinejoin="round"

                        >
                            <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path>
                        </svg>
                        <input
                            type="file"
                            ref={fileInputRef}
                            onChange={handleFileSelect}
                            accept="image/*"
                            className="hidden"
                        />
                    </button>

                    {/* Text Input */}
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask anything..."
                        disabled={isLoading || isUploading}
                        className="flex-1 bg-transparent border-none focus:ring-0 focus:outline-none py-2 text-[#1D1D1F] dark:text-[#E0E0E0] placeholder-gray-500 dark:placeholder-gray-500 text-[16px] leading-relaxed"
                    />

                    {/* Send Button */}
                    <button
                        type="submit"
                        disabled={(!input.trim() && !selectedFile) || isLoading || isUploading}
                        className={`p-2 transition-colors duration-200 ${(!input.trim() && !selectedFile) || isLoading || isUploading
                            ? 'text-gray-300 dark:text-gray-600 cursor-not-allowed'
                            : 'text-[#1D1D1F] dark:text-gray-300 hover:text-black dark:hover:text-white'
                            }`}
                    >
                        {isLoading || isUploading ? (
                            <div className="w-5 h-5 border-[2px] border-gray-300 border-t-gray-500 rounded-full animate-spin" />
                        ) : (
                            <svg suppressHydrationWarning
                                xmlns="http://www.w3.org/2000/svg"
                                width="24"
                                height="24"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="1.5"
                                strokeLinecap="round"
                                strokeLinejoin="round"

                            >
                                <line x1="22" y1="2" x2="11" y2="13"></line>
                                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                            </svg>
                        )}
                    </button>
                </form>

                <div className="text-center mt-3">
                    <p className="text-[10px] text-gray-400 font-medium tracking-tight opacity-0 animate-in fade-in slide-in-from-bottom-1 duration-700 delay-500 fill-mode-forwards">
                        Hilltop AI
                    </p>
                </div>
            </div>
        </div>
    );
}
