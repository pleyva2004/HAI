import { useState, useRef, useEffect } from 'react';
import { Message } from '@/types';
import { api } from '@/services/api';
import MessageBubble from './MessageBubble';
import ChatInput from './ChatInput';

interface ChatInterfaceProps {
    selectedStudent?: { id: string; name: string; grade: string } | null;
}

export default function ChatInterface({ selectedStudent }: ChatInterfaceProps) {
    const [messages, setMessages] = useState<Message[]>([
        {
            id: 'welcome',
            role: 'assistant',
            content: "Hello! I'm Max. I can help you generate SAT practice questions. \n\nYou can describe a question type (e.g., 'quadratic equation word problem') or upload a screenshot of an existing question to get started.",
        },
    ]);

    const [activeWorkflowId, setActiveWorkflowId] = useState<string | null>(null);

    // Reset chat when student changes
    useEffect(() => {
        setActiveWorkflowId(null);
        if (selectedStudent) {
            setMessages([{
                id: `welcome-${selectedStudent.id}`,
                role: 'assistant',
                content: `Hello! I'm ready to generate questions for **${selectedStudent.name}** (${selectedStudent.grade}). \n\nWhat topic should we focus on?`,
            }]);
        } else {
            // Reset to default if deselected (or maybe keep history? For now reset is safer context)
            setMessages([{
                id: 'welcome-default',
                role: 'assistant',
                content: "Hello! I'm Max. I can help you generate SAT practice questions. \n\nYou can describe a question type (e.g., 'quadratic equation word problem') or upload a screenshot of an existing question to get started.",
            }]);
        }
    }, [selectedStudent]);
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const handleSendMessage = async (text: string, image?: string, settings?: {
        section: 'Math' | 'Reading and Writing';
        difficulty: 'Easy' | 'Medium' | 'Hard';
        provideAnswer: boolean;
        fileOut: boolean;
    }) => {
        const isFeedback = activeWorkflowId !== null && !image;
        const userMessageId = Date.now().toString();
        const newUserMessage: Message = {
            id: userMessageId,
            role: 'user',
            content: text,
            image: image,
            isFeedback: isFeedback,
        };

        // Add user message immediately
        setMessages((prev) => [...prev, newUserMessage]);
        setIsLoading(true);

        // Add placeholder AI loading message
        const loadingMessageId = (Date.now() + 1).toString();
        setMessages((prev) => [
            ...prev,
            {
                id: loadingMessageId,
                role: 'assistant',
                isLoading: true, // This triggers the loading bubble animation
            },
        ]);

        try {
            // Append student context if selected
            let finalDescription = text;
            if (selectedStudent) {
                finalDescription = `[Context: Student ${selectedStudent.name}, ${selectedStudent.grade}] ${text}`;
            }

            let response;
            if (isFeedback) {
                response = await api.submitFeedback(activeWorkflowId, finalDescription);
            } else {
                response = await api.generateQuestion({
                    description: finalDescription,
                    image: image,
                    provide_answer: settings?.provideAnswer,
                    file_out: settings?.fileOut,
                    requested_section: settings?.section,
                    requested_difficulty: settings?.difficulty,
                });
            }

            // Update the active workflow ID
            setActiveWorkflowId(response.metadata.workflow_id);

            // Update the loading message with actual content
            setMessages((prev) =>
                prev.map((msg) =>
                    msg.id === loadingMessageId
                        ? {
                            ...msg,
                            isLoading: false,
                            content: "Here is a generated question based on your request:",
                            question: response,
                        }
                        : msg
                )
            );

        } catch (error) {
            console.error('Generation error:', error);

            // Clear workflow ID on error so next message starts fresh
            setActiveWorkflowId(null);

            // Update loading message to show error
            setMessages((prev) =>
                prev.map((msg) =>
                    msg.id === loadingMessageId
                        ? {
                            ...msg,
                            isLoading: false,
                            error: true,
                            content: "I'm sorry, I encountered an error while generating the question. Please try again.",
                        }
                        : msg
                )
            );
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-full bg-transparent relative">
            {/* Chat Area */}
            <div className="flex-1 overflow-y-auto w-full scroll-smooth pt-20">
                <div className="container max-w-4xl mx-auto px-4 min-h-full flex flex-col justify-start pb-4">

                    {messages.map((msg, index) => (
                        <div key={msg.id} className={index === 0 ? 'mt-8' : ''}>
                            <MessageBubble message={msg} />
                        </div>
                    ))}

                    <div ref={messagesEndRef} />
                </div>
            </div>

            {/* Input Area - Fixed placement handled by component itself */}
            <ChatInput onSendMessage={handleSendMessage} isLoading={isLoading} />
        </div>
    );
}
