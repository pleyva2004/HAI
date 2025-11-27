'use client';

import { ChatMessage as ChatMessageType } from '@/lib/api/types';
import { QuestionList } from '../questions/QuestionList';

interface ChatMessageProps {
  message: ChatMessageType;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-3xl ${isUser ? 'bg-blue-100' : 'bg-gray-100'} rounded-lg p-4`}>
        <div className="text-sm text-gray-600 mb-1">
          {isUser ? 'You' : 'AI Assistant'}
        </div>
        <div className="text-gray-900">{message.content}</div>

        {message.metadata?.file_name && (
          <div className="text-xs text-gray-500 mt-2">
            📎 {message.metadata.file_name}
          </div>
        )}

        {message.metadata?.questions && (
          <div className="mt-4">
            <QuestionList questions={message.metadata.questions} />
          </div>
        )}
      </div>
    </div>
  );
}
