'use client';

import { InputPanel } from './InputPanel';
import { ChatMessage } from './ChatMessage';
import { useChatStore } from '@/lib/stores/chatStore';
import { useGenerate } from '@/lib/hooks/useGenerate';
import { toast } from 'sonner';

export function ChatInterface() {
  const { messages, addMessage, setGenerating, isGenerating } = useChatStore();
  const generateMutation = useGenerate();

  const handleSubmit = async (data: {
    description: string;
    numQuestions: number;
    targetDifficulty?: number;
    file?: File;
  }) => {
    // Add user message
    addMessage({
      id: Date.now().toString(),
      role: 'user',
      content: data.description || 'Generate questions from uploaded file',
      timestamp: new Date(),
      metadata: { file_name: data.file?.name },
    });

    setGenerating(true);

    try {
      const result = await generateMutation.mutateAsync({
        description: data.description,
        num_questions: data.numQuestions,
        target_difficulty: data.targetDifficulty,
        prefer_real: false,
        file: data.file,
      });

      // Add assistant response
      addMessage({
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Generated ${result.questions.length} questions successfully!`,
        timestamp: new Date(),
        metadata: { questions: result.questions },
      });

      toast.success(`Generated ${result.questions.length} questions!`);
    } catch (error) {
      console.error('Generation error:', error);
      toast.error('Failed to generate questions. Please try again.');

      addMessage({
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, there was an error generating questions. Please try again.',
        timestamp: new Date(),
      });
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto min-h-screen flex flex-col">
      {/* Messages */}
      <div className="flex-1 space-y-4 mb-6 py-8">
        {messages.length === 0 ? (
          <div className="text-center text-gray-500 py-12">
            <h2 className="text-2xl font-bold mb-2">SAT Question Generator</h2>
            <p className="mb-4">Upload example questions or describe what you need</p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8 text-left">
              <div className="p-4 border rounded-lg">
                <h3 className="font-semibold mb-2">📝 Style Matching</h3>
                <p className="text-sm">90%+ match to your uploaded examples</p>
              </div>
              <div className="p-4 border rounded-lg">
                <h3 className="font-semibold mb-2">🎯 Precise Difficulty</h3>
                <p className="text-sm">Calibrated on 0-100 scale, not vague labels</p>
              </div>
              <div className="p-4 border rounded-lg">
                <h3 className="font-semibold mb-2">✅ Multi-Model Validation</h3>
                <p className="text-sm">Cross-checked by GPT-4 and Claude</p>
              </div>
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))
        )}
      </div>

      {/* Input Panel */}
      <InputPanel onSubmit={handleSubmit} isGenerating={isGenerating} />
    </div>
  );
}
