'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { Slider } from '@/components/ui/slider';
import { Card } from '@/components/ui/card';
import { Upload, Send } from 'lucide-react';

interface InputPanelProps {
  onSubmit: (data: {
    description: string;
    numQuestions: number;
    targetDifficulty?: number;
    file?: File;
  }) => void;
  isGenerating?: boolean;
}

export function InputPanel({ onSubmit, isGenerating = false }: InputPanelProps) {
  const [description, setDescription] = useState('');
  const [numQuestions, setNumQuestions] = useState(5);
  const [targetDifficulty, setTargetDifficulty] = useState<number>(50);
  const [file, setFile] = useState<File | undefined>();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = () => {
    if (!description && !file) return;

    onSubmit({
      description,
      numQuestions,
      targetDifficulty,
      file,
    });

    // Reset form
    setDescription('');
    setFile(undefined);
  };

  return (
    <Card className="p-6 sticky bottom-0 bg-white border-t shadow-lg">
      <div className="space-y-4">
        {/* Description Input */}
        <Textarea
          placeholder="Describe the type of SAT questions you want to generate..."
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          rows={3}
          disabled={isGenerating}
        />

        {/* File Upload */}
        <div className="flex items-center gap-2">
          <Input
            type="file"
            accept=".pdf,.png,.jpg,.jpeg"
            onChange={handleFileChange}
            disabled={isGenerating}
            className="flex-1"
          />
          {file && (
            <span className="text-sm text-gray-600 flex items-center gap-1">
              <Upload className="w-4 h-4" />
              {file.name}
            </span>
          )}
        </div>

        {/* Configuration */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Number of Questions */}
          <div>
            <label className="text-sm font-medium mb-2 block">
              Number of Questions: {numQuestions}
            </label>
            <Slider
              value={[numQuestions]}
              onValueChange={(value) => setNumQuestions(value[0])}
              min={1}
              max={20}
              step={1}
              disabled={isGenerating}
            />
          </div>

          {/* Target Difficulty */}
          <div>
            <label className="text-sm font-medium mb-2 block">
              Target Difficulty: {targetDifficulty}/100
            </label>
            <Slider
              value={[targetDifficulty]}
              onValueChange={(value) => setTargetDifficulty(value[0])}
              min={0}
              max={100}
              step={5}
              disabled={isGenerating}
            />
          </div>
        </div>

        {/* Submit Button */}
        <Button
          onClick={handleSubmit}
          disabled={(!description && !file) || isGenerating}
          className="w-full"
        >
          <Send className="w-4 h-4 mr-2" />
          {isGenerating ? 'Generating...' : 'Generate Questions'}
        </Button>
      </div>
    </Card>
  );
}
