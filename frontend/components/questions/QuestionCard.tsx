import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { SATQuestion } from '@/lib/api/types';
import { DifficultyBadge } from './DifficultyBadge';
import { StyleMatchScore } from './StyleMatchScore';
import { SourceBadge } from './SourceBadge';

interface QuestionCardProps {
  question: SATQuestion;
  index: number;
}

export function QuestionCard({ question, index }: QuestionCardProps) {
  return (
    <Card className="w-full mb-4 hover:shadow-lg transition-shadow">
      <CardHeader>
        <div className="flex justify-between items-start">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-sm font-semibold text-gray-500">
                Question {index + 1}
              </span>
              <SourceBadge isReal={question.is_real} />
              <Badge variant="outline">{question.category}</Badge>
            </div>
            <CardTitle className="text-lg">{question.question}</CardTitle>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        {/* Answer Choices */}
        <div className="space-y-2 mb-4">
          {Object.entries(question.choices).map(([key, value]) => (
            <div
              key={key}
              className={`p-3 rounded-lg border ${
                key === question.correct_answer
                  ? 'bg-green-50 border-green-300'
                  : 'bg-gray-50 border-gray-200'
              }`}
            >
              <span className="font-semibold mr-2">{key}.</span>
              {value}
            </div>
          ))}
        </div>

        {/* Metadata Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4 border-t">
          {/* Difficulty */}
          <div>
            <p className="text-sm text-gray-500 mb-1">Difficulty</p>
            <DifficultyBadge difficulty={question.difficulty} />
          </div>

          {/* Style Match Score (AI-generated only) */}
          {!question.is_real && question.style_match_score !== undefined && question.style_match_score !== null && (
            <div>
              <p className="text-sm text-gray-500 mb-1">Style Match</p>
              <StyleMatchScore score={question.style_match_score} />
            </div>
          )}

          {/* Predicted Correct Rate */}
          {question.predicted_correct_rate !== undefined && question.predicted_correct_rate !== null && (
            <div>
              <p className="text-sm text-gray-500 mb-1">Predicted Correct Rate</p>
              <p className="text-lg font-semibold">
                {question.predicted_correct_rate.toFixed(0)}%
              </p>
            </div>
          )}
        </div>

        {/* Explanation (Collapsible) */}
        <details className="mt-4">
          <summary className="cursor-pointer text-sm font-medium text-blue-600 hover:text-blue-800">
            View Explanation
          </summary>
          <p className="mt-2 text-sm text-gray-700 p-3 bg-blue-50 rounded-lg">
            {question.explanation}
          </p>
        </details>
      </CardContent>
    </Card>
  );
}
