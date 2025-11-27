import { SATQuestion } from '@/lib/api/types';
import { QuestionCard } from './QuestionCard';

interface QuestionListProps {
  questions: SATQuestion[];
}

export function QuestionList({ questions }: QuestionListProps) {
  if (questions.length === 0) {
    return null;
  }

  return (
    <div className="space-y-4">
      {questions.map((question, index) => (
        <QuestionCard key={question.id} question={question} index={index} />
      ))}
    </div>
  );
}
