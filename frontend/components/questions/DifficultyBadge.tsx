import { Badge } from '@/components/ui/badge';

interface DifficultyBadgeProps {
  difficulty: number; // 0-100
}

export function DifficultyBadge({ difficulty }: DifficultyBadgeProps) {
  const getColor = (score: number) => {
    if (score < 30) return 'bg-green-100 text-green-800 border-green-300';
    if (score < 60) return 'bg-yellow-100 text-yellow-800 border-yellow-300';
    return 'bg-red-100 text-red-800 border-red-300';
  };

  const getLabel = (score: number) => {
    if (score < 30) return 'Easy';
    if (score < 60) return 'Medium';
    return 'Hard';
  };

  return (
    <div className="flex items-center gap-2">
      <Badge className={getColor(difficulty)}>{getLabel(difficulty)}</Badge>
      <span className="text-sm text-gray-600">{difficulty.toFixed(0)}/100</span>
    </div>
  );
}
