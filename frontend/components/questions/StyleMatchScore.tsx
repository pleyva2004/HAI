import { Progress } from '@/components/ui/progress';

interface StyleMatchScoreProps {
  score: number; // 0-1
}

export function StyleMatchScore({ score }: StyleMatchScoreProps) {
  const percentage = Math.round(score * 100);

  const getColor = (pct: number) => {
    if (pct >= 90) return 'bg-green-500';
    if (pct >= 70) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="space-y-1">
      <div className="flex justify-between items-center">
        <span className="text-sm font-semibold">{percentage}%</span>
      </div>
      <Progress value={percentage} className="h-2" />
    </div>
  );
}
