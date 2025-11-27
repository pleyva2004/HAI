import { Badge } from '@/components/ui/badge';

interface SourceBadgeProps {
  isReal: boolean;
}

export function SourceBadge({ isReal }: SourceBadgeProps) {
  if (isReal) {
    return (
      <Badge className="bg-amber-100 text-amber-800 border-amber-300">
        Official SAT
      </Badge>
    );
  }

  return (
    <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-300">
      AI Generated
    </Badge>
  );
}
