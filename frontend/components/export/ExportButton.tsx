'use client';

import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Download } from 'lucide-react';
import { SATQuestion } from '@/lib/api/types';
import { useExport } from '@/lib/hooks/useExport';

interface ExportButtonProps {
  questions: SATQuestion[];
}

export function ExportButton({ questions }: ExportButtonProps) {
  const { exportToPDF, exportToJSON } = useExport();

  if (questions.length === 0) return null;

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="outline" className="gap-2">
          <Download className="w-4 h-4" />
          Export Questions
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Export Questions</DialogTitle>
          <DialogDescription>
            Choose a format to export {questions.length} questions
          </DialogDescription>
        </DialogHeader>
        <div className="space-y-2 pt-4">
          <Button
            onClick={() => exportToPDF(questions)}
            variant="outline"
            className="w-full justify-start"
          >
            Export as PDF
          </Button>
          <Button
            onClick={() => exportToJSON(questions)}
            variant="outline"
            className="w-full justify-start"
          >
            Export as JSON
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
