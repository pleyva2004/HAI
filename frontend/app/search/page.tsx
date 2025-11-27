'use client';

import { useState } from 'react';
import { Header } from '@/components/layout/Header';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Card } from '@/components/ui/card';
import { QuestionList } from '@/components/questions/QuestionList';
import { useSearch } from '@/lib/hooks/useSearch';
import { Search as SearchIcon } from 'lucide-react';
import { toast } from 'sonner';

export default function SearchPage() {
  const [query, setQuery] = useState('');
  const [difficultyRange, setDifficultyRange] = useState<[number, number]>([0, 100]);
  const [limit, setLimit] = useState(10);

  const searchMutation = useSearch();

  const handleSearch = async () => {
    if (!query) {
      toast.error('Please enter a search query');
      return;
    }

    try {
      await searchMutation.mutateAsync({
        query,
        difficulty_min: difficultyRange[0],
        difficulty_max: difficultyRange[1],
        limit,
      });

      toast.success(`Found ${searchMutation.data?.length || 0} questions`);
    } catch (error) {
      toast.error('Search failed. Please try again.');
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      <main className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-8">Search Question Bank</h1>

          <Card className="p-6 mb-8">
            <div className="space-y-4">
              {/* Search Input */}
              <div className="flex gap-2">
                <Input
                  placeholder="Search for questions (e.g., 'algebra quadratic equations')..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                  className="flex-1"
                />
                <Button onClick={handleSearch} disabled={searchMutation.isPending}>
                  <SearchIcon className="w-4 h-4 mr-2" />
                  {searchMutation.isPending ? 'Searching...' : 'Search'}
                </Button>
              </div>

              {/* Filters */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Difficulty Range */}
                <div>
                  <label className="text-sm font-medium mb-2 block">
                    Difficulty Range: {difficultyRange[0]} - {difficultyRange[1]}
                  </label>
                  <Slider
                    value={difficultyRange}
                    onValueChange={(value) => setDifficultyRange(value as [number, number])}
                    min={0}
                    max={100}
                    step={5}
                  />
                </div>

                {/* Number of Results */}
                <div>
                  <label className="text-sm font-medium mb-2 block">
                    Number of Results: {limit}
                  </label>
                  <Slider
                    value={[limit]}
                    onValueChange={(value) => setLimit(value[0])}
                    min={1}
                    max={50}
                    step={1}
                  />
                </div>
              </div>
            </div>
          </Card>

          {/* Results */}
          {searchMutation.data && searchMutation.data.length > 0 && (
            <div>
              <h2 className="text-xl font-semibold mb-4">
                Results ({searchMutation.data.length})
              </h2>
              <QuestionList questions={searchMutation.data} />
            </div>
          )}

          {searchMutation.data && searchMutation.data.length === 0 && (
            <div className="text-center text-gray-500 py-12">
              No questions found. Try adjusting your search criteria.
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
