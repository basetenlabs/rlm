'use client';

import { useCallback, useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface FileUploaderProps {
  onFileLoaded: (fileName: string, content: string) => void;
}

// The uploader reads the whole file into memory via file.text(); past this size the
// browser tab risks freezing. Warn (don't silently crash) and point at the trimmer.
const MAX_SAFE_BYTES = 150 * 1024 * 1024;

function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024).toFixed(0)} KB`;
}

export function FileUploader({ onFileLoaded }: FileUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback(async (file: File) => {
    setError(null);
    if (!file.name.endsWith('.jsonl')) {
      setError('Please upload a .jsonl file');
      return;
    }
    if (file.size > MAX_SAFE_BYTES) {
      const ok = window.confirm(
        `${file.name} is ${formatBytes(file.size)} — loading it may freeze the browser.\n` +
        `Consider trimming it first (scripts/trim_rlm_log.py). Load anyway?`
      );
      if (!ok) return;
    }

    setIsLoading(true);
    try {
      const content = await file.text();
      onFileLoaded(file.name, content);
    } catch (err) {
      console.error('Error reading file:', err);
      setError('Failed to read file');
    } finally {
      setIsLoading(false);
    }
  }, [onFileLoaded]);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFile(file);
    }
  }, [handleFile]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFile(file);
    }
  }, [handleFile]);

  return (
    <Card 
      className={cn(
        'border-2 border-dashed transition-all duration-200',
        isDragging 
          ? 'border-[oklch(0.65_0.18_145)] bg-[oklch(0.65_0.18_145/0.05)] scale-[1.01]' 
          : 'border-[oklch(0.25_0.03_145)] hover:border-[oklch(0.5_0.12_145)]'
      )}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
    >
      <CardContent className="p-8 text-center">
        <div className={cn(
          'w-14 h-14 mx-auto mb-4 rounded-xl flex items-center justify-center transition-all border',
          isDragging 
            ? 'bg-[oklch(0.65_0.18_145/0.15)] border-[oklch(0.65_0.18_145/0.3)] scale-105' 
            : 'bg-muted/20 border-[oklch(0.25_0.03_145)]'
        )}>
          <span className={cn(
            'text-2xl transition-colors font-mono',
            isDragging ? 'text-[oklch(0.65_0.18_145)]' : 'text-muted-foreground'
          )}>
            {isLoading ? '...' : '↑'}
          </span>
        </div>
        
        <h3 className="text-sm font-medium mb-1">
          {isDragging ? 'Drop here' : 'Upload .jsonl'}
        </h3>
        <p className="text-muted-foreground text-xs mb-4">
          Drag & drop or click to browse
        </p>
        
        <input
          type="file"
          id="file-upload"
          accept=".jsonl"
          onChange={handleFileSelect}
          className="hidden"
        />
        <Button 
          asChild 
          size="sm"
          className="bg-[oklch(0.55_0.15_145)] hover:bg-[oklch(0.6_0.17_145)] text-white"
        >
          <label htmlFor="file-upload" className="cursor-pointer">
            {isLoading ? 'Loading...' : 'Choose File'}
          </label>
        </Button>

        {error && (
          <p className="mt-3 text-xs text-red-600 dark:text-red-400">{error}</p>
        )}
      </CardContent>
    </Card>
  );
}
