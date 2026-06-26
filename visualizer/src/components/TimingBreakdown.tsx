'use client';

import { IterationTiming } from '@/lib/types';

const SEGMENTS: Array<{ key: keyof IterationTiming; label: string; bar: string; text: string }> = [
  { key: 'lmGen', label: 'LM generation', bar: 'bg-sky-500 dark:bg-sky-400', text: 'text-sky-600 dark:text-sky-400' },
  { key: 'codePure', label: 'Code execution', bar: 'bg-emerald-500 dark:bg-emerald-400', text: 'text-emerald-600 dark:text-emerald-400' },
  { key: 'subCall', label: 'Sub-LM calls', bar: 'bg-fuchsia-500 dark:bg-fuchsia-400', text: 'text-fuchsia-600 dark:text-fuchsia-400' },
];

function fmt(s: number): string {
  return s >= 100 ? `${s.toFixed(0)}s` : `${s.toFixed(2)}s`;
}

interface TimingBreakdownProps {
  timing: IterationTiming;
  compact?: boolean;
}

export function TimingBreakdown({ timing, compact = false }: TimingBreakdownProps) {
  const { total } = timing;
  const pct = (v: number) => (total > 0 ? (v / total) * 100 : 0);

  return (
    <div className={compact ? 'space-y-1.5' : 'space-y-2'}>
      <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-muted">
        {SEGMENTS.map(({ key, bar, label }) => {
          const width = pct(timing[key]);
          if (width <= 0) return null;
          return (
            <div
              key={key}
              className={bar}
              style={{ width: `${width}%` }}
              title={`${label}: ${fmt(timing[key])} (${width.toFixed(0)}%)`}
            />
          );
        })}
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] font-mono">
        {SEGMENTS.map(({ key, label, bar, text }) => (
          <span key={key} className="flex items-center gap-1">
            <span className={`inline-block h-2 w-2 rounded-sm ${bar}`} />
            <span className="text-muted-foreground">{compact ? label.split(' ')[0] : label}</span>
            <span className={text}>{fmt(timing[key])}</span>
          </span>
        ))}
        <span className="ml-auto text-muted-foreground">
          total <span className="text-foreground">{fmt(total)}</span>
        </span>
      </div>
    </div>
  );
}
