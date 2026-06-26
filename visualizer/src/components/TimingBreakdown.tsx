'use client';

import { IterationTiming } from '@/lib/types';
import { TIMING_SEGMENTS, formatDuration } from '@/lib/timing';

interface TimingBreakdownProps {
  timing: IterationTiming;
}

export function TimingBreakdown({ timing }: TimingBreakdownProps) {
  const { total, lmGenKnown } = timing;
  const pct = (v: number) => (total > 0 ? (v / total) * 100 : 0);

  return (
    <div className="space-y-2">
      <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-muted">
        {TIMING_SEGMENTS.map(({ key, bar, label }) => {
          const width = pct(timing[key]);
          if (width <= 0) return null;
          return (
            <div
              key={key}
              className={bar}
              style={{ width: `${width}%` }}
              title={`${label}: ${formatDuration(timing[key])} (${width.toFixed(0)}%)`}
            />
          );
        })}
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] font-mono">
        {TIMING_SEGMENTS.map(({ key, label, bar, text }) => {
          // Root-model time isn't logged for this iteration — say so rather than "0s".
          const unknown = key === 'lmGen' && !lmGenKnown;
          return (
            <span key={key} className="flex items-center gap-1">
              <span className={`inline-block h-2 w-2 rounded-sm ${bar}`} />
              <span className="text-muted-foreground">{label}</span>
              <span className={unknown ? 'text-muted-foreground/50' : text}>
                {unknown ? '—' : formatDuration(timing[key])}
              </span>
            </span>
          );
        })}
        <span className="ml-auto text-muted-foreground">
          total <span className="text-foreground">{formatDuration(total)}</span>
          {!lmGenKnown && <span className="text-muted-foreground/50"> (measured)</span>}
        </span>
      </div>
    </div>
  );
}
