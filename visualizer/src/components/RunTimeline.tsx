'use client';

import { useMemo } from 'react';
import { cn } from '@/lib/utils';
import { RLMIteration } from '@/lib/types';
import { getIterationTiming, TIMING_SEGMENTS, formatDuration } from '@/lib/timing';

interface RunTimelineProps {
  iterations: RLMIteration[];
  selectedIteration: number;
  onSelectIteration: (index: number) => void;
  parseSeconds?: number | null;
}

export function RunTimeline({ iterations, selectedIteration, onSelectIteration, parseSeconds }: RunTimelineProps) {
  const timings = useMemo(() => iterations.map(getIterationTiming), [iterations]);
  const iterTotal = timings.reduce((a, t) => a + t.total, 0);
  const parse = parseSeconds && parseSeconds > 0 ? parseSeconds : 0;
  const grandTotal = iterTotal + parse;
  if (grandTotal <= 0) return null;

  const pct = (v: number) => (v / grandTotal) * 100;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3 text-[10px] font-mono text-muted-foreground">
        <span className="uppercase tracking-wider font-medium">Run timeline</span>
        <span>{iterations.length} iterations</span>
        <span className="ml-auto">
          wall-clock <span className="text-foreground">{formatDuration(grandTotal)}</span>
          {parse > 0 && <span className="text-muted-foreground/70"> · parse {formatDuration(parse)}</span>}
        </span>
      </div>

      <div className="flex h-9 w-full items-stretch gap-px overflow-x-auto rounded-md bg-muted/40 p-0.5">
        {parse > 0 && (
          <div
            className="flex-shrink-0 rounded-sm bg-muted-foreground/30"
            style={{ width: `${Math.max(pct(parse), 0.5)}%`, minWidth: 4 }}
            title={`Document parsing: ${formatDuration(parse)}`}
          />
        )}
        {timings.map((t, idx) => {
          const isSelected = idx === selectedIteration;
          return (
            <button
              key={idx}
              onClick={() => onSelectIteration(idx)}
              title={`Iteration ${idx + 1}: ${formatDuration(t.total)} (LM ${t.lmGenKnown ? formatDuration(t.lmGen) : '?'} · code ${formatDuration(t.codePure)} · sub ${formatDuration(t.subCall)})`}
              className={cn(
                'group relative flex flex-col-reverse overflow-hidden rounded-sm transition-all',
                isSelected ? 'ring-2 ring-primary ring-offset-1 ring-offset-background' : 'opacity-80 hover:opacity-100'
              )}
              style={{ width: `${Math.max(pct(t.total), 0.4)}%`, minWidth: 5 }}
            >
              {TIMING_SEGMENTS.map(({ key, bar }) => {
                const h = t.total > 0 ? (t[key] / t.total) * 100 : 0;
                if (h <= 0) return null;
                return <span key={key} className={bar} style={{ height: `${h}%` }} />;
              })}
            </button>
          );
        })}
      </div>
    </div>
  );
}
