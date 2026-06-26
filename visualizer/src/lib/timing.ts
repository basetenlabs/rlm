// Canonical per-iteration timing: one source of truth for how an RLM iteration's
// wall-clock decomposes into LM-generation / pure-code / sub-LM-call time, plus the
// shared segment model and duration formatting used by every timing view.
import { RLMIteration, IterationTiming } from './types';

export function formatDuration(seconds: number): string {
  if (seconds >= 3600) return `${(seconds / 3600).toFixed(1)}h`;
  if (seconds >= 60) return `${(seconds / 60).toFixed(1)}m`;
  if (seconds >= 100) return `${seconds.toFixed(0)}s`;
  return `${seconds.toFixed(2)}s`;
}

// Decompose an iteration's wall-clock. codePure and subCall are always measured;
// lmGen (root-model generation) is only knowable when iteration_time was logged —
// otherwise lmGenKnown is false and callers must render "unknown", not 0.
// Segments always satisfy lmGen + codePure + subCall === total.
export function getIterationTiming(iter: RLMIteration): IterationTiming {
  let codeTotal = 0;
  let subCall = 0;
  for (const block of iter.code_blocks ?? []) {
    codeTotal += block.result?.execution_time ?? 0;
    for (const call of block.result?.rlm_calls ?? []) {
      subCall += call.execution_time ?? 0;
    }
  }
  const codePure = Math.max(0, codeTotal - subCall);
  const lmGenKnown = iter.iteration_time != null;
  // Never let the reported total dip below measured code time (clock skew / async
  // overlap), which would otherwise push segments past 100%.
  const total = lmGenKnown ? Math.max(iter.iteration_time as number, codeTotal) : codeTotal;
  const lmGen = total - codeTotal; // 0 when unknown; honest flag carries the distinction
  return { total, lmGen, codePure, subCall, lmGenKnown };
}

export interface TimingSegment {
  key: 'lmGen' | 'codePure' | 'subCall';
  label: string;
  bar: string;
  text: string;
}

export const TIMING_SEGMENTS: TimingSegment[] = [
  { key: 'lmGen', label: 'LM generation', bar: 'bg-sky-500 dark:bg-sky-400', text: 'text-sky-600 dark:text-sky-400' },
  { key: 'codePure', label: 'Code execution', bar: 'bg-emerald-500 dark:bg-emerald-400', text: 'text-emerald-600 dark:text-emerald-400' },
  { key: 'subCall', label: 'Sub-LM calls', bar: 'bg-fuchsia-500 dark:bg-fuchsia-400', text: 'text-fuchsia-600 dark:text-fuchsia-400' },
];
