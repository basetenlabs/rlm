import { NextResponse } from 'next/server';
import { readdir, stat, readFile } from 'fs/promises';
import path from 'path';
import { RecentTrace } from '@/lib/types';

// List trajectory logs for the run picker. Scans public/logs/ (single-file logs) and
// public/logs/live/ (per-task concurrent runs mirrored by scripts/watch_rlm_all.sh).
// Returns name/size/mtime plus derived task, isLive (fresh mtime) and iterationCount.
export const dynamic = 'force-dynamic';

const LIVE_WINDOW_MS = 20_000;          // mtime newer than this => actively mirrored
const COUNT_MAX_BYTES = 20 * 1024 * 1024; // skip line-counting files larger than this

async function describe(baseDir: string, rel: string): Promise<RecentTrace | null> {
  try {
    const full = path.join(baseDir, rel);
    const s = await stat(full);
    if (!s.isFile()) return null;
    let iterationCount: number | undefined;
    if (s.size <= COUNT_MAX_BYTES) {
      try {
        const buf = await readFile(full);
        let n = 0;
        for (let i = 0; i < buf.length; i++) if (buf[i] === 10) n++;
        iterationCount = Math.max(0, n - 1); // minus the metadata record
      } catch {
        /* ignore */
      }
    }
    return {
      name: rel,
      size: s.size,
      mtime: s.mtimeMs,
      task: path.basename(rel).replace(/\.jsonl$/, ''),
      isLive: Date.now() - s.mtimeMs < LIVE_WINDOW_MS,
      iterationCount,
    };
  } catch {
    return null;
  }
}

export async function GET() {
  const dir = path.join(process.cwd(), 'public', 'logs');
  const out: RecentTrace[] = [];

  try {
    for (const f of await readdir(dir)) {
      if (f.endsWith('.jsonl')) {
        const t = await describe(dir, f);
        if (t) out.push(t);
      }
    }
  } catch {
    /* public/logs missing — fine */
  }

  try {
    const liveDir = path.join(dir, 'live');
    for (const f of await readdir(liveDir)) {
      if (f.endsWith('.jsonl')) {
        const t = await describe(dir, path.join('live', f));
        if (t) out.push(t);
      }
    }
  } catch {
    /* no live/ dir yet — fine */
  }

  // Live runs first, then most-recent.
  out.sort((a, b) => (Number(b.isLive) - Number(a.isLive)) || (b.mtime - a.mtime));
  return NextResponse.json({ files: out.slice(0, 50) });
}
