import { NextResponse } from 'next/server';
import { readdir, stat } from 'fs/promises';
import path from 'path';

// List recent trajectory logs in public/logs/ for the "Recent Traces" panel.
// Returns only filename + size + mtime so the client never loads whole files on mount.
export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const dir = path.join(process.cwd(), 'public', 'logs');
    const entries = await readdir(dir);
    const stats = await Promise.all(
      entries
        .filter((f) => f.endsWith('.jsonl'))
        .map(async (name) => {
          const s = await stat(path.join(dir, name));
          return { name, size: s.size, mtime: s.mtimeMs };
        })
    );
    stats.sort((a, b) => b.mtime - a.mtime);
    return NextResponse.json({ files: stats.slice(0, 10) });
  } catch {
    // Directory missing or unreadable — empty list, not an error.
    return NextResponse.json({ files: [] });
  }
}
