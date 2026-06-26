// Shared formatting + file-size guard, used by the upload and recent-traces paths.

// Past this size, reading a whole log into the browser via file.text() risks
// freezing the tab. Warn (don't silently crash) and point at the trimmer.
export const MAX_SAFE_BYTES = 150 * 1024 * 1024;

export function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${bytes} B`;
}

// Returns true if the user wants to proceed loading an oversized file.
export function confirmLargeFile(name: string, size: number): boolean {
  return window.confirm(
    `${name} is ${formatBytes(size)} — loading it may freeze the browser.\n` +
    `Consider trimming it first (scripts/trim_rlm_log.py). Load anyway?`
  );
}
