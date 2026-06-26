'use client';

import { useState, useCallback, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { FileUploader } from './FileUploader';
import { LogViewer } from './LogViewer';
import { AsciiRLM } from './AsciiGlobe';
import { ThemeToggle } from './ThemeToggle';
import { Button } from '@/components/ui/button';
import { parseLogFile } from '@/lib/parse-logs';
import { RLMLogFile, RecentTrace } from '@/lib/types';
import { MAX_SAFE_BYTES, formatBytes, confirmLargeFile } from '@/lib/format';
import { cn } from '@/lib/utils';

export function Dashboard() {
  const [logFiles, setLogFiles] = useState<RLMLogFile[]>([]);
  const [selectedLog, setSelectedLog] = useState<RLMLogFile | null>(null);
  const [demoLogs, setDemoLogs] = useState<RecentTrace[]>([]);
  const [loadingDemos, setLoadingDemos] = useState(true);
  // Live mode: poll a continuously-mirrored trajectory and re-render as it grows.
  // Feed it with `scripts/watch_rlm_live.sh <run-id>` (writes public/logs/live.jsonl).
  const [liveMode, setLiveMode] = useState(false);
  const LIVE_FILE = 'live.jsonl';

  // Load the recent-traces list on mount. The API returns only name/size/mtime,
  // so we never pull whole (potentially multi-GB) log files into memory here.
  useEffect(() => {
    async function loadDemoPreviews() {
      try {
        const listResponse = await fetch('/api/logs');
        if (!listResponse.ok) throw new Error('Failed to fetch log list');
        const { files } = await listResponse.json();
        setDemoLogs(files as RecentTrace[]);
      } catch (e) {
        console.error('Failed to load recent traces:', e);
      } finally {
        setLoadingDemos(false);
      }
    }

    loadDemoPreviews();
  }, []);

  // Live polling: re-fetch the mirrored trajectory and update the open log.
  useEffect(() => {
    if (!liveMode) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const res = await fetch(`/logs/${LIVE_FILE}?t=${Date.now()}`, { cache: 'no-store' });
        if (!res.ok || cancelled) return;
        const content = await res.text();
        if (cancelled || !content.trim()) return;
        const parsed = parseLogFile(LIVE_FILE, content);
        setSelectedLog(parsed);
        setLogFiles(prev =>
          prev.some(f => f.fileName === LIVE_FILE)
            ? prev.map(f => (f.fileName === LIVE_FILE ? parsed : f))
            : [...prev, parsed]
        );
      } catch {
        /* transient fetch/parse error mid-write — ignore, next tick retries */
      }
    };
    poll();
    const id = setInterval(poll, 4000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [liveMode]);

  const handleFileLoaded = useCallback((fileName: string, content: string) => {
    const parsed = parseLogFile(fileName, content);
    setLogFiles(prev => {
      if (prev.some(f => f.fileName === fileName)) {
        return prev.map(f => f.fileName === fileName ? parsed : f);
      }
      return [...prev, parsed];
    });
    setSelectedLog(parsed);
  }, []);

  const loadDemoLog = useCallback(async (fileName: string, size: number) => {
    if (size > MAX_SAFE_BYTES && !confirmLargeFile(fileName, size)) return;
    try {
      const response = await fetch(`/logs/${fileName}`);
      if (!response.ok) throw new Error('Failed to load demo log');
      const content = await response.text();
      handleFileLoaded(fileName, content);
    } catch (error) {
      console.error('Error loading demo log:', error);
      alert('Failed to load demo log. Make sure the log files are in the public/logs folder.');
    }
  }, [handleFileLoaded]);

  if (selectedLog) {
    return (
      <LogViewer
        logFile={selectedLog}
        live={liveMode}
        onBack={() => { setSelectedLog(null); setLiveMode(false); }}
      />
    );
  }

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0 grid-pattern opacity-30 dark:opacity-15" />
      <div className="absolute top-0 left-1/3 w-[500px] h-[500px] bg-primary/5 rounded-full blur-3xl" />
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-primary/3 rounded-full blur-3xl" />
      
      <div className="relative z-10">
        {/* Header */}
        <header className="border-b border-border">
          <div className="max-w-7xl mx-auto px-6 py-6">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold tracking-tight">
                  <span className="text-primary">RLM</span>
                  <span className="text-muted-foreground ml-2 font-normal">Visualizer</span>
                </h1>
                <p className="text-sm text-muted-foreground mt-1">
                  Debug recursive language model execution traces
                </p>
              </div>
              <div className="flex items-center gap-4">
                <ThemeToggle />
                <div className="flex items-center gap-2 text-[10px] text-muted-foreground font-mono">
                  <span className="flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                    READY
                  </span>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-6 py-8">
          <div className="grid lg:grid-cols-2 gap-10">
            {/* Left Column - Upload & ASCII Art */}
            <div className="space-y-8">
              {/* Upload Section */}
              <div>
                <h2 className="text-sm font-medium mb-3 flex items-center gap-2 text-muted-foreground">
                  <span className="text-primary font-mono">01</span>
                  Upload Log File
                </h2>
                <FileUploader onFileLoaded={handleFileLoaded} />
                <div className="mt-3 flex items-center gap-3">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setLiveMode(true)}
                    className="font-mono text-xs"
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse mr-2" />
                    Watch Live
                  </Button>
                  <span className="text-[10px] text-muted-foreground font-mono">
                    polls <code>/logs/live.jsonl</code> every 4s — feed with{' '}
                    <code>scripts/watch_rlm_live.sh</code>
                  </span>
                </div>
              </div>
              
              {/* ASCII Architecture Diagram */}
              <div className="hidden lg:block">
                <h2 className="text-sm font-medium mb-3 flex items-center gap-2 text-muted-foreground">
                  <span className="text-primary font-mono">◈</span>
                  RLM Architecture
                </h2>
                <div className="bg-muted/50 border border-border rounded-lg p-4 overflow-x-auto">
                  <AsciiRLM />
                </div>
              </div>
            </div>

            {/* Right Column - Demo Logs & Loaded Files */}
            <div className="space-y-8">
              {/* Demo Logs Section */}
              <div>
                <h2 className="text-sm font-medium mb-3 flex items-center gap-2 text-muted-foreground">
                  <span className="text-primary font-mono">02</span>
                  Recent Traces
                  <span className="text-[10px] text-muted-foreground/60 ml-1">(latest 10)</span>
                </h2>
                
                {loadingDemos ? (
                  <Card>
                    <CardContent className="p-6 text-center">
                      <div className="animate-pulse flex items-center justify-center gap-2 text-muted-foreground text-sm">
                        Loading traces...
                      </div>
                    </CardContent>
                  </Card>
                ) : demoLogs.length === 0 ? (
                  <Card className="border-dashed">
                    <CardContent className="p-6 text-center text-muted-foreground text-sm">
                      No log files found in /public/logs/
                    </CardContent>
                  </Card>
                ) : (
                  <ScrollArea className="h-[320px]">
                    <div className="space-y-2 pr-4">
                      {demoLogs.map((demo) => (
                        <Card
                          key={demo.name}
                          onClick={() => loadDemoLog(demo.name, demo.size)}
                          className={cn(
                            'cursor-pointer transition-all hover:scale-[1.01]',
                            'hover:border-primary/50 hover:bg-primary/5'
                          )}
                        >
                          <CardContent className="p-3">
                            <div className="flex items-center gap-3">
                              <div className="w-2.5 h-2.5 rounded-full bg-muted-foreground/30 flex-shrink-0" />
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2">
                                  <span className="font-mono text-xs text-foreground/80 truncate">
                                    {demo.name}
                                  </span>
                                  <Badge
                                    variant="outline"
                                    className={cn(
                                      'text-[9px] px-1.5 py-0 h-4 ml-auto flex-shrink-0',
                                      demo.size > MAX_SAFE_BYTES && 'border-amber-500/50 text-amber-600 dark:text-amber-400'
                                    )}
                                  >
                                    {formatBytes(demo.size)}
                                  </Badge>
                                </div>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </ScrollArea>
                )}
              </div>

              {/* Loaded Files Section */}
              {logFiles.length > 0 && (
                <div>
                  <h2 className="text-sm font-medium mb-3 flex items-center gap-2 text-muted-foreground">
                    <span className="text-primary font-mono">03</span>
                    Loaded Files
                  </h2>
                  <ScrollArea className="h-[200px]">
                    <div className="space-y-2 pr-4">
                      {logFiles.map((log) => (
                        <Card
                          key={log.fileName}
                          className={cn(
                            'cursor-pointer transition-all hover:scale-[1.01]',
                            'hover:border-primary/50 hover:bg-primary/5'
                          )}
                          onClick={() => setSelectedLog(log)}
                        >
                          <CardContent className="p-3">
                            <div className="flex items-center gap-3">
                              <div className="relative flex-shrink-0">
                                <div className={cn(
                                  'w-2.5 h-2.5 rounded-full',
                                  log.metadata.finalAnswer 
                                    ? 'bg-primary' 
                                    : 'bg-muted-foreground/30'
                                )} />
                                {log.metadata.finalAnswer && (
                                  <div className="absolute inset-0 w-2.5 h-2.5 rounded-full bg-primary animate-ping opacity-50" />
                                )}
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 mb-1">
                                  <span className="font-mono text-xs truncate text-foreground/80">
                                    {log.fileName}
                                  </span>
                                  <Badge variant="outline" className="text-[9px] px-1.5 py-0 h-4">
                                    {log.metadata.totalIterations} iter
                                  </Badge>
                                </div>
                                <p className="text-[11px] text-muted-foreground truncate">
                                  {log.metadata.contextQuestion}
                                </p>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </ScrollArea>
                </div>
              )}
            </div>
          </div>
        </main>

        {/* Footer */}
        <footer className="border-t border-border mt-8">
          <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
            <p className="text-[10px] text-muted-foreground font-mono">
              RLM Visualizer • Recursive Language Models
            </p>
            <p className="text-[10px] text-muted-foreground font-mono">
              Prompt → [LM ↔ REPL] → Answer
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
}
