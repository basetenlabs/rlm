// Types matching the RLM log format

export interface RLMChatCompletion {
  prompt: string | Record<string, unknown>;
  response: string;
  prompt_tokens: number;
  completion_tokens: number;
  execution_time: number;
  // Present in real harness logs; normalized into the token fields above by parse-logs.
  root_model?: string;
  usage_summary?: Record<string, unknown>;
  error?: string | null;
}

export interface REPLResult {
  stdout: string;
  stderr: string;
  locals: Record<string, unknown>;
  execution_time: number;
  rlm_calls: RLMChatCompletion[];
}

export interface CodeBlock {
  code: string;
  result: REPLResult;
}

export interface RLMIteration {
  type?: string;
  iteration: number;
  timestamp: string;
  prompt: Array<{ role: string; content: string }>;
  response: string;
  code_blocks: CodeBlock[];
  final_answer: string | [string, string] | null;
  iteration_time: number | null;
}

// Metadata saved at the start of a log file about RLM configuration
export interface RLMConfigMetadata {
  root_model: string | null;
  max_depth: number | null;
  max_iterations: number | null;
  backend: string | null;
  backend_kwargs: Record<string, unknown> | null;
  environment_type: string | null;
  environment_kwargs: Record<string, unknown> | null;
  other_backends: string[] | null;
  // Wall-clock seconds spent parsing the data room before the RLM loop (future runs only).
  parse_seconds: number | null;
}

// Per-iteration timing decomposition, derived from data already in the log.
// lmGen + codePure + subCall === total always; lmGenKnown is false when
// iteration_time was not logged, in which case lmGen is 0 but truly unknown.
export interface IterationTiming {
  total: number;     // best-known total (>= measured code time)
  lmGen: number;     // root-model response generation (0 when unknown)
  codePure: number;  // Python REPL execution, excluding sub-LM calls made during exec
  subCall: number;   // sum of sub-LM (rlm_query) call latencies
  lmGenKnown: boolean;
}

// Lightweight recent-trace descriptor shared by /api/logs and the dashboard.
export interface RecentTrace {
  name: string;            // path relative to public/logs/, e.g. "live/pinnacle-heartland.jsonl"
  size: number;
  mtime: number;
  task?: string;           // derived task slug, e.g. "pinnacle-heartland"
  isLive?: boolean;        // mtime is fresh (actively being mirrored)
  iterationCount?: number; // iteration records in the file (best-effort)
}

export interface RLMLogFile {
  fileName: string;
  filePath: string;
  iterations: RLMIteration[];
  metadata: LogMetadata;
  config: RLMConfigMetadata;
}

export interface LogMetadata {
  totalIterations: number;
  totalCodeBlocks: number;
  totalSubLMCalls: number;
  contextQuestion: string;
  finalAnswer: string | null;
  totalExecutionTime: number;
  hasErrors: boolean;
}

export function extractFinalAnswer(answer: string | [string, string] | null): string | null {
  if (!answer) return null;
  if (Array.isArray(answer)) {
    return answer[1];
  }
  return answer;
}

