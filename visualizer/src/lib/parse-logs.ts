import { RLMIteration, RLMLogFile, LogMetadata, RLMConfigMetadata, IterationTiming, extractFinalAnswer } from './types';

// Real harness logs record sub-call token usage under `usage_summary`, shaped like
//   { model_usage_summaries: { "<model>": { total_input_tokens, total_output_tokens } } }
// rather than flat prompt_tokens/completion_tokens. Normalize (summing across models,
// since one sub-call may fan out to several) so the UI can rely on the flat fields.
function sumUsage(usage: Record<string, unknown> | undefined, kind: 'input' | 'output'): number {
  if (!usage) return 0;
  const totalKey = kind === 'input' ? 'total_input_tokens' : 'total_output_tokens';
  const mus = usage['model_usage_summaries'];
  if (mus && typeof mus === 'object') {
    let sum = 0;
    for (const m of Object.values(mus as Record<string, unknown>)) {
      const v = (m as Record<string, unknown>)?.[totalKey];
      if (typeof v === 'number') sum += v;
    }
    return sum;
  }
  // Fallbacks: flat OpenAI shape, or a top-level total.
  const flatKey = kind === 'input' ? 'prompt_tokens' : 'completion_tokens';
  const flat = usage[flatKey] ?? usage[totalKey];
  return typeof flat === 'number' ? flat : 0;
}

function normalizeIteration(iter: RLMIteration): RLMIteration {
  for (const block of iter.code_blocks ?? []) {
    for (const call of block.result?.rlm_calls ?? []) {
      if (typeof call.prompt_tokens !== 'number') {
        call.prompt_tokens = sumUsage(call.usage_summary, 'input');
      }
      if (typeof call.completion_tokens !== 'number') {
        call.completion_tokens = sumUsage(call.usage_summary, 'output');
      }
    }
  }
  return iter;
}

// Decompose an iteration's wall-clock into LM-generation / pure-code / sub-call time.
// All values come from data already in the log (see IterationTiming).
export function getIterationTiming(iter: RLMIteration): IterationTiming {
  let codeTotal = 0;
  let subCall = 0;
  for (const block of iter.code_blocks ?? []) {
    codeTotal += block.result?.execution_time ?? 0;
    for (const call of block.result?.rlm_calls ?? []) {
      subCall += call.execution_time ?? 0;
    }
  }
  const total = iter.iteration_time ?? codeTotal;
  const lmGen = Math.max(0, total - codeTotal);
  const codePure = Math.max(0, codeTotal - subCall);
  return { total, lmGen, codePure, subCall };
}

// Extract the context variable from code block locals
export function extractContextVariable(iterations: RLMIteration[]): string | null {
  for (const iter of iterations) {
    for (const block of iter.code_blocks) {
      if (block.result?.locals?.context) {
        const ctx = block.result.locals.context;
        if (typeof ctx === 'string') {
          return ctx;
        }
      }
    }
  }
  return null;
}

// Default config when metadata is not present (backwards compatibility)
function getDefaultConfig(): RLMConfigMetadata {
  return {
    root_model: null,
    max_depth: null,
    max_iterations: null,
    backend: null,
    backend_kwargs: null,
    environment_type: null,
    environment_kwargs: null,
    other_backends: null,
    parse_seconds: null,
  };
}

export interface ParsedJSONL {
  iterations: RLMIteration[];
  config: RLMConfigMetadata;
}

export function parseJSONL(content: string): ParsedJSONL {
  const lines = content.trim().split('\n').filter(line => line.trim());
  const iterations: RLMIteration[] = [];
  let config: RLMConfigMetadata = getDefaultConfig();
  
  for (const line of lines) {
    try {
      const parsed = JSON.parse(line);
      
      // Check if this is a metadata entry
      if (parsed.type === 'metadata') {
        config = {
          root_model: parsed.root_model ?? null,
          max_depth: parsed.max_depth ?? null,
          max_iterations: parsed.max_iterations ?? null,
          backend: parsed.backend ?? null,
          backend_kwargs: parsed.backend_kwargs ?? null,
          environment_type: parsed.environment_type ?? null,
          environment_kwargs: parsed.environment_kwargs ?? null,
          other_backends: parsed.other_backends ?? null,
          parse_seconds: parsed.parse_seconds ?? null,
        };
      } else {
        // This is an iteration entry
        iterations.push(normalizeIteration(parsed as RLMIteration));
      }
    } catch (e) {
      console.error('Failed to parse line:', line, e);
    }
  }
  
  return { iterations, config };
}

export function extractContextQuestion(iterations: RLMIteration[]): string {
  if (iterations.length === 0) return 'No context found';
  
  const firstIteration = iterations[0];
  const prompt = firstIteration.prompt;
  
  // Look for user message that contains the actual question
  for (const msg of prompt) {
    if (msg.role === 'user' && msg.content) {
      // Try to extract quoted query
      const queryMatch = msg.content.match(/original query: "([^"]+)"/);
      if (queryMatch) {
        return queryMatch[1];
      }
      
      // Check if it contains the actual query pattern
      if (msg.content.includes('answer the prompt')) {
        continue;
      }
      
      // Take first substantial user message
      if (msg.content.length > 50 && msg.content.length < 500) {
        return msg.content.slice(0, 200) + (msg.content.length > 200 ? '...' : '');
      }
    }
  }
  
  // Fallback: look in system prompt for context info
  const systemMsg = prompt.find(m => m.role === 'system');
  if (systemMsg?.content) {
    const contextMatch = systemMsg.content.match(/context variable.*?:(.*?)(?:\n|$)/i);
    if (contextMatch) {
      return contextMatch[1].trim().slice(0, 200);
    }
  }
  
  // Check code block output for actual context
  for (const iter of iterations) {
    for (const block of iter.code_blocks) {
      if (block.result?.locals?.context) {
        const ctx = block.result.locals.context;
        if (typeof ctx === 'string' && ctx.length < 500) {
          return ctx;
        }
      }
    }
  }
  
  return 'Context available in REPL environment';
}

export function computeMetadata(iterations: RLMIteration[]): LogMetadata {
  let totalCodeBlocks = 0;
  let totalSubLMCalls = 0;
  let totalExecutionTime = 0;
  let hasErrors = false;
  let finalAnswer: string | null = null;
  
  for (const iter of iterations) {
    totalCodeBlocks += iter.code_blocks.length;
    
    // Use iteration_time if available, otherwise sum code block times
    if (iter.iteration_time != null) {
      totalExecutionTime += iter.iteration_time;
    } else {
      for (const block of iter.code_blocks) {
        if (block.result) {
          totalExecutionTime += block.result.execution_time || 0;
        }
      }
    }
    
    for (const block of iter.code_blocks) {
      if (block.result) {
        if (block.result.stderr) {
          hasErrors = true;
        }
        if (block.result.rlm_calls) {
          totalSubLMCalls += block.result.rlm_calls.length;
        }
      }
    }
    
    if (iter.final_answer) {
      finalAnswer = extractFinalAnswer(iter.final_answer);
    }
  }
  
  return {
    totalIterations: iterations.length,
    totalCodeBlocks,
    totalSubLMCalls,
    contextQuestion: extractContextQuestion(iterations),
    finalAnswer,
    totalExecutionTime,
    hasErrors,
  };
}

export function parseLogFile(fileName: string, content: string): RLMLogFile {
  const { iterations, config } = parseJSONL(content);
  const metadata = computeMetadata(iterations);
  
  return {
    fileName,
    filePath: fileName,
    iterations,
    metadata,
    config,
  };
}

