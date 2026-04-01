import fs from 'node:fs';
import path from 'node:path';
import { execSync } from 'node:child_process';

export interface ValidationStageEvidence {
  status: string;
  durationSeconds: number | null;
}

export interface ValidationEvidence {
  runId: string;
  generatedUtc: string;
  quickMode: boolean;
  workloadCount: number | null;
  measuredRuns: number | null;
  commitHash: string | null;
  stageTotals: {
    pass: number;
    fail: number;
    total: number;
  };
  stages: Record<string, ValidationStageEvidence>;
}

type RawValidationStage = {
  duration_seconds?: number;
  status?: string;
};

type RawValidationManifest = {
  run_id?: string;
  generated_utc?: string;
  quick_mode?: boolean;
  benchmark_meta?: {
    workload_count_effective?: number;
    measured_runs?: number;
  };
  stages?: Record<string, RawValidationStage>;
};

function readJsonFile(filePath: string): unknown {
  const rawBuffer = fs.readFileSync(filePath);
  const utf8Text = rawBuffer.toString('utf8').replace(/^\uFEFF/, '');

  try {
    return JSON.parse(utf8Text);
  } catch {
    const utf16Text = rawBuffer.toString('utf16le').replace(/^\uFEFF/, '');
    return JSON.parse(utf16Text);
  }
}

function toFiniteNumber(value: unknown): number | null {
  if (typeof value !== 'number') {
    return null;
  }

  return Number.isFinite(value) ? value : null;
}

function resolveCommitHash(projectRoot: string): string | null {
  try {
    return execSync('git rev-parse --short HEAD', {
      cwd: projectRoot,
      stdio: ['ignore', 'pipe', 'ignore'],
      encoding: 'utf-8',
    }).trim();
  } catch {
    return null;
  }
}

function resolveManifestPath(): string | null {
  const configured = process.env.BITBYBIT_VALIDATION_MANIFEST;
  const candidates = [
    configured,
    path.resolve(process.cwd(), '..', 'sim', 'validation_manifest_latest.json'),
    path.resolve(process.cwd(), 'sim', 'validation_manifest_latest.json'),
  ].filter((value): value is string => Boolean(value));

  return Array.from(new Set(candidates)).find((candidate) => fs.existsSync(candidate)) ?? null;
}

export function getValidationEvidence(): ValidationEvidence | null {
  const manifestPath = resolveManifestPath();
  if (!manifestPath) {
    return null;
  }

  const raw = readJsonFile(manifestPath) as RawValidationManifest;
  if (!raw || typeof raw.run_id !== 'string') {
    return null;
  }

  const stageEntries = Object.entries(raw.stages ?? {}).map(([name, stage]) => {
    const status = typeof stage?.status === 'string' ? stage.status.toUpperCase() : 'UNKNOWN';
    return [
      name,
      {
        status,
        durationSeconds: toFiniteNumber(stage?.duration_seconds),
      },
    ] as const;
  });

  const pass = stageEntries.filter(([, stage]) => stage.status === 'PASS').length;
  const fail = stageEntries.filter(([, stage]) => stage.status === 'FAIL').length;

  const projectRoot = path.resolve(path.dirname(manifestPath), '..');

  return {
    runId: raw.run_id,
    generatedUtc:
      typeof raw.generated_utc === 'string' ? raw.generated_utc : new Date().toISOString(),
    quickMode: raw.quick_mode === true,
    workloadCount: toFiniteNumber(raw.benchmark_meta?.workload_count_effective),
    measuredRuns: toFiniteNumber(raw.benchmark_meta?.measured_runs),
    commitHash: resolveCommitHash(projectRoot),
    stageTotals: {
      pass,
      fail,
      total: stageEntries.length,
    },
    stages: Object.fromEntries(stageEntries),
  };
}
