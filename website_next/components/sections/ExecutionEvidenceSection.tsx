'use client';

import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle2, Clock3, FileCheck2 } from 'lucide-react';
import type { ValidationEvidence } from '@/lib/validation-evidence';

interface ExecutionEvidenceSectionProps {
  evidence?: ValidationEvidence | null;
}

type ActionStatus = 'in-progress' | 'queued' | 'complete';

const immediateActions: Array<{
  id: string;
  title: string;
  status: ActionStatus;
  detail: string;
}> = [
  {
    id: 'Action 4',
    title: 'Cycle Delta Quantification',
    status: 'in-progress',
    detail: 'Measure and publish baseline vs imprint cycle deltas across workload matrix.',
  },
  {
    id: 'Action 5',
    title: 'Prefetch Overlap Race Gate',
    status: 'in-progress',
    detail: 'Close race-condition proof with deterministic overlap stress and pass/fail gate.',
  },
  {
    id: 'Action 9',
    title: 'Judge Evidence Map Refresh',
    status: 'queued',
    detail: 'Propagate latest full-run anchors to all judge-facing claim surfaces.',
  },
];

function formatUtc(value: string | undefined): string {
  if (!value) {
    return 'Unavailable';
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toISOString().replace('T', ' ').replace('Z', ' UTC');
}

function getActionChip(status: ActionStatus): string {
  if (status === 'complete') {
    return 'text-oxide-green border-oxide-green/40 bg-oxide-green/10';
  }

  if (status === 'in-progress') {
    return 'text-neon-cyan border-neon-cyan/40 bg-neon-cyan/10';
  }

  return 'text-metal-silver border-metal-silver/40 bg-metal-silver/10';
}

function getStageIcon(status: string) {
  if (status === 'PASS') {
    return <CheckCircle2 className="h-4 w-4 text-oxide-green" />;
  }

  if (status === 'FAIL') {
    return <AlertTriangle className="h-4 w-4 text-error-red" />;
  }

  return <Clock3 className="h-4 w-4 text-metal-silver" />;
}

export default function ExecutionEvidenceSection({ evidence }: ExecutionEvidenceSectionProps) {
  const runId = evidence?.runId ?? 'Unavailable';
  const generatedUtc = formatUtc(evidence?.generatedUtc);
  const commitHash = evidence?.commitHash ?? 'Unavailable';
  const modeLabel = evidence ? (evidence.quickMode ? 'Quick checkpoint' : 'Full matrix') : 'Mode unavailable';

  const stageTotals = evidence?.stageTotals ?? {
    pass: 0,
    fail: 0,
    total: 0,
  };

  const stageRows = Object.entries(evidence?.stages ?? {}).sort(([left], [right]) =>
    left.localeCompare(right),
  );

  return (
    <section id="execution-evidence" className="relative py-24 border-t border-white/5 bg-[#060F1C]">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.45 }}
          className="mb-10"
        >
          <div className="inline-flex items-center gap-2 px-4 py-1 rounded-full border border-neon-cyan/30 bg-neon-cyan/10 text-neon-cyan text-xs font-mono uppercase tracking-widest">
            <FileCheck2 className="h-3.5 w-3.5" />
            Wave 10 Execution Evidence
          </div>
          <h2 className="mt-4 font-space text-3xl md:text-4xl text-white font-bold">
            Chain Status And Immediate Actions
          </h2>
          <p className="mt-3 text-metal-silver/90 max-w-3xl font-inter text-sm md:text-base">
            Latest orchestration anchor, validation stage outcomes, and elite-plan immediate actions.
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <motion.article
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.35 }}
            className="glass-surface rounded-2xl p-6 border border-white/10 lg:col-span-2"
          >
            <h3 className="font-space text-xl text-white font-bold mb-4">Latest Anchor</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 font-mono text-sm">
              <div>
                <p className="text-metal-silver/70 uppercase tracking-wide text-xs">Run ID</p>
                <p className="text-white mt-1 break-all" data-testid="execution-run-id">
                  {runId}
                </p>
              </div>
              <div>
                <p className="text-metal-silver/70 uppercase tracking-wide text-xs">Commit</p>
                <p className="text-white mt-1 break-all" data-testid="execution-commit-hash">
                  {commitHash}
                </p>
              </div>
              <div>
                <p className="text-metal-silver/70 uppercase tracking-wide text-xs">Validation Mode</p>
                <p className="text-white mt-1" data-testid="execution-validation-mode">
                  {modeLabel}
                </p>
              </div>
              <div>
                <p className="text-metal-silver/70 uppercase tracking-wide text-xs">Generated UTC</p>
                <p className="text-white mt-1">{generatedUtc}</p>
              </div>
              <div>
                <p className="text-metal-silver/70 uppercase tracking-wide text-xs">Workloads</p>
                <p className="text-white mt-1">{evidence?.workloadCount ?? 'Unavailable'}</p>
              </div>
              <div>
                <p className="text-metal-silver/70 uppercase tracking-wide text-xs">Measured Runs</p>
                <p className="text-white mt-1">{evidence?.measuredRuns ?? 'Unavailable'}</p>
              </div>
            </div>
          </motion.article>

          <motion.article
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.4, delay: 0.05 }}
            className="glass-surface rounded-2xl p-6 border border-white/10"
          >
            <h3 className="font-space text-xl text-white font-bold mb-4">Gate Summary</h3>
            <div className="space-y-3 font-mono text-sm">
              <div className="flex justify-between">
                <span className="text-metal-silver/75">Stages Total</span>
                <span className="text-white" data-testid="execution-stage-total">
                  {stageTotals.total}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-metal-silver/75">Pass</span>
                <span className="text-oxide-green" data-testid="execution-stage-pass">
                  {stageTotals.pass}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-metal-silver/75">Fail</span>
                <span className="text-error-red" data-testid="execution-stage-fail">
                  {stageTotals.fail}
                </span>
              </div>
            </div>
          </motion.article>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
          <motion.article
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.45 }}
            className="glass-surface rounded-2xl p-6 border border-white/10 lg:col-span-2"
          >
            <h3 className="font-space text-xl text-white font-bold mb-4">Stage Breakdown</h3>
            {stageRows.length === 0 ? (
              <p className="text-metal-silver/80 text-sm">Validation stage data is currently unavailable.</p>
            ) : (
              <div className="space-y-2">
                {stageRows.map(([name, stage]) => (
                  <div
                    key={name}
                    className="flex items-center justify-between rounded-xl border border-white/10 bg-black/20 px-3 py-2"
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      {getStageIcon(stage.status)}
                      <span className="font-mono text-xs md:text-sm text-white truncate">{name}</span>
                    </div>
                    <div className="font-mono text-xs text-metal-silver/85 ml-3 whitespace-nowrap">
                      {stage.durationSeconds === null
                        ? stage.status
                        : `${stage.status} | ${stage.durationSeconds.toFixed(2)}s`}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </motion.article>

          <motion.article
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.05 }}
            className="glass-surface rounded-2xl p-6 border border-white/10"
          >
            <h3 className="font-space text-xl text-white font-bold mb-4">Immediate Actions</h3>
            <div className="space-y-3">
              {immediateActions.map((action) => (
                <div key={action.id} className="rounded-xl border border-white/10 bg-black/20 p-3">
                  <div className="flex items-center justify-between gap-3">
                    <span className="font-mono text-xs text-metal-silver uppercase tracking-wide">
                      {action.id}
                    </span>
                    <span
                      className={`font-mono text-[10px] uppercase tracking-wider px-2 py-1 rounded border ${getActionChip(action.status)}`}
                    >
                      {action.status.replace('-', ' ')}
                    </span>
                  </div>
                  <p className="mt-2 text-white font-space text-sm">{action.title}</p>
                  <p className="mt-1 text-metal-silver/80 text-xs leading-relaxed">{action.detail}</p>
                </div>
              ))}
            </div>
          </motion.article>
        </div>
      </div>
    </section>
  );
}
