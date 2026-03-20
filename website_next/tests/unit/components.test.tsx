import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import React from 'react';

// Shared
import CountUp from '@/components/shared/CountUp';

// Sections
import NavigationBar from '@/components/navigation/NavigationBar';
import WorkflowBackground from '@/components/shared/WorkflowBackground';
import HeroSection from '@/components/sections/HeroSection';
import ProblemSection from '@/components/sections/ProblemSection';
import PipelineSection from '@/components/sections/PipelineSection';
import ArchitectureSection from '@/components/sections/ArchitectureSection';
import MetricsDashboardSection from '@/components/sections/MetricsDashboardSection';
import ComparisonSection from '@/components/sections/ComparisonSection';
import RoadmapSection from '@/components/sections/RoadmapSection';
import FooterSection from '@/components/sections/FooterSection';

describe('Auto-GIT Component Mount Tests', () => {
  it('CountUp renders initial value correctly in JSDOM (requires intersection to animating to target)', () => {
    render(<CountUp end={420} duration={0.1} />);
    expect(screen.getByText('0')).toBeInTheDocument();
  });

  it('NavigationBar mounts and renders core routing links', () => {
    render(<NavigationBar />);
    expect(screen.getByText('Architecture')).toBeInTheDocument();
    expect(screen.getByText('Evolution')).toBeInTheDocument();
  });

  it('WorkflowBackground renders SVG pattern correctly', () => {
    const { container } = render(<WorkflowBackground color="cyan" density="medium" />);
    expect(container.querySelector('svg')).toBeInTheDocument();
  });

  // Section mount smoke tests
  const sections = [
    { name: 'HeroSection', component: <HeroSection /> },
    { name: 'PipelineSection', component: <PipelineSection /> },
    { name: 'ArchitectureSection', component: <ArchitectureSection /> },
    { name: 'MetricsDashboardSection', component: <MetricsDashboardSection /> },
    { name: 'ComparisonSection', component: <ComparisonSection /> },
    { name: 'RoadmapSection', component: <RoadmapSection /> },
    { name: 'FooterSection', component: <FooterSection /> }
  ];

  sections.forEach(({ name, component }) => {
    it(`mounts ${name} without crashing`, () => {
      const { container } = render(component);
      expect(container).toBeDefined();
    });
  });
});
