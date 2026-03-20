import NavigationBar from '@/components/navigation/NavigationBar';
import HeroSection from '@/components/sections/HeroSection';
import ProblemSection from '@/components/sections/ProblemSection';
import ArchitectureSection from '@/components/sections/ArchitectureSection';
import PipelineSection from '@/components/sections/PipelineSection';
import MetricsDashboardSection from '@/components/sections/MetricsDashboardSection';
import ComparisonSection from '@/components/sections/ComparisonSection';
import LiveDemoSection from '@/components/sections/LiveDemoSection';
import RoadmapSection from '@/components/sections/RoadmapSection';
import FooterSection from '@/components/sections/FooterSection';

export default function Home() {
  return (
    <main className="min-h-screen bg-silicon-black text-metal-silver w-full mx-auto font-sans selection:bg-neon-cyan/30 selection:text-white">
      <NavigationBar />
      <HeroSection />
      <ProblemSection />
      <ArchitectureSection />
      <PipelineSection />
      <MetricsDashboardSection />
      <ComparisonSection />
      <LiveDemoSection />
      <RoadmapSection />
      <FooterSection />
    </main>
  );
}
