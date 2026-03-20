import type { Metadata } from 'next';
import { Space_Grotesk, Inter, JetBrains_Mono, DM_Sans } from 'next/font/google';
import './globals.css';

const spaceGrotesk = Space_Grotesk({ subsets: ['latin'], variable: '--font-space-grotesk' });
const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });
const jetbrainsMono = JetBrains_Mono({ subsets: ['latin'], variable: '--font-jetbrains-mono' });
const dmSans = DM_Sans({ subsets: ['latin'], variable: '--font-dm-sans' });

export const metadata: Metadata = {
  title: 'BitbyBit | Custom Silicon Architecture for Transformers',
  description: 'A ground-up, cycle-accurate Verilog-2005 architecture explicitly engineered for Transformer inference. Zero-multiplier ternary logic, Silicon Imprinting, and 112-cycle inference.',
  keywords: ['Custom Silicon', 'Verilog', 'Transformer Hardware', 'Inference Accelerator', 'BitNet', 'Gemma 3', 'FPGA', 'RTL'],
  authors: [{ name: 'BitbyBit Team' }],
  openGraph: {
    title: 'BitbyBit | Custom Silicon Architecture',
    description: 'Ground-up Verilog-2005 architecture for Transformer inference.',
    url: 'https://bitbybit-sandy.vercel.app',
    siteName: 'BitbyBit Custom Silicon',
    locale: 'en_US',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'BitbyBit | Custom Silicon Architecture',
    description: 'Ground-up Verilog-2005 architecture for Transformer inference.',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark scroll-smooth">
      <body
        className={`${spaceGrotesk.variable} ${inter.variable} ${jetbrainsMono.variable} ${dmSans.variable} font-sans antialiased bg-silicon-black text-metal-silver`}
      >
        {children}
      </body>
    </html>
  );
}
