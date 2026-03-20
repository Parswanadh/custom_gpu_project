'use client';

import { XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts';

const data = [
  { layer: 'Layer 1', dynamic: 28, imprint: 9 },
  { layer: 'Layer 2', dynamic: 56, imprint: 18 },
  { layer: 'Layer 3', dynamic: 84, imprint: 27 },
  { layer: 'Layer 4', dynamic: 112, imprint: 36 },
  { layer: 'Layer 5', dynamic: 140, imprint: 45 },
  { layer: 'Layer 6', dynamic: 168, imprint: 54 },
  { layer: 'Layer 7', dynamic: 196, imprint: 63 },
  { layer: 'Layer 8', dynamic: 224, imprint: 72 },
  { layer: 'Layer 9', dynamic: 252, imprint: 81 },
  { layer: 'Layer 10', dynamic: 280, imprint: 90 },
  { layer: 'Layer 11', dynamic: 308, imprint: 99 },
  { layer: 'Layer 12', dynamic: 341, imprint: 112 },
];

export default function RunsChart() {
  const CustomTooltip = ({ active, payload, label }: { active?: boolean; payload?: { value: number }[]; label?: string; }) => {
    if (active && payload && payload.length) {
      return (
        <div className="glass-surface p-4 rounded-lg border border-white/10 shadow-xl">
          <p className="font-space font-bold text-white mb-2">{label}</p>
          <div className="space-y-1">
            <p className="font-mono text-xs text-die-copper">
              Dynamic Pipeline: {payload[0].value} cycles
            </p>
            <p className="font-mono text-xs text-neon-cyan">
              Imprinted Core: {payload[1].value} cycles
            </p>
          </div>
          <p className="mt-2 pt-2 border-t border-white/10 font-mono text-[10px] text-metal-silver uppercase tracking-wider">
            Icarus Verilog vvp Log
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full h-[400px]">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex gap-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-die-copper rounded-sm" />
            <span className="text-[10px] font-mono text-metal-silver uppercase tracking-wider">Dynamic (341 cy)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-neon-cyan rounded-sm" />
            <span className="text-[10px] font-mono text-metal-silver uppercase tracking-wider">Imprint (112 cy)</span>
          </div>
        </div>
        <span className="text-[10px] font-mono text-oxide-green uppercase tracking-wider">Simulation Verified ✓</span>
      </div>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={data}
          margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
        >
          <defs>
            <linearGradient id="colorDynamic" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#B87333" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#B87333" stopOpacity={0}/>
            </linearGradient>
            <linearGradient id="colorImprint" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#00F5FF" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#00F5FF" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
          <XAxis 
            dataKey="layer" 
            stroke="#8892A4" 
            tick={{ fill: '#8892A4', fontSize: 10, fontFamily: 'var(--font-jetbrains-mono)' }} 
            axisLine={false}
            tickLine={false}
            dy={10}
            interval={1}
          />
          <YAxis 
            stroke="#8892A4" 
            tick={{ fill: '#8892A4', fontSize: 10, fontFamily: 'var(--font-jetbrains-mono)' }}
            axisLine={false}
            tickLine={false}
            tickFormatter={(val) => `${val} cy`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Area 
            type="monotone" 
            dataKey="dynamic" 
            stroke="#B87333" 
            strokeWidth={2}
            fillOpacity={1} 
            fill="url(#colorDynamic)" 
            animationDuration={2500}
          />
          <Area 
            type="monotone" 
            dataKey="imprint" 
            stroke="#00F5FF" 
            strokeWidth={2}
            fillOpacity={1} 
            fill="url(#colorImprint)" 
            animationDuration={2000}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
