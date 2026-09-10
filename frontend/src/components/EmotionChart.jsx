import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';
import './EmotionChart.css';

const EMOTION_COLORS = {
  joy: '#F59E0B',
  sadness: '#3B82F6',
  anger: '#EF4444',
  fear: '#8B5CF6',
  surprise: '#EC4899',
  disgust: '#10B981',
  neutral: '#94A3B8',
};

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    const color = EMOTION_COLORS[data.emotion.toLowerCase()] || '#267F4B';
    return (
      <div className="chart-tooltip">
        <span className="tooltip-color-indicator" style={{ background: color }}></span>
        <span className="chart-tooltip-name">{data.emotion}</span>
        <span className="chart-tooltip-value">{(data.score * 100).toFixed(1)}%</span>
      </div>
    );
  }
  return null;
};

export default function EmotionChart({ emotions, chartType = 'radar' }) {
  if (!emotions) return null;

  const data = Object.entries(emotions).map(([emotion, score]) => ({
    emotion: emotion.charAt(0).toUpperCase() + emotion.slice(1),
    score,
    fill: EMOTION_COLORS[emotion.toLowerCase()] || '#267F4B',
  }));

  return (
    <div className="emotion-chart card animate-fade-in-up" id="emotion-chart">
      <div className="chart-card-header">
        <div className="chart-title-group">
          <svg className="chart-title-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#267F4B" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20" />
            <path d="M2 12h20" />
          </svg>
          <h3 className="chart-title">Emotion Radar</h3>
        </div>
      </div>

      <div className="chart-wrapper">
        {chartType === 'radar' ? (
          <ResponsiveContainer width="100%" height={325}>
            <RadarChart cx="50%" cy="52%" outerRadius="75%" data={data}>
              <PolarGrid stroke="rgba(38, 127, 75, 0.14)" />
              <PolarAngleAxis
                dataKey="emotion"
                tick={{ fill: '#334155', fontSize: 12, fontWeight: 600 }}
              />
              <PolarRadiusAxis
                angle={90}
                domain={[0, 1]}
                tick={{ fill: '#94A3B8', fontSize: 10 }}
              />
              <Radar
                name="Emotion"
                dataKey="score"
                stroke="#267F4B"
                fill="#267F4B"
                fillOpacity={0.2}
                strokeWidth={2.5}
                dot={{ fill: '#267F4B', stroke: '#FFFFFF', strokeWidth: 2, r: 4.5 }}
                activeDot={{ fill: '#1E6B3E', stroke: '#FFFFFF', strokeWidth: 2.5, r: 6 }}
              />
              <Tooltip content={<CustomTooltip />} />
            </RadarChart>
          </ResponsiveContainer>
        ) : (
          <ResponsiveContainer width="100%" height={325}>
            <BarChart data={data} layout="vertical" margin={{ left: 10, right: 30 }}>
              <XAxis type="number" domain={[0, 1]} tick={{ fill: '#64748B', fontSize: 11 }} />
              <YAxis
                type="category"
                dataKey="emotion"
                width={80}
                tick={{ fill: '#334155', fontSize: 12, fontWeight: 500 }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="score" radius={[0, 6, 6, 0]} barSize={20}>
                {data.map((entry, index) => (
                  <Cell key={index} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}

