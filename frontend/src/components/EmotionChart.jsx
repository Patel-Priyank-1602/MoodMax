import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie,
} from 'recharts';
import './EmotionChart.css';

const EMOTION_COLORS = {
  joy: '#fdcb6e',
  sadness: '#74b9ff',
  anger: '#ff7675',
  fear: '#a29bfe',
  surprise: '#fd79a8',
  disgust: '#00b894',
  neutral: '#636e72',
};

const EMOTION_EMOJIS = {
  joy: '😄', sadness: '😢', anger: '😠', fear: '😨',
  surprise: '😲', disgust: '🤢', neutral: '😐',
};

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="chart-tooltip glass-card">
        <span className="chart-tooltip-emoji">{EMOTION_EMOJIS[data.emotion] || ''}</span>
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
    fill: EMOTION_COLORS[emotion] || '#6c5ce7',
  }));

  return (
    <div className="emotion-chart glass-card" id="emotion-chart">
      <h3 className="chart-title">Emotion Radar</h3>

      <div className="chart-wrapper">
        {chartType === 'radar' ? (
          <ResponsiveContainer width="100%" height={320}>
            <RadarChart cx="50%" cy="50%" outerRadius="70%" data={data}>
              <PolarGrid stroke="rgba(108, 92, 231, 0.15)" />
              <PolarAngleAxis
                dataKey="emotion"
                tick={{ fill: '#9d9db5', fontSize: 12, fontWeight: 500 }}
              />
              <PolarRadiusAxis
                angle={90}
                domain={[0, 1]}
                tick={{ fill: '#5a5a7a', fontSize: 10 }}
              />
              <Radar
                name="Emotion"
                dataKey="score"
                stroke="#6c5ce7"
                fill="rgba(108, 92, 231, 0.25)"
                strokeWidth={2}
                dot={{ fill: '#a29bfe', r: 4 }}
              />
              <Tooltip content={<CustomTooltip />} />
            </RadarChart>
          </ResponsiveContainer>
        ) : (
          <ResponsiveContainer width="100%" height={320}>
            <BarChart data={data} layout="vertical" margin={{ left: 10, right: 30 }}>
              <XAxis type="number" domain={[0, 1]} tick={{ fill: '#5a5a7a', fontSize: 11 }} />
              <YAxis
                type="category"
                dataKey="emotion"
                width={80}
                tick={{ fill: '#9d9db5', fontSize: 12 }}
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
