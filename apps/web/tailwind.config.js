/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: '#FFFFFF',
        surface: {
          DEFAULT: '#F8FAFC',
          subtle: '#F1F5F9',
          card: '#FFFFFF',
        },
        ink: {
          DEFAULT: '#0F172A',
          body: '#475569',
          muted: '#64748B',
          subtle: '#94A3B8',
        },
        border: {
          subtle: '#E2E8F0',
          strong: '#CBD5E1',
        },
        cobalt: {
          DEFAULT: '#1D4ED8',
          hover: '#1E40AF',
          surface: '#EFF6FF',
        },
        status: {
          mastered: '#059669',
          'mastered-bg': '#ECFDF5',
          'mastered-border': '#A7F3D0',
          active: '#0284C7',
          'active-bg': '#F0F9FF',
          'active-border': '#BAE6FD',
          fading: '#D97706',
          'fading-bg': '#FFFBEB',
          'fading-border': '#FDE68A',
          risk: '#DC2626',
          'risk-bg': '#FEF2F2',
          'risk-border': '#FECACA',
        },
      },
      borderRadius: {
        lg: '8px',
        md: '6px',
        sm: '4px',
      },
    },
  },
  plugins: [],
};
