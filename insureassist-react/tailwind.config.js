/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Space Grotesk"', 'sans-serif'],
        sans: ['"Inter"', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'monospace'],
      },
      colors: {
        ink: '#0d151d',
        panel: '#131e28',
        'panel-2': '#0f1922',
        gold: { DEFAULT: '#c9a24b', soft: '#3a3222' },
        teal: { DEFAULT: '#57c6bd', soft: '#173330' },
      },
      boxShadow: {
        glow: '0 20px 40px -28px rgba(0,0,0,0.6)',
        'glow-lg': '0 25px 60px -20px rgba(0,0,0,0.5)',
      },
      keyframes: {
        'pulse-ring': {
          '0%': { transform: 'scale(0.6)', opacity: '0.7' },
          '100%': { transform: 'scale(2.1)', opacity: '0' },
        },
        sparkle: {
          '0%, 100%': { opacity: '1', transform: 'scale(1)' },
          '50%': { opacity: '0.4', transform: 'scale(1.3)' },
        },
      },
      animation: {
        'pulse-ring': 'pulse-ring 1.8s ease-out infinite',
        sparkle: 'sparkle 1.4s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
