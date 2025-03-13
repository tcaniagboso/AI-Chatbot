/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        chatbg: '#343541',
        messagebg: '#444654',
        inputbg: '#40414f',
      },
    },
  },
  plugins: [],
}
