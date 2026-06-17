/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#17201c",
        moss: "#315a49",
        linen: "#f7f4ec",
        wheat: "#ece2c7",
        coral: "#d8664d",
        cyan: "#0f8b8d"
      },
      boxShadow: {
        panel: "0 12px 35px rgba(23, 32, 28, 0.08)"
      }
    }
  },
  plugins: []
};
