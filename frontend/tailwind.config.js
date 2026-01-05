/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
        "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
        "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
    ],
    darkMode: 'class',
    theme: {
        extend: {
            colors: {
                maroon: {
                    DEFAULT: "var(--color-maroon)",
                    dark: "var(--color-maroon-dark)",
                },
                gold: {
                    DEFAULT: "var(--color-gold)",
                    light: "var(--color-gold-light)",
                },
                cream: "var(--color-cream)",
                "dark-gray": "var(--color-dark-gray)",
                "light-gray": "var(--color-light-gray)",
            },
        },
    },
    plugins: [],
};
