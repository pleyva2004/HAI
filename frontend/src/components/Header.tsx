import Image from 'next/image';
import Link from 'next/link';

export default function Header() {
    return (
        <header className="fixed top-0 left-0 right-0 z-50 h-16 transition-all duration-300">
            {/* Glass Background Layer */}
            <div className="absolute inset-0 bg-white/60 dark:bg-black/40 backdrop-blur-xl border-b border-white/20 dark:border-white/10 shadow-[0_4px_30px_rgba(0,0,0,0.03)] supports-[backdrop-filter]:bg-white/60 dark:supports-[backdrop-filter]:bg-black/40" />

            <div className="container relative h-full flex items-center justify-between">
                <Link href="/" className="flex items-center gap-3 group">
                    <div className="relative w-8 h-8 transition-transform duration-300 group-hover:scale-110 group-active:scale-95">
                        <Image
                            src="/logo.png"
                            alt="Hilltop IA"
                            fill
                            className="object-contain drop-shadow-sm"
                            priority
                            suppressHydrationWarning
                        />
                    </div>
                    <div>
                        <h1 className="text-lg font-semibold text-[#1D1D1F] dark:text-white tracking-tight group-hover:opacity-80 transition-opacity">
                            Hilltop AI
                        </h1>
                    </div>
                </Link>

                <div className="flex items-center">
                    <button
                        onClick={() => {
                            if (document.documentElement.classList.contains('dark')) {
                                document.documentElement.classList.remove('dark');
                                localStorage.theme = 'light';
                            } else {
                                document.documentElement.classList.add('dark');
                                localStorage.theme = 'dark';
                            }
                        }}
                        className="p-2 rounded-full bg-black/5 hover:bg-black/10 dark:bg-white/10 dark:hover:bg-white/20 transition-all text-gray-600 dark:text-gray-300 backdrop-blur-md border border-transparent dark:border-white/10"
                        title="Toggle theme"
                    >
                        {/* Sun Icon (shown in dark mode) */}
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="hidden dark:block">
                            <circle cx="12" cy="12" r="5"></circle>
                            <line x1="12" y1="1" x2="12" y2="3"></line>
                            <line x1="12" y1="21" x2="12" y2="23"></line>
                            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                            <line x1="1" y1="12" x2="3" y2="12"></line>
                            <line x1="21" y1="12" x2="23" y2="12"></line>
                            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
                        </svg>

                        {/* Moon Icon (shown in light mode) */}
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="block dark:hidden">
                            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
                        </svg>
                    </button>
                    <script dangerouslySetInnerHTML={{
                        __html: `
                            try {
                                if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
                                    document.documentElement.classList.add('dark')
                                } else {
                                    document.documentElement.classList.remove('dark')
                                }
                            } catch (_) {}
                        `
                    }} />
                </div>
            </div>
        </header>
    );
}
