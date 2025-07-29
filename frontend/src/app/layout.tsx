import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'AI Portfolio Analyzer | Smart Investment Insights',
  description: 'Transform your portfolio screenshots into actionable AI-powered investment insights with real-time market data and personalized recommendations.',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen overflow-x-hidden">
        {/* Animated background elements */}
        <div className="fixed inset-0 overflow-hidden pointer-events-none">
          <div className="absolute -top-40 -right-40 w-96 h-96 bg-gradient-to-br from-blue-400/20 to-indigo-600/20 rounded-full blur-3xl animate-float"></div>
          <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-gradient-to-tr from-purple-400/20 to-pink-600/20 rounded-full blur-3xl animate-float" style={{animationDelay: '3s'}}></div>
          <div className="absolute top-1/3 right-1/4 w-64 h-64 bg-gradient-to-r from-cyan-400/10 to-blue-600/10 rounded-full blur-2xl animate-pulse-slow"></div>
          <div className="absolute bottom-1/3 left-1/4 w-48 h-48 bg-gradient-to-r from-indigo-400/15 to-purple-600/15 rounded-full blur-2xl animate-float" style={{animationDelay: '4s'}}></div>
        </div>

        {/* Premium header */}
        <header className="floating-header sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-6 lg:px-8">
            <div className="flex justify-between items-center py-6">
              <div className="flex items-center space-x-4">
                <div className="relative">
                  <div className="p-3 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-2xl shadow-xl">
                    <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                  </div>
                  <div className="pulse-ring"></div>
                </div>
                <div>
                  <h1 className="text-3xl font-bold text-gradient">
                    AI Portfolio Analyzer
                  </h1>
                  <p className="text-sm text-gray-600 font-medium">Powered by Advanced Machine Learning</p>
                </div>
              </div>
              <div className="flex items-center space-x-4">
                <div className="hidden md:flex items-center space-x-3 px-4 py-2 bg-gradient-to-r from-green-100 to-emerald-100 rounded-full border border-green-200 shadow-lg">
                  <div className="relative">
                    <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                    <div className="absolute inset-0 w-3 h-3 bg-green-400 rounded-full animate-ping"></div>
                  </div>
                  <span className="text-sm text-green-700 font-bold">Live Market Data</span>
                </div>
                <div className="px-4 py-2 bg-gradient-to-r from-blue-100 to-indigo-100 rounded-full border border-blue-200 shadow-md">
                  <span className="text-sm font-bold text-blue-700">v2.0 Pro</span>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Main content with margin for header */}
        <main className="relative max-w-7xl mx-auto px-6 lg:px-8 py-12">
          {children}
        </main>

        {/* Premium footer */}
        <footer className="relative mt-24 py-12 text-center">
          <div className="max-w-7xl mx-auto px-6">
            <div className="glass-effect rounded-2xl p-8 shadow-glow">
              <div className="flex flex-wrap justify-center items-center space-x-8 text-gray-600">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                  <span className="font-medium">Real-time Analysis</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="font-medium">Live Market Data</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse"></div>
                  <span className="font-medium">AI-Powered Insights</span>
                </div>
              </div>
            </div>
          </div>
        </footer>
      </body>
    </html>
  )
}