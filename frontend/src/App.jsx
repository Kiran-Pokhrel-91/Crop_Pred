import { NavLink, Route, Routes } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import MapVisualization from './pages/MapVisualization'
import Prediction from './pages/Prediction'

const navigation = [
  { to: '/', label: 'Dashboard', end: true },
  { to: '/predict', label: 'Krishi Intelligence' },
  { to: '/map', label: 'KrishiMap' },
]

function navClass({ isActive }) {
  return [
    'rounded-md px-3 py-2 text-sm font-medium transition-colors',
    isActive
      ? 'bg-crop-600 text-white'
      : 'text-slate-600 hover:bg-crop-50 hover:text-crop-700',
  ].join(' ')
}

export default function App() {
  return (
    <div className="flex min-h-screen flex-col bg-slate-50">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-crop-600">
              KrishiDrishti
            </p>
            <h1 className="text-lg font-semibold text-slate-900">
              Nepal&apos;s Agricultural Intelligence Platform
            </h1>
            <p className="text-sm text-slate-500">From agricultural guesswork to data-driven decisions.</p>
            <p className="mt-0.5 text-sm font-medium text-slate-700">
              Helping 77 districts predict food shortages and optimize crop yields amidst climate change.
            </p>
          </div>
          <nav className="flex flex-wrap gap-1" aria-label="Primary">
            {navigation.map((item) => (
              <NavLink key={item.to} to={item.to} end={item.end} className={navClass}>
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-8">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/predict" element={<Prediction />} />
          <Route path="/map" element={<MapVisualization />} />
          <Route path="*" element={<p className="text-slate-600">Page not found.</p>} />
        </Routes>
      </main>

      <footer className="border-t border-slate-200 bg-white">
        <div className="mx-auto max-w-6xl px-4 py-4 text-xs text-slate-500">
          Pilot intelligence platform combining KrishiMap, KrishiCalendar, and AI-assisted insights. Estimates are indicative and should be verified with local agricultural expertise.
        </div>
      </footer>
    </div>
  )
}
