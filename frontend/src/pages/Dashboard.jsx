import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { fetchCropSeasons, fetchCrops, fetchDistrictDashboard, fetchDistricts, getErrorMessage } from '../api'
import DistrictSelect from '../components/DistrictSelect'

function StatCard({ label, value, hint }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <p className="text-sm text-slate-500">{label}</p>
      <p className="mt-2 text-2xl font-semibold text-slate-900">{value}</p>
      {hint ? <p className="mt-1 text-xs text-slate-500">{hint}</p> : null}
    </div>
  )
}

function formatNum(value, digits = 1) {
  if (value == null || Number.isNaN(Number(value))) return '—'
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits })
}

function StatusPill({ label }) {
  const styles = {
    excellent: 'bg-green-100 text-green-800',
    good: 'bg-emerald-100 text-emerald-800',
    moderate: 'bg-yellow-100 text-yellow-800',
    poor: 'bg-orange-100 text-orange-800',
    stressed: 'bg-red-100 text-red-800',
    unknown: 'bg-slate-100 text-slate-700',
  }
  const key = (label || 'unknown').toLowerCase()
  return <span className={`inline-flex rounded-full px-2.5 py-1 text-xs font-medium capitalize ${styles[key] || styles.unknown}`}>{label || 'unknown'}</span>
}

export default function Dashboard() {
  const [loadingBase, setLoadingBase] = useState(true)
  const [loadingDash, setLoadingDash] = useState(false)
  const [error, setError] = useState('')
  const [dashError, setDashError] = useState('')
  const [crops, setCrops] = useState([])
  const [seasons, setSeasons] = useState([])
  const [districts, setDistricts] = useState([])
  const [selectedDistrict, setSelectedDistrict] = useState('Rupandehi')
  const [dashboard, setDashboard] = useState(null)

  useEffect(() => {
    let active = true
    async function loadBase() {
      setLoadingBase(true)
      setError('')
      try {
        const [cropsRes, seasonsRes, districtsRes] = await Promise.allSettled([
          fetchCrops(),
          fetchCropSeasons(),
          fetchDistricts(),
        ])
        if (!active) return
        if (cropsRes.status === 'fulfilled') setCrops(cropsRes.value.results || [])
        if (seasonsRes.status === 'fulfilled') setSeasons(seasonsRes.value.results || [])
        if (districtsRes.status === 'fulfilled') {
          const list = (districtsRes.value.results || []).slice().sort((a, b) => a.district.localeCompare(b.district))
          setDistricts(list)
          if (list.length) {
            const preferred = list.some((d) => d.district === 'Rupandehi') ? 'Rupandehi' : list[0].district
            setSelectedDistrict((prev) => (list.some((d) => d.district === prev) ? prev : preferred))
          }
        }
        const failures = [cropsRes, seasonsRes, districtsRes].filter((r) => r.status === 'rejected')
        if (failures.length === 3) setError(getErrorMessage(failures[0].reason))
      } finally {
        if (active) setLoadingBase(false)
      }
    }
    loadBase()
    return () => { active = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!selectedDistrict) return
    let active = true
    async function loadDashboard() {
      setLoadingDash(true)
      setDashError('')
      try {
        const res = await fetchDistrictDashboard(selectedDistrict)
        if (!active) return
        setDashboard(res.dashboard || null)
      } catch (err) {
        if (!active) return
        setDashboard(null)
        setDashError(getErrorMessage(err))
      } finally {
        if (active) setLoadingDash(false)
      }
    }
    loadDashboard()
    return () => { active = false }
  }, [selectedDistrict])

  const selectedInfo = useMemo(() => districts.find((d) => d.district === selectedDistrict), [districts, selectedDistrict])
  const env = dashboard?.environment_status || {}
  const currentCrops = dashboard?.current_cultivation || []

  return (
    <section className="space-y-6">
      <header className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h2 className="text-2xl font-semibold text-slate-900">KrishiDrishti District Dashboard</h2>
          <p className="mt-1 max-w-3xl text-slate-600">
            Helping 77 districts predict food shortages and optimize crop yields amidst climate change.
          </p>
          <p className="mt-1 max-w-3xl text-slate-600">
            View current cultivation records, crop production status, and environmental indicators by district.
          </p>
        </div>
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end">
          <label className="text-sm min-w-64">
            <span className="mb-1 block font-medium text-slate-700">District</span>
            <DistrictSelect
              districts={districts}
              value={selectedDistrict}
              onChange={setSelectedDistrict}
              placeholder="Search district or province..."
            />
          </label>
          <Link to="/predict" className="inline-flex items-center justify-center rounded-md bg-crop-600 px-4 py-2 text-sm font-medium text-white hover:bg-crop-700">
            Run crop intelligence
          </Link>
        </div>
      </header>

      {error ? <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">Failed to load base data: {error}</div> : null}
      {dashError ? <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">Failed to load district dashboard: {dashError}</div> : null}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Selected district" value={dashboard?.district || selectedDistrict || (loadingBase ? '…' : '—')} hint={dashboard?.province || selectedInfo?.province || '—'} />
        <StatCard label="Latest data year" value={loadingDash ? '…' : dashboard?.latest_year || '—'} hint="Most recent cultivation records available" />
        <StatCard label="Crops recorded" value={loadingDash ? '…' : dashboard?.summary?.crop_count ?? currentCrops.length ?? '—'} hint={`Grid cells: ${dashboard?.grid_cells ?? selectedInfo?.grid_cells ?? '—'}`} />
        <StatCard label="Total production" value={loadingDash ? '…' : dashboard?.summary?.total_production_mt != null ? `${formatNum(dashboard.summary.total_production_mt, 1)} MT` : '—'} hint={dashboard?.summary?.total_area_ha != null ? `Area: ${formatNum(dashboard.summary.total_area_ha, 1)} ha` : 'Area/production as available'} />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm lg:col-span-2">
          <div className="flex items-center justify-between gap-3">
            <h3 className="text-lg font-semibold text-slate-900">Current cultivation & production ({dashboard?.latest_year || 'latest'})</h3>
            <span className="text-xs text-slate-500">{loadingDash ? 'Loading…' : `${currentCrops.length} crops`}</span>
          </div>
          {currentCrops.length ? (
            <div className="mt-4 overflow-x-auto">
              <table className="min-w-full text-left text-sm">
                <thead className="text-slate-500">
                  <tr>
                    <th className="px-3 py-2 font-medium">Crop</th>
                    <th className="px-3 py-2 font-medium">Area (ha)</th>
                    <th className="px-3 py-2 font-medium">Production (MT)</th>
                    <th className="px-3 py-2 font-medium">Yield (kg/ha)</th>
                    <th className="px-3 py-2 font-medium">Yield (t/ha)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {currentCrops.map((c) => (
                    <tr key={c.crop}>
                      <td className="px-3 py-2 font-medium text-slate-900">{c.crop}</td>
                      <td className="px-3 py-2 text-slate-700">{formatNum(c.area_ha, 1)}</td>
                      <td className="px-3 py-2 text-slate-700">{formatNum(c.production_mt, 1)}</td>
                      <td className="px-3 py-2 text-slate-700">{formatNum(c.yield_kg_ha, 1)}</td>
                      <td className="px-3 py-2 text-slate-700">{formatNum(c.yield_ton_ha, 3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="mt-4 text-sm text-slate-500">{loadingDash ? 'Loading cultivation data…' : 'No cultivation records available for this district.'}</p>
          )}
          {dashboard?.data_note ? <p className="mt-3 text-xs text-slate-500">{dashboard.data_note}</p> : null}
        </div>

        <div className="space-y-6">
          <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
            <h3 className="text-lg font-semibold text-slate-900">Current status (environment)</h3>
            {loadingDash ? (
              <p className="mt-3 text-sm text-slate-500">Loading status…</p>
            ) : (
              <div className="mt-4 space-y-3 text-sm">
                <div className="flex items-center justify-between"><span className="text-slate-500">Vegetation (NDVI)</span><StatusPill label={env.vegetation_status?.label} /></div>
                <div className="flex items-center justify-between"><span className="text-slate-500">NDVI mean</span><span className="font-medium text-slate-900">{formatNum(env.ndvi_mean, 3)}</span></div>
                <div className="flex items-center justify-between"><span className="text-slate-500">Temperature</span><span className="font-medium text-slate-900">{env.temperature_c_mean != null ? `${formatNum(env.temperature_c_mean, 1)} °C` : '—'}</span></div>
                <div className="flex items-center justify-between"><span className="text-slate-500">Rainfall (annual)</span><span className="font-medium text-slate-900">{env.rainfall_mm_mean != null ? `${formatNum(env.rainfall_mm_mean, 1)} mm` : '—'}</span></div>
                <div className="flex items-center justify-between"><span className="text-slate-500">Soil moisture</span><span className="font-medium text-slate-900">{formatNum(env.soil_moisture_mean, 3)}</span></div>
                <p className="text-xs text-slate-500 pt-1">{env.vegetation_status?.note || 'Satellite/climate indicators from available district data.'}</p>
              </div>
            )}
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-slate-900">KrishiMap & Calendar</h3>
              <Link to="/map" className="text-sm font-medium text-crop-700 hover:text-crop-600">Map</Link>
            </div>
            <p className="mt-2 text-sm text-slate-600">District geospatial coverage and seasonal crop planning context.</p>
            <div className="mt-3 space-y-2">
              {(seasons.length ? seasons : crops.map(c => ({ crop: c.name, seasons: c.seasons || [] }))).slice(0, 4).map((item) => (
                <div key={item.crop} className="rounded-md bg-slate-50 px-3 py-2 text-sm">
                  <p className="font-medium text-slate-900">{item.crop}</p>
                  <p className="text-slate-600">{(item.seasons || []).map(s => `${s.planting} → ${s.harvest}`).join(' · ') || '—'}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
        <p className="font-medium">KrishiAlert (Pilot)</p>
        <p className="mt-1">Vegetation status is satellite-derived (NDVI context), not a confirmed crop disease or field diagnosis. Disease image prediction and farmer reporting/verification workflows are planned for future phases.</p>
      </div>
    </section>
  )
}
