import { useEffect, useMemo, useState } from 'react'
import { createPrediction, fetchCropSeasons, fetchCrops, fetchDistricts, getErrorMessage } from '../api'
import DistrictSelect, { resolveDistrictInput } from '../components/DistrictSelect'

function formatNum(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return '—'
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits })
}

function QualityBadge({ level }) {
  const styles = {
    high: 'bg-green-100 text-green-800',
    medium: 'bg-yellow-100 text-yellow-800',
    low: 'bg-red-100 text-red-800',
  }
  return <span className={`inline-flex rounded-full px-2.5 py-1 text-xs font-medium ${styles[level] || 'bg-slate-100 text-slate-700'}`}>{level || 'unknown'}</span>
}

function SuitabilityBadge({ level }) {
  const styles = {
    high: 'bg-emerald-100 text-emerald-800',
    medium: 'bg-blue-100 text-blue-800',
    low: 'bg-amber-100 text-amber-800',
  }
  return <span className={`inline-flex rounded-full px-2.5 py-1 text-xs font-medium capitalize ${styles[level] || 'bg-slate-100 text-slate-700'}`}>{level || 'unknown'}</span>
}

function SeasonBadge({ season }) {
  if (!season) return <span className="text-slate-500">—</span>
  const inSeason = season.in_season
  const cls = inSeason ? 'bg-green-100 text-green-800' : inSeason === false ? 'bg-amber-100 text-amber-800' : 'bg-slate-100 text-slate-700'
  return <span className={`inline-flex rounded-full px-2.5 py-1 text-xs font-medium ${cls}`}>{inSeason ? 'In season' : inSeason === false ? 'Off season' : 'Season n/a'}</span>
}

const today = new Date().toISOString().slice(0, 10)

export default function Prediction() {
  const [form, setForm] = useState({ district: 'Rupandehi', date: today, crop: '', areaHa: '' })
  const [crops, setCrops] = useState([])
  const [seasons, setSeasons] = useState([])
  const [districts, setDistricts] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState(null)

  useEffect(() => {
    fetchCrops().then((res) => setCrops(res.results || [])).catch(() => {})
    fetchCropSeasons().then((res) => setSeasons(res.results || [])).catch(() => {})
    fetchDistricts()
      .then((res) => {
        const list = (res.results || []).slice().sort((a, b) => a.district.localeCompare(b.district))
        setDistricts(list)
        if (list.length) {
          const preferred = list.some((d) => d.district === 'Rupandehi') ? 'Rupandehi' : list[0].district
          setForm((f) => (list.some((d) => d.district === f.district) ? f : { ...f, district: preferred }))
        }
      })
      .catch(() => {})
  }, [])

  const selectedDistrict = useMemo(() => districts.find((d) => d.district === form.district), [districts, form.district])
  const selectedMonth = useMemo(() => {
    const d = new Date(form.date)
    return Number.isNaN(d.getTime()) ? null : d.getMonth() + 1
  }, [form.date])

  const calendarCrops = useMemo(() => {
    const items = seasons.length ? seasons : crops.map((c) => ({ crop: c.name, seasons: c.seasons || [] }))
    return items.map((item) => {
      const seasonList = item.seasons || []
      // backend seasons don't include month arrays; show windows; if prediction result later gives season info use that. fallback: mark unknown.
      return { ...item, seasonList }
    })
  }, [seasons, crops])

  const areaHa = Number(form.areaHa)
  const productionEstimate = useMemo(() => {
    if (!result?.recommended_crop || !Number.isFinite(areaHa) || areaHa <= 0) return null
    const yieldKgHa = Number(result.recommended_crop.predicted_yield_kg_ha)
    if (!Number.isFinite(yieldKgHa)) return null
    const totalKg = yieldKgHa * areaHa
    return { areaHa, totalKg, totalTon: totalKg / 1000, totalQuintal: totalKg / 100 }
  }, [result, areaHa])

  async function onSubmit(e) {
    e.preventDefault()
    setError('')
    setResult(null)
    let district = form.district
    if (!district) {
      // try resolve if user typed province/name but selection not committed
      const inputEl = e.currentTarget.querySelector('input[type="text"]')
      const typed = inputEl?.value || ''
      const resolved = resolveDistrictInput(districts, typed)
      if (resolved) district = resolved
    }
    if (!district) { setError('Please select a district.'); return }
    setLoading(true)
    try {
      const payload = { district, date: form.date }
      if (form.crop) payload.crop = form.crop
      const data = await createPrediction(payload)
      setResult(data)
      if (district !== form.district) setForm((f) => ({ ...f, district }))
    } catch (err) {
      setError(getErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="space-y-6">
      <header>
        <p className="text-sm font-medium text-crop-700">KrishiMap · KrishiCalendar · Decision Support</p>
        <h2 className="mt-1 text-2xl font-semibold text-slate-900">Crop intelligence & planning</h2>
        <p className="mt-1 max-w-3xl text-slate-600">
          Get district-specific crop recommendations, estimated yield per hectare, seasonal guidance, and satellite-based vegetation context for planning.
          AI-assisted outputs are preliminary decision support and should be verified with local agricultural expertise.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-[420px,1fr]">
        <div className="space-y-4">
          <form onSubmit={onSubmit} className="space-y-4 rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
            <h3 className="text-base font-semibold text-slate-900">Analysis input</h3>
            <label className="block text-sm">
              <span className="mb-1 block font-medium text-slate-700">District</span>
              <DistrictSelect
                districts={districts}
                value={form.district}
                onChange={(d) => setForm((f) => ({ ...f, district: d }))}
                placeholder="Search district or province..."
                required
              />
              <span className="mt-1 block text-xs text-slate-500">Type district or province (e.g. Sudurpashchim) and press Enter to select.</span>
            </label>

            <label className="block text-sm">
              <span className="mb-1 block font-medium text-slate-700">Date</span>
              <input type="date" value={form.date} onChange={(e) => setForm((f) => ({ ...f, date: e.target.value }))} className="w-full rounded-md border border-slate-300 px-3 py-2 focus:border-crop-500 focus:outline-none focus:ring-1 focus:ring-crop-500" required />
            </label>

            <label className="block text-sm">
              <span className="mb-1 block font-medium text-slate-700">Crop focus (optional)</span>
              <select value={form.crop} onChange={(e) => setForm((f) => ({ ...f, crop: e.target.value }))} className="w-full rounded-md border border-slate-300 px-3 py-2 focus:border-crop-500 focus:outline-none focus:ring-1 focus:ring-crop-500">
                <option value="">All supported crops (recommend best fit)</option>
                {crops.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
              </select>
              <span className="mt-1 block text-xs text-slate-500">Recommendations consider confidence, district suitability/history, and seasonal context.</span>
            </label>

            <label className="block text-sm">
              <span className="mb-1 block font-medium text-slate-700">Cultivated area for estimate (ha, optional)</span>
              <input type="number" min="0" step="any" value={form.areaHa} onChange={(e) => setForm((f) => ({ ...f, areaHa: e.target.value }))} placeholder="e.g. 0.5" className="w-full rounded-md border border-slate-300 px-3 py-2 focus:border-crop-500 focus:outline-none focus:ring-1 focus:ring-crop-500" />
              <span className="mt-1 block text-xs text-slate-500">Used to estimate total production from predicted yield (kg/ha × area). Clearly an estimate, not a verified record.</span>
            </label>

            {selectedDistrict ? (
              <div className="rounded-md bg-slate-50 p-3 text-sm text-slate-700">
                <p className="font-medium">Selected district</p>
                <p>{selectedDistrict.district}, {selectedDistrict.province}</p>
                <p className="text-slate-500">{selectedDistrict.grid_cells} analysis grid cells</p>
              </div>
            ) : null}

            <button type="submit" disabled={loading || !form.district} className="w-full rounded-md bg-crop-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-crop-700 disabled:cursor-not-allowed disabled:opacity-60">
              {loading ? 'Analyzing satellite & climate data…' : 'Run crop intelligence'}
            </button>

            {error ? <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">{error}</div> : null}
            <p className="text-xs text-slate-500">May take time when fetching real-time satellite/climate features. Predictions are location/date-specific decision support.</p>
          </form>

          <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
            <h3 className="text-base font-semibold text-slate-900">KrishiCalendar (seasonal guidance)</h3>
            <p className="mt-1 text-xs text-slate-500">Typical planting and harvesting windows. Local conditions and officer guidance should be considered.</p>
            <div className="mt-3 max-h-72 space-y-2 overflow-auto pr-1">
              {calendarCrops.map((item) => (
                <div key={item.crop} className="rounded-md border border-slate-100 bg-slate-50 p-3 text-sm">
                  <p className="font-medium text-slate-900">{item.crop}</p>
                  <ul className="mt-1 space-y-1 text-slate-600">
                    {(item.seasonList || []).length ? item.seasonList.map((s, i) => (
                      <li key={i}>• {s.season}: planting {s.planting}, harvest {s.harvest}</li>
                    )) : <li>• Season guidance unavailable</li>}
                  </ul>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
            <p className="font-medium">KrishiAlert (pilot note)</p>
            <p className="mt-1">Crop disease image prediction is not enabled in this pilot. Vegetation health shown in results is satellite-derived NDVI context, not a confirmed disease diagnosis. Report suspected diseases to agricultural officers for verification.</p>
          </div>
        </div>

        <div className="space-y-4">
          {result ? (
            <>
              <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="text-sm text-slate-500">Recommended crop (decision support)</p>
                    <h3 className="text-2xl font-semibold text-slate-900">{result.recommended_crop?.crop || '—'}</h3>
                    <p className="mt-1 text-slate-700">Estimated yield: {formatNum(result.recommended_crop?.predicted_yield_kg_ha, 1)} kg/ha ({formatNum(result.recommended_crop?.yield_ton_ha, 3)} t/ha)</p>
                    {productionEstimate ? (
                      <p className="mt-1 text-sm font-medium text-crop-700">
                        Estimated production for {formatNum(productionEstimate.areaHa, 2)} ha: {formatNum(productionEstimate.totalTon, 2)} t ({formatNum(productionEstimate.totalQuintal, 1)} quintal)
                      </p>
                    ) : null}
                    <p className="mt-1 text-xs text-slate-500">Ranked by recommendation score (confidence × district suitability × seasonal context), not raw yield.</p>
                  </div>
                  <div className="text-right text-sm">
                    <p className="text-slate-500">Model confidence</p>
                    <p className="text-lg font-semibold text-slate-900">{formatNum((result.recommended_crop?.confidence || 0) * 100, 1)}%</p>
                    <p className="mt-1 text-slate-500">Suitability: <SuitabilityBadge level={result.recommended_crop?.suitability?.level} /></p>
                    <p className="mt-1 text-slate-500">Season: <SeasonBadge season={result.recommended_crop?.season} /></p>
                  </div>
                </div>
                <dl className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
                  <div><dt className="text-slate-500">District</dt><dd className="font-medium text-slate-900">{result.location?.selected_district || result.location?.district}</dd></div>
                  <div><dt className="text-slate-500">Province</dt><dd className="font-medium text-slate-900">{result.location?.selected_province || result.location?.province}</dd></div>
                  <div><dt className="text-slate-500">Analysis cell</dt><dd className="font-medium text-slate-900">{result.location?.grid_id}</dd></div>
                  <div><dt className="text-slate-500">Data quality</dt><dd className="font-medium text-slate-900"><QualityBadge level={result.data_quality?.level} /></dd></div>
                </dl>
                {result.recommended_crop?.recommendation_reason ? <p className="mt-3 text-sm text-slate-600">{result.recommended_crop.recommendation_reason}</p> : null}
                <p className="mt-2 text-xs text-slate-500">Vegetation health indicator (NDVI context): <span className="font-medium capitalize text-slate-700">{result.recommended_crop?.health?.label || 'unknown'}</span>. This is not a disease diagnosis.</p>
              </div>

              <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
                <h3 className="text-lg font-semibold text-slate-900">Crop comparison & estimates</h3>
                <div className="mt-4 overflow-x-auto">
                  <table className="min-w-full text-left text-sm">
                    <thead className="text-slate-500">
                      <tr>
                        <th className="px-3 py-2 font-medium">Crop</th>
                        <th className="px-3 py-2 font-medium">Yield (kg/ha)</th>
                        <th className="px-3 py-2 font-medium">Est. prod.*</th>
                        <th className="px-3 py-2 font-medium">Confidence</th>
                        <th className="px-3 py-2 font-medium">Suitability</th>
                        <th className="px-3 py-2 font-medium">Season</th>
                        <th className="px-3 py-2 font-medium">Score</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {(result.predictions || []).map((p) => {
                        const est = Number.isFinite(areaHa) && areaHa > 0 ? (Number(p.predicted_yield_kg_ha) * areaHa) / 1000 : null
                        return (
                          <tr key={p.crop} className={p.crop === result.recommended_crop?.crop ? 'bg-crop-50/50' : ''}>
                            <td className="px-3 py-2 font-medium text-slate-900">
                              <div>{p.crop}</div>
                              <div className="text-xs font-normal text-slate-500">{p.recommendation_reason}</div>
                            </td>
                            <td className="px-3 py-2 text-slate-700">{formatNum(p.predicted_yield_kg_ha, 1)} <span className="text-xs text-slate-500">({formatNum(p.yield_ton_ha, 2)} t/ha)</span></td>
                            <td className="px-3 py-2 text-slate-700">{est != null ? `${formatNum(est, 2)} t` : '—'}</td>
                            <td className="px-3 py-2 text-slate-700">{formatNum(p.confidence * 100, 1)}%</td>
                            <td className="px-3 py-2 text-slate-700"><SuitabilityBadge level={p.suitability?.level} /></td>
                            <td className="px-3 py-2 text-slate-700"><SeasonBadge season={p.season} /></td>
                            <td className="px-3 py-2 text-slate-700">{formatNum((p.recommendation_score || 0) * 100, 1)}%</td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
                <p className="mt-3 text-xs text-slate-500">*Estimated production = predicted yield × entered cultivated area (indicative). District averages and field conditions may vary.</p>

                <div className="mt-4 grid gap-4 text-sm sm:grid-cols-2">
                  <div className="rounded-md bg-slate-50 p-3">
                    <p className="font-medium text-slate-900">Data sources ({result.period?.start_date} to {result.period?.end_date})</p>
                    <ul className="mt-2 space-y-1 text-slate-700">
                      {Object.entries(result.sources || {}).map(([name, meta]) => (
                        <li key={name}><span className="capitalize">{name}</span>: {meta.status} (images: {meta.image_count ?? '—'})</li>
                      ))}
                    </ul>
                  </div>
                  <div className="rounded-md bg-slate-50 p-3">
                    <p className="font-medium text-slate-900">Model & guidance</p>
                    <p className="mt-2 text-slate-700">{result.model?.name} · target {result.model?.target}</p>
                    <p className="text-slate-700">Feature coverage: {formatNum((result.data_quality?.feature_coverage || 0) * 100, 1)}%</p>
                    <p className="mt-1 text-slate-600">Use with local agronomic advice; not a substitute for verified field surveys or officer decisions.</p>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="rounded-lg border border-dashed border-slate-300 bg-white p-6 text-sm text-slate-500">
              Select a district and date to view crop recommendations, estimated yields, seasonal suitability, and production estimates for planning.
            </div>
          )}
        </div>
      </div>
    </section>
  )
}
