import { useEffect, useMemo, useState } from 'react'
import { CircleMarker, MapContainer, Popup, TileLayer, useMap } from 'react-leaflet'
import { fetchDistricts, fetchGridCells, getErrorMessage } from '../api'
import DistrictSelect from '../components/DistrictSelect'

function FitBounds({ points }) {
  const map = useMap()
  useEffect(() => {
    if (!points.length) return
    const bounds = points.map((p) => [p.latitude, p.longitude])
    if (bounds.length === 1) map.setView(bounds[0], 10)
    else map.fitBounds(bounds, { padding: [20, 20] })
  }, [map, points])
  return null
}

export default function MapVisualization() {
  const [districts, setDistricts] = useState([])
  const [selectedDistrict, setSelectedDistrict] = useState('Rupandehi')
  const [cells, setCells] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    fetchDistricts().then((res) => {
      const list = (res.results || []).slice().sort((a, b) => a.district.localeCompare(b.district))
      setDistricts(list)
      if (list.length) {
        const preferred = list.some((d) => d.district === 'Rupandehi') ? 'Rupandehi' : list[0].district
        setSelectedDistrict((prev) => (list.some((d) => d.district === prev) ? prev : preferred))
      }
    }).catch(() => {})
  }, [])

  useEffect(() => {
    let active = true
    async function loadCells() {
      setLoading(true)
      setError('')
      try {
        const params = { limit: 500 }
        if (selectedDistrict) params.district = selectedDistrict
        const res = await fetchGridCells(params)
        if (!active) return
        setCells((res.results || []).filter((c) => c.latitude != null && c.longitude != null))
      } catch (err) {
        if (active) setError(getErrorMessage(err))
      } finally {
        if (active) setLoading(false)
      }
    }
    loadCells()
    return () => { active = false }
  }, [selectedDistrict])

  const points = useMemo(() => {
    if (cells.length) return cells
    return districts.filter((d) => d.latitude != null && d.longitude != null)
  }, [cells, districts])

  const selectedInfo = useMemo(() => districts.find(d => d.district === selectedDistrict), [districts, selectedDistrict])

  return (
    <section className="space-y-6">
      <header>
        <p className="text-sm font-medium text-crop-700">KrishiMap</p>
        <h2 className="mt-1 text-2xl font-semibold text-slate-900">Farm mapping & production intelligence view</h2>
        <p className="mt-1 max-w-3xl text-slate-600">
          Explore district-level geospatial coverage used for agricultural analysis. This pilot visualizes analysis grid cells and district distribution to support planning, monitoring, and location-specific decisions.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-[320px,1fr]">
        <div className="space-y-4">
          <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <label className="block text-sm">
              <span className="mb-1 block font-medium text-slate-700">District</span>
              <DistrictSelect
                districts={districts}
                value={selectedDistrict}
                onChange={setSelectedDistrict}
                placeholder="Search district or province..."
                allowAll
                allLabel="All districts (overview)"
              />
              <span className="mt-1 block text-xs text-slate-500">Type a district or province (e.g. Sudurpashchim) and press Enter to select.</span>
            </label>
            <p className="mt-2 text-xs text-slate-500">Showing {cells.length ? `${cells.length} grid cells` : `${districts.length} district points`}.</p>
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm text-sm text-slate-700 space-y-2">
            <p className="font-medium text-slate-900">KrishiMap summary</p>
            <p>• Total districts: <span className="font-medium">{districts.length || '—'}</span></p>
            <p>• Selected district: <span className="font-medium">{selectedInfo?.district || 'All (overview)'}</span></p>
            {selectedInfo ? <p>• Province: <span className="font-medium">{selectedInfo.province}</span></p> : null}
            {selectedInfo ? <p>• Analysis grid cells: <span className="font-medium">{selectedInfo.grid_cells}</span></p> : null}
            <p className="text-xs text-slate-500 pt-1">Pilot scope uses satellite/climate grid data. Farmer-submitted farm boundaries and verified records can be integrated in future phases.</p>
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm text-sm text-slate-700">
            <p className="font-medium text-slate-900">How to use</p>
            <ul className="mt-2 space-y-1">
              <li>• Select a district to view its analysis cells.</li>
              <li>• Use <span className="font-medium">Krishi Intelligence</span> to get crop recommendations and production estimates for a district and date.</li>
              <li>• Use outputs as planning guidance; verify important records with agriculture officers.</li>
            </ul>
          </div>

          {error ? <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">{error}</div> : null}
          {loading ? <p className="text-sm text-slate-500">Loading map data…</p> : null}
        </div>

        <div className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
          <div className="h-[600px]">
            <MapContainer center={[28.3949, 84.124]} zoom={7} style={{ height: '100%', width: '100%' }}>
              <TileLayer attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
              <FitBounds points={points} />
              {points.map((p, i) => (
                <CircleMarker key={p.grid_id || `${p.district}-${i}`} center={[Number(p.latitude), Number(p.longitude)]} radius={cells.length ? 3 : 6} pathOptions={{ color: '#46702f', fillColor: '#5b8c3e', fillOpacity: 0.7, weight: 1 }}>
                  <Popup>
                    <div className="text-sm">
                      {p.grid_id ? <p className="font-medium">{p.grid_id}</p> : null}
                      <p>{p.district}{p.province ? `, ${p.province}` : ''}</p>
                      {p.grid_cells ? <p>{p.grid_cells} analysis grid cells</p> : null}
                      <p>{Number(p.latitude).toFixed(4)}, {Number(p.longitude).toFixed(4)}</p>
                    </div>
                  </Popup>
                </CircleMarker>
              ))}
            </MapContainer>
          </div>
        </div>
      </div>
    </section>
  )
}
