import { useEffect, useMemo, useRef, useState } from 'react'

function normalize(text) {
  return String(text || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim()
}

function compact(text) {
  return normalize(text).replace(/\s+/g, '')
}

function districtLabel(d) {
  if (!d) return ''
  return d.province ? `${d.district} (${d.province})` : d.district
}

export function resolveDistrictInput(districts, input, { allowAll = false } = {}) {
  const qn = normalize(input)
  const qc = compact(input)
  if (!qn) return allowAll ? '' : null

  const exactDistrict = districts.find((d) => normalize(d.district) === qn || compact(d.district) === qc)
  if (exactDistrict) return exactDistrict.district

  const exactLabel = districts.find((d) => normalize(districtLabel(d)) === qn || compact(districtLabel(d)) === qc)
  if (exactLabel) return exactLabel.district

  const provinceMatches = districts.filter((d) => normalize(d.province || '') === qn || compact(d.province || '') === qc)
  if (provinceMatches.length >= 1) return provinceMatches[0].district

  const filtered = districts.filter((d) => {
    const dn = normalize(d.district)
    const pn = normalize(d.province || '')
    const dc = compact(d.district)
    const pc = compact(d.province || '')
    return (
      dn.includes(qn) || pn.includes(qn) || dc.includes(qc) || pc.includes(qc) ||
      (dn && qn.includes(dn)) || (pn && qn.includes(pn)) || (dc && qc.includes(dc)) || (pc && qc.includes(pc))
    )
  })

  if (filtered.length === 1) return filtered[0].district
  if (filtered.length > 1) {
    const starts = filtered.filter(
      (d) =>
        normalize(d.district).startsWith(qn) ||
        normalize(d.province || '').startsWith(qn) ||
        compact(d.district).startsWith(qc) ||
        compact(d.province || '').startsWith(qc)
    )
    if (starts.length >= 1) return starts[0].district
    return filtered[0].district
  }

  return null
}

export default function DistrictSelect({
  districts = [],
  value = '',
  onChange,
  placeholder = 'Search district or province...',
  allowAll = false,
  allLabel = 'All districts (overview)',
  required = false,
}) {
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [activeIndex, setActiveIndex] = useState(0)
  const wrapRef = useRef(null)
  const inputRef = useRef(null)

  const selected = useMemo(() => districts.find((d) => d.district === value) || null, [districts, value])

  useEffect(() => {
    if (!open) {
      setQuery(value ? districtLabel(selected) || value : '')
    }
  }, [value, selected, open])

  useEffect(() => {
    function onDoc(e) {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [])

  const filtered = useMemo(() => {
    const qn = normalize(query)
    const qc = compact(query)
    if (!qn) return districts
    return districts.filter((d) => {
      const dn = normalize(d.district)
      const pn = normalize(d.province || '')
      const dc = compact(d.district)
      const pc = compact(d.province || '')
      return (
        dn.includes(qn) || pn.includes(qn) || dc.includes(qc) || pc.includes(qc) ||
        (dn && qn.includes(dn)) || (pn && qn.includes(pn))
      )
    })
  }, [districts, query])

  function commitSelection(districtName) {
    onChange?.(districtName)
    if (districtName) {
      const d = districts.find((x) => x.district === districtName)
      setQuery(districtLabel(d) || districtName)
    } else {
      setQuery('')
    }
    setOpen(false)
  }

  function tryResolveAndSelect(val) {
    const resolved = resolveDistrictInput(districts, val, { allowAll })
    if (resolved !== null) {
      onChange?.(resolved)
      return resolved
    }
    if (allowAll && !normalize(val)) {
      onChange?.('')
      return ''
    }
    return null
  }

  function onInputChange(e) {
    const val = e.target.value
    setQuery(val)
    setOpen(true)
    setActiveIndex(0)
    tryResolveAndSelect(val)
  }

  function onKeyDown(e) {
    const items = allowAll ? [{ __all: true }, ...filtered] : filtered
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setOpen(true)
      setActiveIndex((i) => Math.min(i + 1, Math.max(items.length - 1, 0)))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActiveIndex((i) => Math.max(i - 1, 0))
    } else if (e.key === 'Enter') {
      if (open && items.length > 0) {
        e.preventDefault()
        const idx = activeIndex >= 0 && activeIndex < items.length ? activeIndex : 0
        const item = items[idx]
        if (item?.__all) commitSelection('')
        else if (item) commitSelection(item.district)
        else {
          const resolved = tryResolveAndSelect(query)
          if (resolved !== null) commitSelection(resolved)
        }
      } else {
        const resolved = tryResolveAndSelect(query)
        if (resolved !== null) {
          e.preventDefault()
          commitSelection(resolved)
        }
      }
    } else if (e.key === 'Escape') {
      setOpen(false)
      setQuery(value ? districtLabel(selected) || value : '')
    }
  }

  function onBlur() {
    setTimeout(() => {
      if (!wrapRef.current) return
      if (wrapRef.current.contains(document.activeElement)) return
      const resolved = resolveDistrictInput(districts, query, { allowAll })
      if (resolved !== null) {
        const d = districts.find((x) => x.district === resolved)
        setQuery(resolved ? districtLabel(d) || resolved : '')
        if (resolved !== value) onChange?.(resolved)
      } else {
        setQuery(value ? districtLabel(selected) || value : '')
      }
      setOpen(false)
    }, 120)
  }

  const listItems = allowAll ? [{ __all: true }, ...filtered] : filtered

  return (
    <div ref={wrapRef} className="relative">
      <input
        ref={inputRef}
        type="text"
        value={query}
        onChange={onInputChange}
        onFocus={() => { setOpen(true); setActiveIndex(0) }}
        onBlur={onBlur}
        onKeyDown={onKeyDown}
        placeholder={placeholder}
        className="w-full rounded-md border border-slate-300 bg-white px-3 py-2 focus:border-crop-500 focus:outline-none focus:ring-1 focus:ring-crop-500"
        autoComplete="off"
      />
      {required ? <input tabIndex={-1} required value={value || ''} onChange={() => {}} className="sr-only absolute h-0 w-0 opacity-0" aria-hidden="true" /> : null}
      {open ? (
        <div className="absolute z-20 mt-1 max-h-64 w-full overflow-auto rounded-md border border-slate-200 bg-white shadow-lg">
          {listItems.length === 0 ? (
            <div className="px-3 py-2 text-sm text-slate-500">No districts found</div>
          ) : (
            listItems.map((item, idx) => {
              const isAll = item.__all
              const label = isAll ? allLabel : districtLabel(item)
              const isSelected = isAll ? !value : item.district === value
              const active = idx === activeIndex
              return (
                <button
                  key={isAll ? '__all' : `${item.province}-${item.district}`}
                  type="button"
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => commitSelection(isAll ? '' : item.district)}
                  className={`block w-full px-3 py-2 text-left text-sm ${active ? 'bg-crop-50 text-crop-800' : 'text-slate-700 hover:bg-slate-50'} ${isSelected ? 'font-medium' : ''}`}
                >
                  {label}
                </button>
              )
            })
          )}
        </div>
      ) : null}
    </div>
  )
}
