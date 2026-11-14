import apiClient from './client'

export async function fetchApiRoot() {
  const { data } = await apiClient.get('/')
  return data
}

export async function fetchHealth() {
  const { data } = await apiClient.get('/health/')
  return data
}

export async function fetchCrops() {
  const { data } = await apiClient.get('/crops/')
  return data
}

export async function fetchCropSeasons(crop) {
  const { data } = await apiClient.get('/crops/seasons/', { params: crop ? { crop } : {} })
  return data
}

export async function fetchModelInfo() {
  const { data } = await apiClient.get('/model/')
  return data
}

export async function createPrediction(payload) {
  const { data } = await apiClient.post('/predict/', payload, { timeout: 180000 })
  return data
}

export async function fetchPredictionHistory(limit = 10) {
  const { data } = await apiClient.get('/predictions/history/', { params: { page_size: limit } })
  return data
}

export async function fetchNearestGrid(latitude, longitude) {
  const { data } = await apiClient.get('/satellite/grid/nearest/', { params: { latitude, longitude } })
  return data
}

export async function fetchDistricts() {
  const { data } = await apiClient.get('/satellite/districts/')
  return data
}

export async function fetchGridCells(params = {}) {
  const { data } = await apiClient.get('/satellite/grid/', { params })
  return data
}

export function getErrorMessage(error) {
  return (
    error?.response?.data?.error?.message ||
    error?.response?.data?.detail ||
    error?.message ||
    'Request failed'
  )
}

export async function fetchDistrictDashboard(district) {
  const { data } = await apiClient.get('/satellite/district-dashboard/', { params: { district } })
  return data
}
