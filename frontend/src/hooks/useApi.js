/**
 * Generic data-fetching hook with loading/error state.
 * Used across all dashboard pages to avoid boilerplate.
 */
import { useState, useEffect, useCallback } from 'react'

const BASE = '/api/v1'

export function useApi(path, params = {}) {
  const [data, setData]       = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState(null)

  const url = `${BASE}${path}?` + new URLSearchParams(params).toString()

  const fetch_ = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(url)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      setData(await res.json())
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [url])

  useEffect(() => { fetch_() }, [fetch_])

  return { data, loading, error, refetch: fetch_ }
}
