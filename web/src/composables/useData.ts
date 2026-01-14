import { ref, onMounted } from 'vue'
import type { Stats, Document, NetworkData } from '@/types'

export function useStats() {
  const stats = ref<Stats | null>(null)
  const loading = ref(true)
  const error = ref<string | null>(null)

  async function fetchStats() {
    try {
      loading.value = true
      error.value = null
      const response = await fetch('/data/stats.json')

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      stats.value = await response.json()
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to load stats'
      console.error('Error loading stats:', e)
    } finally {
      loading.value = false
    }
  }

  onMounted(() => {
    fetchStats()
  })

  return { stats, loading, error, refetch: fetchStats }
}

export function useDocuments() {
  const documents = ref<Document[]>([])
  const loading = ref(true)
  const error = ref<string | null>(null)

  async function fetchDocuments() {
    try {
      loading.value = true
      error.value = null
      const response = await fetch('/data/documents.json')

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      documents.value = await response.json()
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to load documents'
      console.error('Error loading documents:', e)
    } finally {
      loading.value = false
    }
  }

  onMounted(() => {
    fetchDocuments()
  })

  return { documents, loading, error, refetch: fetchDocuments }
}

export function useNetwork() {
  const network = ref<NetworkData | null>(null)
  const loading = ref(true)
  const error = ref<string | null>(null)

  async function fetchNetwork() {
    try {
      loading.value = true
      error.value = null
      const response = await fetch('/data/network.json')

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      network.value = await response.json()
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to load network data'
      console.error('Error loading network:', e)
    } finally {
      loading.value = false
    }
  }

  onMounted(() => {
    fetchNetwork()
  })

  return { network, loading, error, refetch: fetchNetwork }
}
