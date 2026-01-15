<template>
  <div>
    <h2 class="text-3xl font-bold mb-6 text-gray-900 dark:text-white">Documents</h2>

    <!-- Loading State -->
    <div v-if="loading" class="space-y-4">
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-4 animate-pulse">
        <div class="h-10 bg-gray-200 dark:bg-gray-700 rounded w-full mb-4"></div>
        <div class="flex gap-4">
          <div class="h-10 bg-gray-200 dark:bg-gray-700 rounded w-32"></div>
          <div class="h-10 bg-gray-200 dark:bg-gray-700 rounded w-32"></div>
          <div class="h-10 bg-gray-200 dark:bg-gray-700 rounded w-32"></div>
        </div>
      </div>
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden">
        <div v-for="i in 5" :key="i" class="p-4 border-b border-gray-200 dark:border-gray-700 animate-pulse">
          <div class="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4 mb-2"></div>
          <div class="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
        </div>
      </div>
    </div>

    <!-- Error State -->
    <div v-else-if="error" class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6">
      <div class="flex items-center gap-3">
        <svg class="w-6 h-6 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <p class="text-red-800 dark:text-red-200 font-medium">Error: {{ error }}</p>
      </div>
    </div>

    <!-- Main Content -->
    <div v-else>
      <!-- Search and Filters -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-4 mb-6 transition-colors">
        <!-- Search Bar -->
        <div class="relative mb-4">
          <input
            v-model="searchQuery"
            type="text"
            placeholder="Search documents by filename, topics, people..."
            class="w-full pl-10 pr-4 py-2.5 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
          />
          <svg class="absolute left-3 top-3 w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>

        <!-- Filter Row -->
        <div class="flex flex-wrap gap-3">
          <!-- Doc Type Filter -->
          <select
            v-model="filterDocType"
            class="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
          >
            <option value="">All Types</option>
            <option value="letter">Letter</option>
            <option value="newspaper_article">Newspaper Article</option>
            <option value="other">Other</option>
          </select>

          <!-- Content Type Filter -->
          <select
            v-model="filterContentType"
            class="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
          >
            <option value="">All Content</option>
            <option value="handwritten">Handwritten</option>
            <option value="typed">Typed</option>
            <option value="mixed">Mixed</option>
          </select>

          <!-- Language Filter -->
          <select
            v-model="filterLanguage"
            class="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
          >
            <option value="">All Languages</option>
            <option v-for="lang in availableLanguages" :key="lang" :value="lang">
              {{ languageName(lang) }}
            </option>
          </select>

          <!-- Flagged Filter -->
          <label class="flex items-center gap-2 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700 text-sm text-gray-700 dark:text-gray-300 transition-colors">
            <input
              type="checkbox"
              v-model="filterFlagged"
              class="rounded text-blue-600 focus:ring-blue-500 dark:bg-gray-600 dark:border-gray-500"
            />
            <span>Flagged only</span>
          </label>

          <!-- Clear Filters -->
          <button
            v-if="hasActiveFilters"
            @click="clearFilters"
            class="px-3 py-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
          >
            Clear filters
          </button>
        </div>
      </div>

      <!-- Results Count -->
      <p class="text-sm text-gray-600 dark:text-gray-400 mb-4">
        Showing {{ filteredDocuments.length }} of {{ documents.length }} documents
      </p>

      <!-- Empty State -->
      <div v-if="filteredDocuments.length === 0" class="bg-white dark:bg-gray-800 rounded-lg shadow p-12 text-center">
        <svg class="w-16 h-16 text-gray-300 dark:text-gray-600 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        <p class="text-gray-500 dark:text-gray-400 text-lg mb-2">No documents found</p>
        <p class="text-gray-400 dark:text-gray-500 text-sm">Try adjusting your search or filters</p>
      </div>

      <!-- Documents Table -->
      <div v-else class="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden transition-colors">
        <div class="overflow-x-auto">
          <table class="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
            <thead class="bg-gray-50 dark:bg-gray-900">
              <tr>
                <th
                  @click="toggleSort('filename')"
                  class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                >
                  <div class="flex items-center gap-1">
                    Filename
                    <SortIcon :active="sortField === 'filename'" :direction="sortDirection" />
                  </div>
                </th>
                <th
                  @click="toggleSort('doc_type')"
                  class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                >
                  <div class="flex items-center gap-1">
                    Type
                    <SortIcon :active="sortField === 'doc_type'" :direction="sortDirection" />
                  </div>
                </th>
                <th
                  @click="toggleSort('date')"
                  class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                >
                  <div class="flex items-center gap-1">
                    Date
                    <SortIcon :active="sortField === 'date'" :direction="sortDirection" />
                  </div>
                </th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Topics
                </th>
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Language
                </th>
              </tr>
            </thead>
            <tbody class="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
              <tr
                v-for="doc in paginatedDocuments"
                :key="doc.id"
                @click="openDocument(doc)"
                class="hover:bg-blue-50 dark:hover:bg-gray-700 cursor-pointer transition-colors"
              >
                <td class="px-6 py-4 whitespace-nowrap">
                  <div class="flex items-center gap-2">
                    <span class="text-sm font-medium text-gray-900 dark:text-white">{{ truncateFilename(doc.filename) }}</span>
                    <span v-if="doc.flagged_for_review" class="w-2 h-2 rounded-full bg-yellow-400" title="Flagged for review"></span>
                  </div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                  <span :class="getDocTypeBadgeClass(doc.doc_type)">
                    {{ formatLabel(doc.doc_type) }}
                  </span>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-gray-400">
                  {{ doc.date || 'N/A' }}
                </td>
                <td class="px-6 py-4">
                  <span v-if="doc.topics.length > 0" class="inline-flex flex-wrap gap-1">
                    <span
                      v-for="topic in doc.topics.slice(0, 3)"
                      :key="topic"
                      class="px-2 py-0.5 bg-blue-100 dark:bg-blue-900/50 text-blue-800 dark:text-blue-300 rounded text-xs"
                    >
                      {{ topic }}
                    </span>
                    <span v-if="doc.topics.length > 3" class="text-gray-400 text-xs">
                      +{{ doc.topics.length - 3 }}
                    </span>
                  </span>
                  <span v-else class="text-gray-400 dark:text-gray-500 text-xs">No topics</span>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                  <span
                    v-for="lang in doc.languages.slice(0, 2)"
                    :key="lang"
                    class="px-2 py-0.5 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded text-xs mr-1"
                  >
                    {{ lang.toUpperCase() }}
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Pagination -->
        <div v-if="totalPages > 1" class="bg-gray-50 dark:bg-gray-900 px-6 py-3 flex items-center justify-between border-t border-gray-200 dark:border-gray-700">
          <p class="text-sm text-gray-600 dark:text-gray-400">
            Page {{ currentPage }} of {{ totalPages }}
          </p>
          <div class="flex gap-2">
            <button
              @click="currentPage--"
              :disabled="currentPage === 1"
              class="px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            >
              Previous
            </button>
            <button
              @click="currentPage++"
              :disabled="currentPage === totalPages"
              class="px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            >
              Next
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Document Detail Modal -->
    <DocumentDetailModal
      :show="showModal"
      :document="selectedDocument"
      @close="closeModal"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useDocuments } from '@/composables/useData'
import DocumentDetailModal from '@/components/DocumentDetailModal.vue'
import type { Document } from '@/types'

const { documents, loading, error } = useDocuments()

// Search and Filters
const searchQuery = ref('')
const filterDocType = ref('')
const filterContentType = ref('')
const filterLanguage = ref('')
const filterFlagged = ref(false)

// Sorting
const sortField = ref<'filename' | 'doc_type' | 'date'>('filename')
const sortDirection = ref<'asc' | 'desc'>('asc')

// Pagination
const currentPage = ref(1)
const itemsPerPage = 15

// Modal
const showModal = ref(false)
const selectedDocument = ref<Document | null>(null)

// Language names map
const languageNames: Record<string, string> = {
  nl: 'Dutch',
  en: 'English',
  de: 'German',
  fr: 'French',
  es: 'Spanish',
  it: 'Italian'
}

function languageName(code: string): string {
  return languageNames[code] || code.toUpperCase()
}

// Available languages from documents
const availableLanguages = computed(() => {
  const langs = new Set<string>()
  documents.value.forEach(doc => doc.languages.forEach(l => langs.add(l)))
  return Array.from(langs).sort()
})

// Check if any filters are active
const hasActiveFilters = computed(() => {
  return searchQuery.value || filterDocType.value || filterContentType.value || filterLanguage.value || filterFlagged.value
})

// Clear all filters
function clearFilters() {
  searchQuery.value = ''
  filterDocType.value = ''
  filterContentType.value = ''
  filterLanguage.value = ''
  filterFlagged.value = false
  currentPage.value = 1
}

// Filtered documents
const filteredDocuments = computed(() => {
  let result = [...documents.value]

  // Search filter
  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase()
    result = result.filter(doc =>
      doc.filename.toLowerCase().includes(query) ||
      doc.topics.some(t => t.toLowerCase().includes(query)) ||
      doc.people_mentioned.some(p => p.toLowerCase().includes(query)) ||
      (doc.sender?.toLowerCase().includes(query)) ||
      (doc.recipient?.toLowerCase().includes(query)) ||
      (doc.location?.toLowerCase().includes(query))
    )
  }

  // Doc type filter
  if (filterDocType.value) {
    result = result.filter(doc => doc.doc_type === filterDocType.value)
  }

  // Content type filter
  if (filterContentType.value) {
    result = result.filter(doc => doc.content_type === filterContentType.value)
  }

  // Language filter
  if (filterLanguage.value) {
    result = result.filter(doc => doc.languages.includes(filterLanguage.value))
  }

  // Flagged filter
  if (filterFlagged.value) {
    result = result.filter(doc => doc.flagged_for_review)
  }

  // Sort
  result.sort((a, b) => {
    let aVal: string | null
    let bVal: string | null

    switch (sortField.value) {
      case 'filename':
        aVal = a.filename
        bVal = b.filename
        break
      case 'doc_type':
        aVal = a.doc_type
        bVal = b.doc_type
        break
      case 'date':
        aVal = a.date
        bVal = b.date
        break
      default:
        return 0
    }

    // Handle null values
    if (aVal === null && bVal === null) return 0
    if (aVal === null) return 1
    if (bVal === null) return -1

    const comparison = aVal.localeCompare(bVal)
    return sortDirection.value === 'asc' ? comparison : -comparison
  })

  return result
})

// Pagination
const totalPages = computed(() => Math.ceil(filteredDocuments.value.length / itemsPerPage))

const paginatedDocuments = computed(() => {
  const start = (currentPage.value - 1) * itemsPerPage
  return filteredDocuments.value.slice(start, start + itemsPerPage)
})

// Reset page when filters change
watch([searchQuery, filterDocType, filterContentType, filterLanguage, filterFlagged], () => {
  currentPage.value = 1
})

// Sort toggle
function toggleSort(field: 'filename' | 'doc_type' | 'date') {
  if (sortField.value === field) {
    sortDirection.value = sortDirection.value === 'asc' ? 'desc' : 'asc'
  } else {
    sortField.value = field
    sortDirection.value = 'asc'
  }
}

// Modal handlers
function openDocument(doc: Document) {
  selectedDocument.value = doc
  showModal.value = true
}

function closeModal() {
  showModal.value = false
  selectedDocument.value = null
}

// Helper functions
function formatLabel(text: string): string {
  return text
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

function truncateFilename(filename: string, maxLength = 40): string {
  if (filename.length <= maxLength) return filename
  return filename.slice(0, maxLength - 3) + '...'
}

function getDocTypeBadgeClass(type: string): string {
  const base = 'px-2 py-0.5 text-xs font-medium rounded'
  switch (type) {
    case 'letter':
      return `${base} bg-blue-100 text-blue-800`
    case 'newspaper_article':
      return `${base} bg-green-100 text-green-800`
    default:
      return `${base} bg-gray-100 text-gray-700`
  }
}

// Sort icon component (inline)
const SortIcon = {
  props: {
    active: Boolean,
    direction: String
  },
  template: `
    <svg class="w-4 h-4" :class="active ? 'text-blue-600' : 'text-gray-400'" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path v-if="!active || direction === 'asc'" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 15l7-7 7 7" />
      <path v-if="active && direction === 'desc'" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
    </svg>
  `
}
</script>
