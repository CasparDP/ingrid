<template>
  <div>
    <h2 class="text-3xl font-bold mb-6 text-gray-900 dark:text-white">Dashboard</h2>

    <!-- Loading State with Skeletons -->
    <div v-if="loading" class="space-y-8">
      <!-- Stats Cards Skeleton -->
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div v-for="i in 4" :key="i" class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 animate-pulse">
          <div class="h-4 bg-gray-200 dark:bg-gray-700 rounded w-24 mb-3"></div>
          <div class="h-10 bg-gray-200 dark:bg-gray-700 rounded w-16"></div>
        </div>
      </div>

      <!-- Breakdown Cards Skeleton -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div v-for="i in 3" :key="i" class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 animate-pulse">
          <div class="h-5 bg-gray-200 dark:bg-gray-700 rounded w-32 mb-4"></div>
          <div class="space-y-3">
            <div v-for="j in 3" :key="j" class="flex items-center gap-3">
              <div class="h-3 bg-gray-200 dark:bg-gray-700 rounded flex-1"></div>
              <div class="h-3 bg-gray-200 dark:bg-gray-700 rounded w-8"></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Charts Skeleton -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 animate-pulse">
        <div class="h-6 bg-gray-200 dark:bg-gray-700 rounded w-40 mb-6"></div>
        <div class="space-y-3">
          <div v-for="i in 5" :key="i" class="flex items-center gap-3">
            <div class="h-4 bg-gray-200 dark:bg-gray-700 rounded w-24"></div>
            <div class="h-6 bg-gray-200 dark:bg-gray-700 rounded" :style="{ width: (100 - i * 15) + '%' }"></div>
          </div>
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
      <button
        @click="refetch"
        class="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
      >
        Retry
      </button>
    </div>

    <!-- Main Content -->
    <div v-else-if="stats" class="space-y-8">
      <!-- Stats Cards -->
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow border-l-4 border-blue-500">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-gray-600 dark:text-gray-400 text-sm font-medium uppercase tracking-wide">
                Total Documents
              </p>
              <p class="text-4xl font-bold text-blue-600 dark:text-blue-400 mt-2">{{ stats.total_documents }}</p>
            </div>
            <div class="p-3 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
              <svg class="w-6 h-6 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
          </div>
        </div>

        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow border-l-4 border-green-500">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-gray-600 dark:text-gray-400 text-sm font-medium uppercase tracking-wide">Letters</p>
              <p class="text-4xl font-bold text-green-600 dark:text-green-400 mt-2">
                {{ stats.by_doc_type.letter || 0 }}
              </p>
            </div>
            <div class="p-3 bg-green-100 dark:bg-green-900/30 rounded-lg">
              <svg class="w-6 h-6 text-green-600 dark:text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
            </div>
          </div>
        </div>

        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow border-l-4 border-purple-500">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-gray-600 dark:text-gray-400 text-sm font-medium uppercase tracking-wide">
                Newspaper Articles
              </p>
              <p class="text-4xl font-bold text-purple-600 dark:text-purple-400 mt-2">
                {{ stats.by_doc_type.newspaper_article || 0 }}
              </p>
            </div>
            <div class="p-3 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
              <svg class="w-6 h-6 text-purple-600 dark:text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" />
              </svg>
            </div>
          </div>
        </div>

        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow border-l-4 border-yellow-500">
          <div class="flex items-center justify-between">
            <div>
              <p class="text-gray-600 dark:text-gray-400 text-sm font-medium uppercase tracking-wide">
                Flagged for Review
              </p>
              <p class="text-4xl font-bold text-yellow-600 dark:text-yellow-400 mt-2">{{ stats.flagged_count }}</p>
            </div>
            <div class="p-3 bg-yellow-100 dark:bg-yellow-900/30 rounded-lg">
              <svg class="w-6 h-6 text-yellow-600 dark:text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
            </div>
          </div>
        </div>
      </div>

      <!-- Breakdowns -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <BreakdownCard
          title="By Document Type"
          :data="stats.by_doc_type"
          :color-map="{
            letter: '#10b981',
            newspaper_article: '#8b5cf6',
            other: '#6b7280',
            unknown: '#f59e0b'
          }"
        />
        <BreakdownCard
          title="By Content Type"
          :data="stats.by_content_type"
          :color-map="{
            handwritten: '#3b82f6',
            typed: '#10b981',
            mixed: '#8b5cf6'
          }"
        />
        <BreakdownCard
          title="By Language"
          :data="stats.by_language"
          :color-map="{
            nl: '#f97316',
            en: '#3b82f6',
            de: '#eab308',
            fr: '#ec4899'
          }"
        />
      </div>

      <!-- Charts Section -->
      <div class="space-y-6">
        <!-- Top Topics -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
          <h3 class="text-xl font-semibold mb-4 text-gray-800 dark:text-white">Top 10 Topics</h3>
          <div v-if="stats.top_topics.length > 0">
            <HorizontalBarChart
              :data="stats.top_topics.slice(0, 10)"
              color="#3b82f6"
              :height="350"
            />
          </div>
          <div v-else class="text-center py-12 text-gray-500 dark:text-gray-400">
            <svg class="w-12 h-12 mx-auto mb-3 text-gray-300 dark:text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
            </svg>
            <p>No topics extracted yet</p>
          </div>
        </div>

        <!-- Top People -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
          <h3 class="text-xl font-semibold mb-4 text-gray-800 dark:text-white">Top 10 People Mentioned</h3>
          <div v-if="filteredPeople.length > 0">
            <HorizontalBarChart
              :data="filteredPeople"
              color="#10b981"
              :height="350"
            />
          </div>
          <div v-else class="text-center py-12 text-gray-500 dark:text-gray-400">
            <svg class="w-12 h-12 mx-auto mb-3 text-gray-300 dark:text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
            <p>No people extracted yet</p>
          </div>
        </div>

        <!-- Top Locations -->
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
          <h3 class="text-xl font-semibold mb-4 text-gray-800 dark:text-white">Top 10 Locations</h3>
          <div v-if="filteredLocations.length > 0">
            <HorizontalBarChart
              :data="filteredLocations"
              color="#8b5cf6"
              :height="350"
            />
          </div>
          <div v-else class="text-center py-12 text-gray-500 dark:text-gray-400">
            <svg class="w-12 h-12 mx-auto mb-3 text-gray-300 dark:text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            <p>No locations extracted yet</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useStats } from '@/composables/useData'
import HorizontalBarChart from '@/components/HorizontalBarChart.vue'
import BreakdownCard from '@/components/BreakdownCard.vue'

const { stats, loading, error, refetch } = useStats()

// Filter out invalid entries (null, empty strings, etc.)
const filteredPeople = computed(() => {
  if (!stats.value) return []
  return stats.value.top_people
    .filter((p) => p.name && p.name !== 'null' && p.name.trim() !== '')
    .slice(0, 10)
})

const filteredLocations = computed(() => {
  if (!stats.value) return []
  return stats.value.top_locations
    .filter((l) => l.name && l.name !== 'null' && l.name.trim() !== '')
    .slice(0, 10)
})
</script>
