<template>
  <div>
    <h2 class="text-3xl font-bold mb-6 text-gray-900">Dashboard</h2>

    <div v-if="loading" class="text-center py-12">
      <p class="text-gray-600">Loading statistics...</p>
    </div>

    <div v-else-if="error" class="bg-red-50 border border-red-200 rounded-lg p-4">
      <p class="text-red-800">Error: {{ error }}</p>
      <button
        @click="refetch"
        class="mt-2 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 transition-colors"
      >
        Retry
      </button>
    </div>

    <div v-else-if="stats" class="space-y-8">
      <!-- Stats Cards -->
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div class="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
          <p class="text-gray-600 text-sm font-medium uppercase tracking-wide">
            Total Documents
          </p>
          <p class="text-4xl font-bold text-blue-600 mt-2">{{ stats.total_documents }}</p>
        </div>
        <div class="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
          <p class="text-gray-600 text-sm font-medium uppercase tracking-wide">Letters</p>
          <p class="text-4xl font-bold text-green-600 mt-2">
            {{ stats.by_doc_type.letter || 0 }}
          </p>
        </div>
        <div class="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
          <p class="text-gray-600 text-sm font-medium uppercase tracking-wide">
            Newspaper Articles
          </p>
          <p class="text-4xl font-bold text-purple-600 mt-2">
            {{ stats.by_doc_type.newspaper_article || 0 }}
          </p>
        </div>
        <div class="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
          <p class="text-gray-600 text-sm font-medium uppercase tracking-wide">
            Flagged for Review
          </p>
          <p class="text-4xl font-bold text-yellow-600 mt-2">{{ stats.flagged_count }}</p>
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
        <div class="bg-white rounded-lg shadow-md p-6">
          <h3 class="text-xl font-semibold mb-4 text-gray-800">Top 10 Topics</h3>
          <HorizontalBarChart
            :data="stats.top_topics.slice(0, 10)"
            color="#3b82f6"
            :height="350"
          />
        </div>

        <!-- Top People -->
        <div class="bg-white rounded-lg shadow-md p-6">
          <h3 class="text-xl font-semibold mb-4 text-gray-800">Top 10 People Mentioned</h3>
          <HorizontalBarChart
            :data="filteredPeople"
            color="#10b981"
            :height="350"
          />
        </div>

        <!-- Top Locations -->
        <div class="bg-white rounded-lg shadow-md p-6">
          <h3 class="text-xl font-semibold mb-4 text-gray-800">Top 10 Locations</h3>
          <HorizontalBarChart
            :data="filteredLocations"
            color="#8b5cf6"
            :height="350"
          />
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
