<template>
  <div>
    <h2 class="text-3xl font-bold mb-6 text-gray-900 dark:text-white">Network Graph</h2>

    <!-- Loading State -->
    <div v-if="loading" class="space-y-4">
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-4 animate-pulse">
        <div class="h-6 bg-gray-200 dark:bg-gray-700 rounded w-1/3 mb-4"></div>
        <div class="h-4 bg-gray-200 dark:bg-gray-700 rounded w-2/3"></div>
      </div>
      <div class="bg-gray-100 dark:bg-gray-800 rounded-lg animate-pulse" style="height: 600px;">
        <div class="h-full flex items-center justify-center">
          <div class="text-gray-400 dark:text-gray-500">Loading network visualization...</div>
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
    <div v-else-if="network">
      <!-- Stats Bar -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow p-4 mb-6 transition-colors">
        <div class="flex flex-wrap gap-6">
          <div class="flex items-center gap-3">
            <div class="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
              <svg class="w-5 h-5 text-blue-600 dark:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <div>
              <p class="text-2xl font-bold text-gray-900 dark:text-white">{{ network.nodes.length }}</p>
              <p class="text-sm text-gray-500 dark:text-gray-400">Documents</p>
            </div>
          </div>

          <div class="flex items-center gap-3">
            <div class="p-2 bg-green-100 dark:bg-green-900/30 rounded-lg">
              <svg class="w-5 h-5 text-green-600 dark:text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
              </svg>
            </div>
            <div>
              <p class="text-2xl font-bold text-gray-900 dark:text-white">{{ network.edges.length }}</p>
              <p class="text-sm text-gray-500 dark:text-gray-400">Connections</p>
            </div>
          </div>

          <div class="flex items-center gap-3">
            <div class="p-2 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
              <svg class="w-5 h-5 text-purple-600 dark:text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
              </svg>
            </div>
            <div>
              <p class="text-2xl font-bold text-gray-900 dark:text-white">{{ uniqueTopicsCount }}</p>
              <p class="text-sm text-gray-500 dark:text-gray-400">Unique Topics</p>
            </div>
          </div>

          <div class="ml-auto flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>Click a node to view document details</span>
          </div>
        </div>
      </div>

      <!-- Network Graph -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow overflow-hidden transition-colors">
        <NetworkGraph
          :nodes="network.nodes"
          :edges="network.edges"
          :height="600"
          @node-click="handleNodeClick"
        />
      </div>

      <!-- Shared Topics Info -->
      <div v-if="network.edges.length > 0" class="mt-6 bg-white dark:bg-gray-800 rounded-lg shadow p-4 transition-colors">
        <h3 class="text-lg font-semibold text-gray-800 dark:text-white mb-4">Most Connected Topics</h3>
        <div class="flex flex-wrap gap-2">
          <span
            v-for="topic in topSharedTopics"
            :key="topic.name"
            class="px-3 py-1.5 bg-purple-100 dark:bg-purple-900/50 text-purple-800 dark:text-purple-300 rounded-lg text-sm"
          >
            {{ topic.name }}
            <span class="text-purple-500 dark:text-purple-400 ml-1">({{ topic.count }})</span>
          </span>
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
import { ref, computed } from 'vue'
import { useNetwork, useDocuments } from '@/composables/useData'
import NetworkGraph from '@/components/NetworkGraph.vue'
import DocumentDetailModal from '@/components/DocumentDetailModal.vue'
import type { NetworkNode, Document } from '@/types'

const { network, loading, error } = useNetwork()
const { documents } = useDocuments()

// Modal state
const showModal = ref(false)
const selectedDocument = ref<Document | null>(null)

// Calculate unique topics across all nodes
const uniqueTopicsCount = computed(() => {
  if (!network.value) return 0
  const topics = new Set<string>()
  network.value.nodes.forEach(node => {
    node.topics.forEach(t => topics.add(t))
  })
  return topics.size
})

// Get most shared topics from edges
const topSharedTopics = computed(() => {
  if (!network.value || network.value.edges.length === 0) return []

  const topicCounts: Record<string, number> = {}
  network.value.edges.forEach(edge => {
    edge.shared_topics.forEach(topic => {
      topicCounts[topic] = (topicCounts[topic] || 0) + 1
    })
  })

  return Object.entries(topicCounts)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10)
})

// Handle node click
function handleNodeClick(node: NetworkNode) {
  // Find the full document data
  const doc = documents.value.find(d => d.id === node.id)
  if (doc) {
    selectedDocument.value = doc
    showModal.value = true
  }
}

function closeModal() {
  showModal.value = false
  selectedDocument.value = null
}
</script>
