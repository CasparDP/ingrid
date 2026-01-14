<template>
  <div>
    <h2 class="text-3xl font-bold mb-6">Network Graph</h2>

    <div v-if="loading" class="text-center py-12">
      <p class="text-gray-600">Loading network data...</p>
    </div>

    <div v-else-if="error" class="bg-red-50 border border-red-200 rounded-lg p-4">
      <p class="text-red-800">Error: {{ error }}</p>
    </div>

    <div v-else-if="network">
      <div class="bg-white rounded-lg shadow p-4 mb-4">
        <p class="text-gray-600">
          Network graph with {{ network.nodes.length }} documents and
          {{ network.edges.length }} connections (shared topics).
        </p>
        <p class="text-sm text-gray-500 mt-2">
          D3 visualization will be implemented in Phase 6d.
        </p>
      </div>

      <!-- Placeholder for D3 visualization -->
      <div class="bg-gray-100 rounded-lg flex items-center justify-center" style="height: 600px;">
        <p class="text-gray-500 text-lg">Network visualization coming in Phase 6d</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useNetwork } from '@/composables/useData'

const { network, loading, error } = useNetwork()
</script>
