<template>
  <div>
    <h2 class="text-3xl font-bold mb-6">Documents</h2>

    <div v-if="loading" class="text-center py-12">
      <p class="text-gray-600">Loading documents...</p>
    </div>

    <div v-else-if="error" class="bg-red-50 border border-red-200 rounded-lg p-4">
      <p class="text-red-800">Error: {{ error }}</p>
    </div>

    <div v-else-if="documents.length === 0" class="text-center py-12">
      <p class="text-gray-600">No documents found</p>
    </div>

    <div v-else>
      <div class="bg-white rounded-lg shadow overflow-hidden">
        <table class="min-w-full divide-y divide-gray-200">
          <thead class="bg-gray-50">
            <tr>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Filename
              </th>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Type
              </th>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Date
              </th>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Topics
              </th>
            </tr>
          </thead>
          <tbody class="bg-white divide-y divide-gray-200">
            <tr v-for="doc in documents" :key="doc.id" class="hover:bg-gray-50">
              <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                {{ doc.filename }}
              </td>
              <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                {{ doc.doc_type }}
              </td>
              <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                {{ doc.date || 'N/A' }}
              </td>
              <td class="px-6 py-4 text-sm text-gray-500">
                <span v-if="doc.topics.length > 0" class="inline-flex flex-wrap gap-1">
                  <span v-for="topic in doc.topics.slice(0, 3)" :key="topic"
                        class="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
                    {{ topic }}
                  </span>
                  <span v-if="doc.topics.length > 3" class="text-gray-400 text-xs">
                    +{{ doc.topics.length - 3 }} more
                  </span>
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useDocuments } from '@/composables/useData'

const { documents, loading, error } = useDocuments()
</script>
