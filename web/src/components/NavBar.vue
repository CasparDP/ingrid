<template>
  <nav class="bg-white dark:bg-gray-800 shadow-md transition-colors duration-200">
    <div class="container mx-auto px-4">
      <div class="flex items-center justify-between h-16">
        <!-- Logo -->
        <div class="flex items-center gap-3">
          <div class="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
            <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <h1 class="text-xl font-bold text-gray-800 dark:text-white">Ingrid Archive</h1>
        </div>

        <!-- Navigation Links -->
        <div class="flex items-center gap-2">
          <RouterLink
            to="/"
            class="px-3 py-2 rounded-lg text-sm font-medium transition-colors"
            :class="isActive('/')
              ? 'bg-blue-500 text-white'
              : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'"
          >
            Dashboard
          </RouterLink>
          <RouterLink
            to="/documents"
            class="px-3 py-2 rounded-lg text-sm font-medium transition-colors"
            :class="isActive('/documents')
              ? 'bg-blue-500 text-white'
              : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'"
          >
            Documents
          </RouterLink>
          <RouterLink
            to="/network"
            class="px-3 py-2 rounded-lg text-sm font-medium transition-colors"
            :class="isActive('/network')
              ? 'bg-blue-500 text-white'
              : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'"
          >
            Network
          </RouterLink>

          <!-- Divider -->
          <div class="w-px h-6 bg-gray-300 dark:bg-gray-600 mx-2"></div>

          <!-- Dark Mode Toggle -->
          <button
            @click="$emit('toggleDark')"
            class="p-2 rounded-lg text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            :title="isDark ? 'Switch to light mode' : 'Switch to dark mode'"
          >
            <!-- Sun icon (shown in dark mode) -->
            <svg v-if="isDark" class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
            </svg>
            <!-- Moon icon (shown in light mode) -->
            <svg v-else class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  </nav>
</template>

<script setup lang="ts">
import { useRoute } from 'vue-router'

defineProps<{
  isDark: boolean
}>()

defineEmits<{
  toggleDark: []
}>()

const route = useRoute()

function isActive(path: string) {
  return route.path === path
}
</script>
