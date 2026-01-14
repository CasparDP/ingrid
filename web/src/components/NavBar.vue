<template>
  <nav class="bg-white shadow-md">
    <div class="container mx-auto px-4">
      <div class="flex items-center justify-between h-16">
        <div class="flex items-center">
          <h1 class="text-2xl font-bold text-gray-800">Ingrid Archive</h1>
        </div>
        <div class="flex space-x-4">
          <RouterLink
            to="/"
            class="px-3 py-2 rounded-md text-sm font-medium"
            :class="isActive('/') ? 'bg-blue-500 text-white' : 'text-gray-700 hover:bg-gray-100'"
          >
            Dashboard
          </RouterLink>
          <RouterLink
            to="/documents"
            class="px-3 py-2 rounded-md text-sm font-medium"
            :class="isActive('/documents') ? 'bg-blue-500 text-white' : 'text-gray-700 hover:bg-gray-100'"
          >
            Documents
          </RouterLink>
          <RouterLink
            to="/network"
            class="px-3 py-2 rounded-md text-sm font-medium"
            :class="isActive('/network') ? 'bg-blue-500 text-white' : 'text-gray-700 hover:bg-gray-100'"
          >
            Network
          </RouterLink>
        </div>
      </div>
    </div>
  </nav>
</template>

<script setup lang="ts">
import { useRoute } from 'vue-router'

const route = useRoute()

function isActive(path: string) {
  return route.path === path
}
</script>
