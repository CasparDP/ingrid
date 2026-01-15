<template>
  <div :class="{ 'dark': isDark }" class="min-h-screen">
    <div class="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-200">
      <NavBar :is-dark="isDark" @toggle-dark="toggleDark" />
      <main class="container mx-auto px-4 py-8">
        <RouterView />
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { RouterView } from 'vue-router'
import NavBar from '@/components/NavBar.vue'

const isDark = ref(false)

function toggleDark() {
  isDark.value = !isDark.value
  localStorage.setItem('ingrid-dark-mode', isDark.value ? 'dark' : 'light')
}

onMounted(() => {
  // Check for saved preference or system preference
  const saved = localStorage.getItem('ingrid-dark-mode')
  if (saved) {
    isDark.value = saved === 'dark'
  } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
    isDark.value = true
  }
})
</script>
