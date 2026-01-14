<template>
  <div class="bg-white rounded-lg shadow p-6">
    <h3 class="text-lg font-semibold mb-4 text-gray-800">{{ title }}</h3>
    <div class="space-y-3">
      <div
        v-for="[key, value] in sortedItems"
        :key="key"
        class="flex items-center justify-between"
      >
        <div class="flex items-center flex-1">
          <span class="text-sm font-medium text-gray-700 capitalize">{{
            formatKey(key)
          }}</span>
        </div>
        <div class="flex items-center gap-3 ml-4">
          <div class="flex-1 bg-gray-200 rounded-full h-2 w-32">
            <div
              class="h-2 rounded-full transition-all duration-300"
              :style="{
                width: `${getPercentage(value)}%`,
                backgroundColor: getColor(key)
              }"
            ></div>
          </div>
          <span class="text-sm font-bold text-gray-900 w-8 text-right">{{
            value
          }}</span>
        </div>
      </div>
      <div v-if="sortedItems.length === 0" class="text-sm text-gray-500 italic">
        No data available
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface Props {
  title: string
  data: Record<string, number>
  colorMap?: Record<string, string>
}

const props = withDefaults(defineProps<Props>(), {
  colorMap: () => ({})
})

const defaultColors = [
  '#3b82f6', // blue-500
  '#10b981', // green-500
  '#8b5cf6', // purple-500
  '#f59e0b', // amber-500
  '#ef4444', // red-500
  '#06b6d4', // cyan-500
  '#ec4899' // pink-500
]

const sortedItems = computed(() => {
  return Object.entries(props.data)
    .filter(([, value]) => value > 0)
    .sort(([, a], [, b]) => b - a)
})

const maxValue = computed(() => {
  return Math.max(...Object.values(props.data), 1)
})

function getPercentage(value: number): number {
  return (value / maxValue.value) * 100
}

function getColor(key: string): string {
  const mappedColor = props.colorMap?.[key]
  if (mappedColor) {
    return mappedColor
  }
  const index = Object.keys(props.data).indexOf(key)
  const colorIndex = index % defaultColors.length
  return defaultColors[colorIndex] ?? '#3b82f6'
}

function formatKey(key: string): string {
  // Handle special cases
  if (key === 'newspaper_article') return 'Newspaper Article'
  if (key === 'unknown') return 'Unknown'

  // Capitalize and replace underscores
  return key
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}
</script>
