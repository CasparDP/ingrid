<template>
  <Teleport to="body">
    <Transition name="modal">
      <div v-if="show" class="fixed inset-0 z-50 overflow-y-auto" @click.self="$emit('close')">
        <!-- Backdrop -->
        <div class="fixed inset-0 bg-black/50 transition-opacity" @click="$emit('close')"></div>

        <!-- Modal -->
        <div class="flex min-h-full items-center justify-center p-4">
          <div class="relative bg-white rounded-xl shadow-2xl w-full max-w-3xl max-h-[90vh] overflow-hidden" @click.stop>
            <!-- Header -->
            <div class="sticky top-0 bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
              <h3 class="text-xl font-semibold text-gray-900 truncate pr-4">
                {{ document?.filename }}
              </h3>
              <button
                @click="$emit('close')"
                class="p-2 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 transition-colors"
              >
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <!-- Content -->
            <div class="px-6 py-5 overflow-y-auto" style="max-height: calc(90vh - 140px);">
              <div v-if="document" class="space-y-6">
                <!-- Type and Status Badges -->
                <div class="flex flex-wrap gap-2">
                  <span :class="docTypeBadgeClass">
                    {{ formatLabel(document.doc_type) }}
                  </span>
                  <span :class="contentTypeBadgeClass">
                    {{ formatLabel(document.content_type) }}
                  </span>
                  <span v-for="lang in document.languages" :key="lang"
                        class="px-3 py-1 text-sm font-medium rounded-full bg-gray-100 text-gray-700">
                    {{ languageName(lang) }}
                  </span>
                  <span v-if="document.flagged_for_review"
                        class="px-3 py-1 text-sm font-medium rounded-full bg-yellow-100 text-yellow-800">
                    Flagged for Review
                  </span>
                </div>

                <!-- Metadata Grid -->
                <div class="grid grid-cols-2 gap-4">
                  <div v-if="document.date" class="bg-gray-50 rounded-lg p-4">
                    <dt class="text-sm font-medium text-gray-500 mb-1">Date</dt>
                    <dd class="text-gray-900">{{ formatDate(document.date) }}</dd>
                  </div>
                  <div v-if="document.sender" class="bg-gray-50 rounded-lg p-4">
                    <dt class="text-sm font-medium text-gray-500 mb-1">Sender</dt>
                    <dd class="text-gray-900">{{ document.sender }}</dd>
                  </div>
                  <div v-if="document.recipient" class="bg-gray-50 rounded-lg p-4">
                    <dt class="text-sm font-medium text-gray-500 mb-1">Recipient</dt>
                    <dd class="text-gray-900">{{ document.recipient }}</dd>
                  </div>
                  <div v-if="document.location" class="bg-gray-50 rounded-lg p-4">
                    <dt class="text-sm font-medium text-gray-500 mb-1">Location</dt>
                    <dd class="text-gray-900">{{ document.location }}</dd>
                  </div>
                </div>

                <!-- Summary (Original) -->
                <div v-if="document.summary">
                  <h4 class="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-2">
                    Summary
                    <span v-if="document.summary_language" class="text-xs font-normal text-gray-500">
                      ({{ languageName(document.summary_language) }})
                    </span>
                  </h4>
                  <div class="bg-blue-50 border border-blue-100 rounded-lg p-4">
                    <p class="text-gray-700 leading-relaxed">{{ document.summary }}</p>
                  </div>
                </div>

                <!-- Summary (English) -->
                <div v-if="document.summary_english && document.summary_english !== document.summary">
                  <h4 class="text-sm font-semibold text-gray-700 mb-2">
                    Summary (English)
                  </h4>
                  <div class="bg-green-50 border border-green-100 rounded-lg p-4">
                    <p class="text-gray-700 leading-relaxed">{{ document.summary_english }}</p>
                  </div>
                </div>

                <!-- Topics -->
                <div v-if="document.topics.length > 0">
                  <h4 class="text-sm font-semibold text-gray-700 mb-2">Topics</h4>
                  <div class="flex flex-wrap gap-2">
                    <span v-for="topic in document.topics" :key="topic"
                          class="px-3 py-1.5 bg-blue-100 text-blue-800 rounded-lg text-sm">
                      {{ topic }}
                    </span>
                  </div>
                </div>

                <!-- People Mentioned -->
                <div v-if="document.people_mentioned.length > 0">
                  <h4 class="text-sm font-semibold text-gray-700 mb-2">People Mentioned</h4>
                  <div class="flex flex-wrap gap-2">
                    <span v-for="person in document.people_mentioned" :key="person"
                          class="px-3 py-1.5 bg-purple-100 text-purple-800 rounded-lg text-sm">
                      {{ person }}
                    </span>
                  </div>
                </div>

                <!-- Manual Tags -->
                <div v-if="document.manual_tags.length > 0">
                  <h4 class="text-sm font-semibold text-gray-700 mb-2">Tags</h4>
                  <div class="flex flex-wrap gap-2">
                    <span v-for="tag in document.manual_tags" :key="tag"
                          class="px-3 py-1.5 bg-gray-200 text-gray-700 rounded-lg text-sm">
                      {{ tag }}
                    </span>
                  </div>
                </div>

                <!-- Document ID -->
                <div class="pt-4 border-t border-gray-200">
                  <p class="text-xs text-gray-400">
                    Document ID: {{ document.id }}
                    <span v-if="document.created_at"> | Processed: {{ formatDate(document.created_at) }}</span>
                  </p>
                </div>
              </div>
            </div>

            <!-- Footer -->
            <div class="sticky bottom-0 bg-gray-50 border-t border-gray-200 px-6 py-4">
              <button
                @click="$emit('close')"
                class="w-full px-4 py-2 bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition-colors font-medium"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { Document } from '@/types'

interface Props {
  show: boolean
  document: Document | null
}

const props = defineProps<Props>()
defineEmits<{
  close: []
}>()

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

function formatLabel(text: string): string {
  return text
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

function formatDate(dateStr: string): string {
  try {
    const date = new Date(dateStr)
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    })
  } catch {
    return dateStr
  }
}

const docTypeBadgeClass = computed(() => {
  const base = 'px-3 py-1 text-sm font-medium rounded-full'
  switch (props.document?.doc_type) {
    case 'letter':
      return `${base} bg-blue-100 text-blue-800`
    case 'newspaper_article':
      return `${base} bg-green-100 text-green-800`
    default:
      return `${base} bg-gray-100 text-gray-700`
  }
})

const contentTypeBadgeClass = computed(() => {
  const base = 'px-3 py-1 text-sm font-medium rounded-full'
  switch (props.document?.content_type) {
    case 'handwritten':
      return `${base} bg-orange-100 text-orange-800`
    case 'typed':
      return `${base} bg-cyan-100 text-cyan-800`
    case 'mixed':
      return `${base} bg-pink-100 text-pink-800`
    default:
      return `${base} bg-gray-100 text-gray-700`
  }
})
</script>

<style scoped>
.modal-enter-active,
.modal-leave-active {
  transition: opacity 0.2s ease;
}

.modal-enter-from,
.modal-leave-to {
  opacity: 0;
}

.modal-enter-active .relative,
.modal-leave-active .relative {
  transition: transform 0.2s ease;
}

.modal-enter-from .relative,
.modal-leave-to .relative {
  transform: scale(0.95);
}
</style>
