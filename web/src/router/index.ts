import { createRouter, createWebHistory } from 'vue-router'
import DashboardView from '@/views/DashboardView.vue'
import DocumentsView from '@/views/DocumentsView.vue'
import NetworkView from '@/views/NetworkView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'dashboard',
      component: DashboardView,
      meta: { title: 'Dashboard' }
    },
    {
      path: '/documents',
      name: 'documents',
      component: DocumentsView,
      meta: { title: 'Documents' }
    },
    {
      path: '/network',
      name: 'network',
      component: NetworkView,
      meta: { title: 'Network' }
    }
  ]
})

// Update page title on route change
router.afterEach((to) => {
  document.title = `Ingrid - ${to.meta.title || 'Document Archive'}`
})

export default router
