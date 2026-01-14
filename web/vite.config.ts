import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'

// Auto-detect GitHub Pages base path
const base = process.env.NODE_ENV === 'production' ? '/ingrid/' : '/'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  base,
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
  }
})
