<template>
  <div class="network-graph-container relative" ref="containerRef">
    <!-- Legend -->
    <div class="absolute top-4 left-4 bg-white/90 backdrop-blur rounded-lg shadow-lg p-4 z-10">
      <h4 class="text-sm font-semibold text-gray-700 mb-3">Document Types</h4>
      <div class="space-y-2">
        <div class="flex items-center gap-2">
          <div class="w-4 h-4 rounded-full bg-blue-500"></div>
          <span class="text-sm text-gray-600">Letter</span>
        </div>
        <div class="flex items-center gap-2">
          <div class="w-4 h-4 rounded-full bg-green-500"></div>
          <span class="text-sm text-gray-600">Newspaper Article</span>
        </div>
        <div class="flex items-center gap-2">
          <div class="w-4 h-4 rounded-full bg-gray-400"></div>
          <span class="text-sm text-gray-600">Other</span>
        </div>
      </div>
      <div class="mt-4 pt-3 border-t border-gray-200">
        <p class="text-xs text-gray-500">Node size = connections</p>
        <p class="text-xs text-gray-500">Edge = shared topics (2+)</p>
      </div>
    </div>

    <!-- Controls -->
    <div class="absolute top-4 right-4 bg-white/90 backdrop-blur rounded-lg shadow-lg p-2 z-10 flex gap-2">
      <button
        @click="zoomIn"
        class="p-2 hover:bg-gray-100 rounded-lg transition-colors"
        title="Zoom In"
      >
        <svg class="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
        </svg>
      </button>
      <button
        @click="zoomOut"
        class="p-2 hover:bg-gray-100 rounded-lg transition-colors"
        title="Zoom Out"
      >
        <svg class="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10H7" />
        </svg>
      </button>
      <button
        @click="resetZoom"
        class="p-2 hover:bg-gray-100 rounded-lg transition-colors"
        title="Reset View"
      >
        <svg class="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
      </button>
    </div>

    <!-- SVG Container -->
    <svg ref="svgRef" class="w-full h-full"></svg>

    <!-- Tooltip -->
    <div
      v-if="tooltipData"
      class="absolute bg-gray-900 text-white text-sm rounded-lg px-3 py-2 shadow-xl pointer-events-none z-20"
      :style="{ left: tooltipData.x + 'px', top: tooltipData.y + 'px' }"
    >
      <p class="font-medium">{{ tooltipData.label }}</p>
      <p class="text-gray-300 text-xs">{{ formatLabel(tooltipData.docType) }}</p>
      <p v-if="tooltipData.topics.length" class="text-gray-400 text-xs mt-1">
        Topics: {{ tooltipData.topics.slice(0, 3).join(', ') }}
        <span v-if="tooltipData.topics.length > 3">...</span>
      </p>
    </div>

    <!-- Empty State Overlay -->
    <div
      v-if="!hasEdges"
      class="absolute inset-0 flex items-center justify-center bg-gray-50/80 z-5"
    >
      <div class="text-center p-8">
        <svg class="w-16 h-16 text-gray-300 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
        </svg>
        <p class="text-gray-500 text-lg mb-2">No Connections Found</p>
        <p class="text-gray-400 text-sm max-w-sm">
          Documents need to share 2 or more topics to be connected in the network.
        </p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch, onUnmounted, computed } from 'vue'
import * as d3 from 'd3'
import type { NetworkNode, NetworkEdge } from '@/types'

interface Props {
  nodes: NetworkNode[]
  edges: NetworkEdge[]
  height?: number
}

const props = withDefaults(defineProps<Props>(), {
  height: 600
})

const emit = defineEmits<{
  nodeClick: [node: NetworkNode]
}>()

const containerRef = ref<HTMLDivElement | null>(null)
const svgRef = ref<SVGSVGElement | null>(null)

// Tooltip state
const tooltipData = ref<{
  x: number
  y: number
  label: string
  docType: string
  topics: string[]
} | null>(null)

// D3 references
let simulation: d3.Simulation<d3.SimulationNodeDatum, undefined> | null = null
let svg: d3.Selection<SVGSVGElement, unknown, null, undefined> | null = null
let g: d3.Selection<SVGGElement, unknown, null, undefined> | null = null
let zoom: d3.ZoomBehavior<SVGSVGElement, unknown> | null = null

const hasEdges = computed(() => props.edges.length > 0)

// Node colors by document type
function getNodeColor(docType: string): string {
  switch (docType) {
    case 'letter':
      return '#3b82f6' // blue-500
    case 'newspaper_article':
      return '#22c55e' // green-500
    default:
      return '#9ca3af' // gray-400
  }
}

// Calculate node radius based on connections
function getNodeRadius(nodeId: string): number {
  const connections = props.edges.filter(e => e.source === nodeId || e.target === nodeId).length
  return Math.max(8, Math.min(25, 8 + connections * 3))
}

function formatLabel(text: string): string {
  return text
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ')
}

function renderGraph() {
  if (!svgRef.value || !containerRef.value) return

  // Clear previous graph
  d3.select(svgRef.value).selectAll('*').remove()

  const width = containerRef.value.clientWidth || 800
  const height = props.height

  // Setup SVG
  svg = d3.select(svgRef.value)
    .attr('width', width)
    .attr('height', height)

  // Create main group for zooming
  g = svg.append('g')

  // Setup zoom behavior
  zoom = d3.zoom<SVGSVGElement, unknown>()
    .scaleExtent([0.1, 4])
    .on('zoom', (event) => {
      g!.attr('transform', event.transform)
    })

  svg.call(zoom)

  // Prepare simulation data (deep copy to avoid mutating props)
  const simulationNodes = props.nodes.map(n => ({
    ...n,
    x: width / 2 + (Math.random() - 0.5) * 100,
    y: height / 2 + (Math.random() - 0.5) * 100
  }))

  const simulationEdges = props.edges.map(e => ({
    ...e,
    source: e.source,
    target: e.target
  }))

  // Create force simulation
  simulation = d3.forceSimulation(simulationNodes as d3.SimulationNodeDatum[])
    .force('link', d3.forceLink(simulationEdges)
      .id((d: any) => d.id)
      .distance(100)
      .strength(0.5))
    .force('charge', d3.forceManyBody().strength(-200))
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('collision', d3.forceCollide().radius((d: any) => getNodeRadius(d.id) + 5))

  // Draw edges
  const links = g.append('g')
    .attr('class', 'links')
    .selectAll('line')
    .data(simulationEdges)
    .enter()
    .append('line')
    .attr('stroke', '#cbd5e1')
    .attr('stroke-width', (d: any) => Math.max(1, d.weight))
    .attr('stroke-opacity', 0.6)

  // Draw nodes
  const nodes = g.append('g')
    .attr('class', 'nodes')
    .selectAll('circle')
    .data(simulationNodes)
    .enter()
    .append('circle')
    .attr('r', (d: any) => getNodeRadius(d.id))
    .attr('fill', (d: any) => getNodeColor(d.doc_type))
    .attr('stroke', '#fff')
    .attr('stroke-width', 2)
    .attr('cursor', 'pointer')
    .on('mouseover', function(event: MouseEvent, d: any) {
      d3.select(this)
        .transition()
        .duration(200)
        .attr('r', getNodeRadius(d.id) * 1.3)
        .attr('stroke-width', 3)

      // Highlight connected edges
      links
        .transition()
        .duration(200)
        .attr('stroke-opacity', (l: any) =>
          l.source.id === d.id || l.target.id === d.id ? 1 : 0.2
        )
        .attr('stroke', (l: any) =>
          l.source.id === d.id || l.target.id === d.id ? '#64748b' : '#cbd5e1'
        )

      // Show tooltip
      const rect = containerRef.value!.getBoundingClientRect()
      tooltipData.value = {
        x: event.clientX - rect.left + 10,
        y: event.clientY - rect.top - 10,
        label: d.label,
        docType: d.doc_type,
        topics: d.topics || []
      }
    })
    .on('mouseout', function(_event: MouseEvent, d: any) {
      d3.select(this)
        .transition()
        .duration(200)
        .attr('r', getNodeRadius(d.id))
        .attr('stroke-width', 2)

      // Reset edge highlighting
      links
        .transition()
        .duration(200)
        .attr('stroke-opacity', 0.6)
        .attr('stroke', '#cbd5e1')

      tooltipData.value = null
    })
    .on('click', (_event: MouseEvent, d: any) => {
      const originalNode = props.nodes.find(n => n.id === d.id)
      if (originalNode) {
        emit('nodeClick', originalNode)
      }
    })
    .call(d3.drag<SVGCircleElement, any>()
      .on('start', (event: any, d: any) => {
        if (!event.active) simulation!.alphaTarget(0.3).restart()
        d.fx = d.x
        d.fy = d.y
      })
      .on('drag', (event: any, d: any) => {
        d.fx = event.x
        d.fy = event.y
      })
      .on('end', (event: any, d: any) => {
        if (!event.active) simulation!.alphaTarget(0)
        d.fx = null
        d.fy = null
      })
    )

  // Add labels for larger nodes
  const labels = g.append('g')
    .attr('class', 'labels')
    .selectAll('text')
    .data(simulationNodes.filter((d: any) => getNodeRadius(d.id) > 12))
    .enter()
    .append('text')
    .text((d: any) => d.label.slice(0, 15) + (d.label.length > 15 ? '...' : ''))
    .attr('font-size', '10px')
    .attr('fill', '#374151')
    .attr('text-anchor', 'middle')
    .attr('dy', (d: any) => getNodeRadius(d.id) + 12)
    .attr('pointer-events', 'none')

  // Update positions on simulation tick
  simulation.on('tick', () => {
    links
      .attr('x1', (d: any) => d.source.x)
      .attr('y1', (d: any) => d.source.y)
      .attr('x2', (d: any) => d.target.x)
      .attr('y2', (d: any) => d.target.y)

    nodes
      .attr('cx', (d: any) => d.x)
      .attr('cy', (d: any) => d.y)

    labels
      .attr('x', (d: any) => d.x)
      .attr('y', (d: any) => d.y)
  })
}

// Zoom controls
function zoomIn() {
  if (svg && zoom) {
    svg.transition().duration(300).call(zoom.scaleBy, 1.5)
  }
}

function zoomOut() {
  if (svg && zoom) {
    svg.transition().duration(300).call(zoom.scaleBy, 0.67)
  }
}

function resetZoom() {
  if (svg && zoom) {
    svg.transition().duration(500).call(
      zoom.transform,
      d3.zoomIdentity.translate(0, 0).scale(1)
    )
  }
}

// Handle window resize
function handleResize() {
  renderGraph()
}

onMounted(() => {
  renderGraph()
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  if (simulation) {
    simulation.stop()
  }
})

watch(() => [props.nodes, props.edges], () => {
  renderGraph()
}, { deep: true })
</script>

<style scoped>
.network-graph-container {
  width: 100%;
  min-height: 600px;
  background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
  border-radius: 0.75rem;
  overflow: hidden;
}

svg {
  display: block;
}
</style>
