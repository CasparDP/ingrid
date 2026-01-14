<template>
  <div class="bar-chart-container">
    <svg ref="svgRef"></svg>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import * as d3 from 'd3'

interface ChartData {
  name: string
  count: number
}

interface Props {
  data: ChartData[]
  color?: string
  height?: number
}

const props = withDefaults(defineProps<Props>(), {
  color: '#3b82f6', // blue-500
  height: 400
})

const svgRef = ref<SVGSVGElement | null>(null)

function renderChart() {
  if (!svgRef.value || props.data.length === 0) return

  // Clear previous chart
  d3.select(svgRef.value).selectAll('*').remove()

  const margin = { top: 20, right: 30, bottom: 40, left: 120 }
  const width = svgRef.value.clientWidth || 600
  const height = props.height - margin.top - margin.bottom

  const svg = d3
    .select(svgRef.value)
    .attr('width', width)
    .attr('height', props.height)
    .append('g')
    .attr('transform', `translate(${margin.left},${margin.top})`)

  // Create scales
  const x = d3
    .scaleLinear()
    .domain([0, d3.max(props.data, (d) => d.count) || 0])
    .range([0, width - margin.left - margin.right])

  const y = d3
    .scaleBand()
    .domain(props.data.map((d) => d.name))
    .range([0, height])
    .padding(0.2)

  // Add bars
  svg
    .selectAll('.bar')
    .data(props.data)
    .enter()
    .append('rect')
    .attr('class', 'bar')
    .attr('x', 0)
    .attr('y', (d) => y(d.name) || 0)
    .attr('width', (d) => x(d.count))
    .attr('height', y.bandwidth())
    .attr('fill', props.color)
    .attr('rx', 4)
    .on('mouseover', function () {
      d3.select(this).attr('opacity', 0.8)
    })
    .on('mouseout', function () {
      d3.select(this).attr('opacity', 1)
    })

  // Add value labels
  svg
    .selectAll('.label')
    .data(props.data)
    .enter()
    .append('text')
    .attr('class', 'label')
    .attr('x', (d) => x(d.count) + 5)
    .attr('y', (d) => (y(d.name) || 0) + y.bandwidth() / 2)
    .attr('dy', '.35em')
    .attr('font-size', '12px')
    .attr('fill', '#6b7280')
    .text((d) => d.count)

  // Add Y axis (names)
  svg
    .append('g')
    .call(d3.axisLeft(y))
    .selectAll('text')
    .attr('font-size', '12px')
    .attr('fill', '#374151')

  // Add X axis
  svg
    .append('g')
    .attr('transform', `translate(0,${height})`)
    .call(d3.axisBottom(x).ticks(5))
    .selectAll('text')
    .attr('font-size', '12px')
    .attr('fill', '#6b7280')
}

onMounted(() => {
  renderChart()
  // Re-render on window resize
  window.addEventListener('resize', renderChart)
})

watch(() => props.data, renderChart, { deep: true })
</script>

<style scoped>
.bar-chart-container {
  width: 100%;
  overflow-x: auto;
}

svg {
  display: block;
}
</style>
