// Stats for dashboard
export interface Stats {
  total_documents: number
  by_doc_type: Record<string, number>
  by_content_type: Record<string, number>
  by_language: Record<string, number>
  flagged_count: number
  top_topics: Array<{ name: string; count: number }>
  top_people: Array<{ name: string; count: number }>
  top_locations: Array<{ name: string; count: number }>
}

// Document metadata
export interface Document {
  id: string
  filename: string
  doc_type: string
  content_type: string
  languages: string[]
  date: string | null
  sender: string | null
  recipient: string | null
  location: string | null
  topics: string[]
  people_mentioned: string[]
  summary: string | null
  summary_english: string | null
  summary_language: string | null
  flagged_for_review: boolean
  manual_tags: string[]
  created_at: string | null
}

// Network graph data
export interface NetworkNode {
  id: string
  label: string
  doc_type: string
  content_type: string
  date: string | null
  topics: string[]
  people_mentioned: string[]
}

export interface NetworkEdge {
  source: string
  target: string
  weight: number
  shared_topics: string[]
}

export interface NetworkData {
  nodes: NetworkNode[]
  edges: NetworkEdge[]
}
