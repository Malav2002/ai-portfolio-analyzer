const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001'

export const analyzePortfolio = async (
  file: File, 
  onProgress?: (progress: number) => void
): Promise<any> => {
  const formData = new FormData()
  formData.append('image', file)

  try {
    onProgress?.(0)
    
    const response = await fetch(`${API_BASE_URL}/api/ai/analyze`, {
      method: 'POST',
      body: formData,
    })

    onProgress?.(50)

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(
        errorData.message || 
        errorData.error || 
        `HTTP error! status: ${response.status}`
      )
    }

    onProgress?.(75)
    const result = await response.json()
    onProgress?.(100)

    return result
  } catch (error: any) {
    console.error('Portfolio analysis error:', error)
    throw error
  }
}

export const checkBackendHealth = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`)
    return response.ok
  } catch {
    return false
  }
}
