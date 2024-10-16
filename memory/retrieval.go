package memory

import (
	"math"
	"sort"
	"time"
)

// RetrievedMemory pairs a memory with its retrieval score.
type RetrievedMemory struct {
	Memory MemoryObject
	Score  float32
}

// RetrieveMemories retrieves relevant memories based on a query.
func (ms *MemoryStream) RetrieveMemories(query string) ([]RetrievedMemory, error) {
	// Compute the embedding for the query.
	queryEmbedding, err := getEmbedding(query, ms.Client)
	if err != nil {
		return nil, err
	}

	var retrieved []RetrievedMemory
	for i, memory := range ms.Memories {
		// Compute the embedding for the memory.
		memoryEmbedding, err := getEmbedding(memory.Description, ms.Client)
		if err != nil {
			return nil, err
		}
		// Compute relevance as cosine similarity.
		relevance := cosineSimilarity(queryEmbedding, memoryEmbedding)
		// Compute recency score.
		hoursSinceAccess := time.Since(memory.LastAccessedTime).Hours()
		recencyScore := float32(math.Exp(-hoursSinceAccess / 24.0)) // Decay over one day.
		// Normalize importance to [0,1].
		importanceScore := memory.Importance / 10.0 // Assuming importance is between 0 and 10.
		// Total score.
		totalScore := relevance + recencyScore + float32(importanceScore)

		retrieved = append(retrieved, RetrievedMemory{
			Memory: memory,
			Score:  totalScore,
		})
		// Update last accessed time.
		ms.Memories[i].LastAccessedTime = time.Now()
	}

	// Sort retrieved memories by score in descending order.
	sort.Slice(retrieved, func(i, j int) bool {
		return retrieved[i].Score > retrieved[j].Score
	})

	return retrieved, nil
}

// cosineSimilarity computes the cosine similarity between two vectors.
func cosineSimilarity(a, b []float32) float32 {
	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0.0
	}
	return dotProduct / float32(math.Sqrt(float64(normA))*math.Sqrt(float64(normB)))
}
