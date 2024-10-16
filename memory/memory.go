package memory

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/sashabaranov/go-openai"
)

type OpenAIClient interface {
	CreateChatCompletion(context.Context, openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error)
	CreateEmbeddings(context.Context, openai.EmbeddingRequestConverter) (*openai.EmbeddingResponse, error)
}

// MemoryObject represents a single memory with associated metadata.
type MemoryObject struct {
	Description      string
	CreationTime     time.Time
	LastAccessedTime time.Time
	Importance       float64
	Embedding        []float32
}

// MemoryStream holds all memories of an agent.
type MemoryStream struct {
	Client   OpenAIClient
	Memories []MemoryObject
}

func NewStream(client OpenAIClient) *MemoryStream {
	return &MemoryStream{
		Client:   client,
		Memories: make([]MemoryObject, 0),
	}
}

// AddMemory adds a new memory to the memory stream.
func (ms *MemoryStream) AddMemory(description string) error {
	embed, err := getEmbedding(description, ms.Client)
	if err != nil {
		return fmt.Errorf("failed to get embedding: %w", err)
	}
	importance, err := rateImportance(description, ms.Client)
	if err != nil {
		return fmt.Errorf("failed to rate importance: %w", err)
	}
	memory := MemoryObject{
		Description:      description,
		CreationTime:     time.Now(),
		LastAccessedTime: time.Now(),
		Importance:       importance,
		Embedding:        embed,
	}
	ms.Memories = append(ms.Memories, memory)
	return nil
}

// rateImportance uses the language model to estimate the importance of a reflection.
func rateImportance(reflection string, client OpenAIClient) (float64, error) {
	sysPrompt := "On a scale of 1 to 10, where 1 is mundane (e.g., brushing teeth) and 10 is poignant (e.g., a life-changing event), rate the importance of the given reflection.  Output a single float value only, e.g., 7.5.  Include no other comment or opinion."
	resp, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		Model: openai.GPT4oMini,
		Messages: []openai.ChatCompletionMessage{
			{Role: "system", Content: sysPrompt},
			{Role: "user", Content: reflection},
		},
		Temperature: 1,
	})

	if err != nil {
		return 0, err
	}

	// Parse the model's response to extract the importance rating.
	rating, err := parseImportanceRating(resp.Choices[0].Message.Content)
	if err != nil {
		return 0, err
	}

	return rating, nil
}

// parseImportanceRating extracts the importance score from the response.
func parseImportanceRating(response string) (float64, error) {
	// Assume the response is a number from 1 to 10, parse it.
	rating, err := strconv.ParseFloat(strings.TrimSpace(response), 32)
	if err != nil {
		return 0, err
	}

	return rating, nil
}

// GetRecentMemories returns the N most recent memories.
func (ms *MemoryStream) GetRecentMemories(n int) []MemoryObject {
	if len(ms.Memories) < n {
		n = len(ms.Memories)
	}
	return ms.Memories[len(ms.Memories)-n:]
}

// getEmbedding retrieves the embedding vector for a given text.
func getEmbedding(text string, client OpenAIClient) ([]float32, error) {
	ctx := context.Background()
	resp, err := client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Input: []string{text},
		Model: openai.SmallEmbedding3,
	})
	if err != nil {
		return nil, err
	}
	return resp.Data[0].Embedding, nil
}
