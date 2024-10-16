package react

import (
	"context"
	"fmt"
	"strings"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

type OpenAIClient interface {
	CreateChatCompletion(context.Context, openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error)
}

// React encapsulates the perceive and reaction capabilities of an agent.
type Reactor struct {
	Client OpenAIClient
}

// DecideReaction determines if the agent should react to the observation.
func (r *Reactor) ToObservation(observation, contextSummary string, currentTime time.Time) (bool, string, error) {
	sysPrompt := `Based on the agent's context and observation, determine if the agent should react. 
Respond with 'Yes' or 'No' and provide a brief explanation if 'Yes'.`

	usrPrompt := fmt.Sprintf(`Agent Context:
%s
Observation:
%s`, contextSummary, observation)

	resp, err := r.Client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		Model: openai.GPT4oMini,
		Messages: []openai.ChatCompletionMessage{
			{Role: "system", Content: sysPrompt},
			{Role: "user", Content: usrPrompt},
		},
		Temperature: 1,
	})
	if err != nil {
		return false, "", err
	}

	response := resp.Choices[0].Message.Content
	response = strings.TrimSpace(strings.ToLower(response))

	if strings.HasPrefix(response, "yes") {
		// Extract the reaction explanation.
		reaction := strings.TrimPrefix(response, "yes")
		reaction = strings.TrimSpace(reaction)
		return true, reaction, nil
	}

	return false, "", nil
}
