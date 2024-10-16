package reflect

import (
	"context"
	"fmt"
	"strings"

	"github.com/lordtatty/a25/memory"
	openai "github.com/sashabaranov/go-openai"
)

type OpenAIClient interface {
	CreateChatCompletion(context.Context, openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error)
}

type Reflector struct {
	Client OpenAIClient
}

// Reflect allows the agent to generate higher-level reflections.
func (r *Reflector) Reflect(memories []memory.MemoryObject, ms *memory.MemoryStream) error {
	// Concatenate memory descriptions.
	var memoryTexts []string
	for _, mem := range memories {
		memoryTexts = append(memoryTexts, mem.Description)
	}

	// Generate questions for reflection.
	questions, err := generateReflectionQuestions(memoryTexts, r.Client)
	if err != nil {
		return err
	}

	for _, question := range questions {
		// Retrieve relevant memories for the question.
		retrievedMemories, err := ms.RetrieveMemories(question)
		if err != nil {
			return err
		}

		// Generate insights based on retrieved memories.
		insights, err := generateInsights(question, retrievedMemories, r.Client)
		if err != nil {
			return err
		}

		for _, insight := range insights {
			ms.AddMemory(insight) // Assign calculated importance.
		}
	}

	return nil
}

// generateReflectionQuestions generates questions for reflection.
func generateReflectionQuestions(memories []string, client OpenAIClient) ([]string, error) {
	sysPrompt := "Given only the information provided below, what are 3 most salient high-level questions we can answer about the subjects in the statements?"
	usrPrompt := strings.Join(memories, "\n")

	// Call the language model.
	resp, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		Model: openai.GPT4oMini,
		Messages: []openai.ChatCompletionMessage{
			{Role: "system", Content: sysPrompt},
			{Role: "user", Content: usrPrompt},
		},
		Temperature: 1,
	})
	if err != nil {
		return nil, err
	}
	// Parse the response to extract questions.
	output := resp.Choices[0].Message.Content
	questions := parseQuestions(output)
	return questions, nil
}

// parseQuestions extracts questions from the model's output.
func parseQuestions(output string) []string {
	var questions []string
	lines := strings.Split(output, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// Remove numbering if present.
		if len(line) > 2 && (line[1] == '.' || line[1] == ')') {
			line = strings.TrimSpace(line[2:])
		}
		questions = append(questions, line)
	}
	return questions
}

// generateInsights generates insights based on the question and retrieved memories.
func generateInsights(question string, memories []memory.RetrievedMemory, client OpenAIClient) ([]string, error) {
	// Prepare prompt.
	var memoryTexts []string
	for idx, mem := range memories {
		memoryTexts = append(memoryTexts, fmt.Sprintf("%d. %s", idx+1, mem.Memory.Description))
	}
	sysPrompt := "What 5 high-level insights can you infer from the given statements? (example format: Insight (because of statements 1, 2, 3))"
	usrPrompt := fmt.Sprintf(`Statements about the question "%s":
%s`, question, strings.Join(memoryTexts, "\n"))

	// Call the language model.
	resp, err := client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
		Model: openai.GPT4oMini,
		Messages: []openai.ChatCompletionMessage{
			{Role: "system", Content: sysPrompt},
			{Role: "user", Content: usrPrompt},
		},
		Temperature: 1,
	})
	if err != nil {
		return nil, err
	}
	// Parse the response to extract insights.
	output := resp.Choices[0].Message.Content
	insights := parseInsights(output)
	return insights, nil
}

// parseInsights extracts insights from the model's output.
func parseInsights(output string) []string {
	var insights []string
	lines := strings.Split(output, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// Remove numbering if present.
		if len(line) > 2 && (line[1] == '.' || line[1] == ')') {
			line = strings.TrimSpace(line[2:])
		}
		// Extract the insight before the '('.
		idx := strings.Index(line, "(")
		if idx != -1 {
			line = line[:idx]
		}
		insights = append(insights, strings.TrimSpace(line))
	}
	return insights
}
