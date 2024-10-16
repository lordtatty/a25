package plan

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"
	openai "github.com/sashabaranov/go-openai"
)

type OpenAIClient interface {
	CreateChatCompletion(context.Context, openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error)
}

// Plan represents a high-level plan composed of actions.
type Plan struct {
	actions []Action
}

// Action represents a single action with time and location.
type Action struct {
	ID          string
	Description string
	Location    string
	StartTime   time.Time
	Duration    time.Duration
}

// Actions returns all actions in the plan.
func (p *Plan) Actions() []Action {
	return p.actions
}

// NextAction returns the next action in the plan based on the current time.
func (p *Plan) NextAction() *Action {
	return &p.actions[0]
}

// AddAction adds an action to the plan in chronological order.
func (p *Plan) AddAction(a Action) {
	a.ID = uuid.NewString()
	// Insert the action in the correct position to maintain chronological order
	i := sort.Search(len(p.actions), func(i int) bool {
		return p.actions[i].StartTime.After(a.StartTime)
	})
	p.actions = append(p.actions[:i], append([]Action{a}, p.actions[i:]...)...)
}

// SetActions sets the actions and ensures they are sorted in chronological order.
func (p *Plan) SetActions(actions []Action) {
	p.actions = actions
	sort.Slice(p.actions, func(i, j int) bool {
		return p.actions[i].StartTime.Before(p.actions[j].StartTime)
	})
}

// RemoveAction removes an action from the plan based on its ID.
func (p *Plan) RemoveAction(id string) error {
	for i, a := range p.actions {
		if a.ID == id {
			p.actions = slices.Delete(p.actions, i, i+1)
			return nil
		}
	}
	return fmt.Errorf("action id not found")
}

type Planner struct {
	Client OpenAIClient
}

// parsePlan converts the language model's output into a Plan struct.
func (p *Planner) parsePlan(planText string) ([]Action, error) {
	var actions []Action
	lines := strings.Split(planText, "\n")

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Skip the main title.
		if strings.HasPrefix(line, "**High-Level Plan for the Day") {
			continue
		}

		// Skip bullet points.
		if strings.HasPrefix(line, "-") {
			continue
		}

		// Remove asterisks from headings.
		line = strings.Trim(line, "*")

		// Check if line contains a time range and description.
		if !strings.Contains(line, ": ") || !strings.Contains(line, "-") {
			continue
		}

		// Example format: "8:00 AM - 9:00 AM: Morning Routine"
		// Split line into time block and action description.
		parts := strings.SplitN(line, ": ", 2)
		if len(parts) != 2 {
			continue
		}

		// Extract and parse the time range.
		timeRange := strings.TrimSpace(parts[0])
		timeParts := strings.Split(timeRange, " - ")
		if len(timeParts) != 2 {
			continue
		}

		// Parse the start time and end time.
		startTime, err := time.Parse("3:04 PM", strings.TrimSpace(timeParts[0]))
		if err != nil {
			continue
		}

		endTime, err := time.Parse("3:04 PM", strings.TrimSpace(timeParts[1]))
		if err != nil {
			continue
		}

		// Calculate the duration.
		duration := endTime.Sub(startTime)
		if duration <= 0 {
			continue
		}

		// Extract the action description.
		description := strings.TrimSpace(parts[1])

		// Create and add the action.
		action := Action{
			ID:          uuid.NewString(),
			Description: description,
			StartTime:   startTime,
			Duration:    duration,
		}
		actions = append(actions, action)
	}

	if len(actions) == 0 {
		return nil, errors.New("no actions found in plan")
	}

	return actions, nil
}

// PlanDay generates a high-level plan for the agent's day.
func (p *Planner) PlanDay(currentTime time.Time, agentSummary string) ([]Action, error) {
	// System prompt with detailed instructions for the model to follow.
	sysPrompt := `You are an expert planner. Your task is to generate a detailed, structured daily plan for the agent based on their summary. 
The plan should adhere to the following format:
1. The plan title should be formatted as: '**High-Level Plan for the Day: [Date]**'.
2. Include clear time blocks (e.g., '**8:00 AM - 9:00 AM: Morning Routine**').
3. Under each time block, provide a bullet list with specific activities. Each bullet should describe actions or goals within that time block.
4. Ensure consistency, clarity, and that the activities align with the agent's description and traits.`

	// User prompt with variable input.
	usrPrompt := fmt.Sprintf("Agent Summary:\n%s\nCurrent Time: %s", agentSummary, currentTime.Format("January 2, 2006"))

	// Call the language model.
	resp, err := p.Client.CreateChatCompletion(context.Background(), openai.ChatCompletionRequest{
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

	// Parse the response to extract the plan.
	actions, err := p.parsePlan(resp.Choices[0].Message.Content)
	if err != nil {
		return nil, err
	}

	return actions, nil
}
