package a25

import (
	"context"
	"fmt"
	"time"

	"github.com/lordtatty/a25/memory"
	"github.com/lordtatty/a25/plan"
	"github.com/lordtatty/a25/react"
	"github.com/lordtatty/a25/reflect"
	openai "github.com/sashabaranov/go-openai"
)

type Modules struct {
	Planner   *plan.Planner
	React     *react.Reactor
	Reflector *reflect.Reflector
}

// Agent represents an individual with memories and traits.
type Agent struct {
	Name        string
	Traits      string
	Description string
	Memory      memory.MemoryStream
	Client      OpenAIClient
	CurrentPlan plan.Plan
	Status      AgentStatus
	Modules     Modules
}

// AgentStatus represents the agent's current state.
type AgentStatus struct {
	CurrentTask     string
	CurrentLocation string
}

type OpenAIClient interface {
	CreateChatCompletion(context.Context, openai.ChatCompletionRequest) (*openai.ChatCompletionResponse, error)
	CreateEmbeddings(context.Context, openai.EmbeddingRequestConverter) (*openai.EmbeddingResponse, error)
}

// NewAgent creates a new agent instance.
func NewAgent(name, traits, description string, client OpenAIClient) *Agent {
	m := Modules{
		Planner:   &plan.Planner{Client: client},
		React:     &react.Reactor{Client: client},
		Reflector: &reflect.Reflector{Client: client},
	}
	mem := memory.MemoryStream{Client: client}
	return &Agent{
		Name:        name,
		Traits:      traits,
		Description: description,
		Memory:      mem,
		Client:      client,
		CurrentPlan: plan.Plan{},
		Modules:     m,
	}
}

// AddMemory adds a memory to the agent's memory stream.
func (a *Agent) AddMemory(description string, importance float64) {
	a.Memory.AddMemory(description)
}

// Reflect allows the agent to generate reflections.
func (a *Agent) Reflect() error {
	m := a.Memory.GetRecentMemories(100)
	return a.Modules.Reflector.Reflect(m, &a.Memory)
}

// PlanDay generates a high-level plan for the agent's day.
func (a *Agent) PlanDay(currentTime time.Time) error {
	summary, err := a.GenerateSummary()
	if err != nil {
		return fmt.Errorf("failed to generate agent summary: %w", err)
	}
	newActions, err := a.Modules.Planner.PlanDay(currentTime, summary)
	if err != nil {
		return fmt.Errorf("current plan failed to plan: %w", err)
	}
	a.CurrentPlan.SetActions(newActions)
	// Add the plan to the memory stream.
	a.Memory.AddMemory("Generated plan for the day.")
	return nil
}

// GenerateSummary creates a summary of the agent's state.
func (a *Agent) GenerateSummary() (string, error) {
	// You can customize this method to generate a summary based on the agent's traits, recent memories, etc.
	return fmt.Sprintf("Name: %s\nTraits: %s\nDescription: %s", a.Name, a.Traits, a.Description), nil
}

// PerceiveAndReact processes observations and decides whether to react.
func (a *Agent) PerceiveAndReact(observation string, currentTime time.Time) error {
	// Add the observation to memory.
	a.Memory.AddMemory(observation) // Adjust importance as needed.
	context := fmt.Sprintf("Agent: %s\nTraits: %s\nDescription: %s\nCurrent Task: %s", a.Name, a.Traits, a.Description, a.Status.CurrentTask)
	shouldReact, reactReason, err := a.Modules.React.ToObservation(observation, context, currentTime)
	if err != nil {
		return fmt.Errorf("failed to perceive and react: %w", err)
	}
	if !shouldReact {
		a.Memory.AddMemory(fmt.Sprintf("%s decided not to react to: '%s'", a.Name, observation))
		return nil
	}
	// Update the plan based on the reaction.
	err = a.UpdatePlan(reactReason, currentTime)
	if err != nil {
		return fmt.Errorf("failed to update plan: %w", err)
	}
	// Add reaction to memory.
	a.Memory.AddMemory(fmt.Sprintf("%s decided to react to: '%s', because: %s", a.Name, observation, reactReason))
	return nil
}

// UpdatePlan modifies the agent's plan based on the reaction.
func (a *Agent) UpdatePlan(reaction string, currentTime time.Time) error {
	// You can implement logic to adjust the plan.
	// For simplicity, let's prepend a new action.
	newAction := plan.Action{
		Description: reaction,
		StartTime:   currentTime,
		// Set Duration and Location as needed.
	}
	a.CurrentPlan.AddAction(newAction)
	return nil
}

func (a *Agent) SelectTask() {
	a.CurrentPlan.NextAction()
	a.Status.CurrentTask = a.CurrentPlan.NextAction().Description
	a.Memory.AddMemory("Started Task: " + a.Status.CurrentTask)
}
