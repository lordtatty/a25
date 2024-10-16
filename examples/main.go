// main.go
package main

import (
	"fmt"
	"os"
	"time"

	oailog "github.com/lordtatty/openai-log"
	openai "github.com/sashabaranov/go-openai"

	"github.com/lordtatty/a25"
)

func main() {
	// Set your OpenAI API key.
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("Please set the OPENAI_API_KEY environment variable.")
		return
	}

	// Initialize OpenAI client with logging and GPT4oMini model.
	client := &oailog.AI{
		Client:        openai.NewClient(apiKey),
		DefaultModel:  openai.GPT4oMini,
		EnableLogging: true,
	}
	defer client.Usage.PrintUsage()

	// Create an agent.
	agent := a25.NewAgent(
		"Klaus Mueller",
		"dedicated, curious, analytical",
		"Klaus Mueller is a college student studying urban planning. He is passionate about his research on gentrification in cities.",
		client,
	)

	// Add some initial memories.
	agent.AddMemory("Klaus Mueller is reading a book on gentrification.", 7.0)
	agent.AddMemory("Klaus Mueller is conversing with a librarian about his research project.", 6.5)
	agent.AddMemory("Klaus Mueller had lunch at the campus cafe.", 2.0)
	agent.AddMemory("Klaus Mueller attended a lecture on urban development.", 8.0)
	agent.AddMemory("Klaus Mueller met with Maria Lopez to discuss research.", 7.5)

	// ===== EXISTING FEATURE DEMONSTRATION =====
	// Agent reflects on recent experiences.
	fmt.Println("Agent is reflecting on recent experiences...")
	err := agent.Reflect()
	if err != nil {
		fmt.Println("Error during reflection:", err)
		return
	}

	// Print out the agent's memories after reflection.
	fmt.Println("\nAgent's memories after reflection:")
	for _, mem := range agent.Memory.Memories {
		fmt.Printf("- %s (Importance: %.1f)\n", mem.Description, mem.Importance)
	}

	// ===== NEW FEATURE DEMONSTRATION =====
	// Simulate agent's planning for the day.
	currentTime := time.Now()
	fmt.Println("\nAgent is planning the day...")
	err = agent.PlanDay(currentTime)
	if err != nil {
		fmt.Println("Error during planning:", err)
		return
	}

	// Print the agent's planned actions.
	fmt.Println("\nAgent's planned actions for the day:")
	for _, action := range agent.CurrentPlan.Actions() {
		fmt.Printf("- Action: %s | Location: %s | StartTime: %s\n", action.Description, action.Location, action.StartTime.Format(time.RFC822))
	}

	// Select Task
	agent.SelectTask()

	// Simulate agent perceiving a new observation.
	observation := "Klaus sees a protest happening outside the university."
	fmt.Printf("\nAgent perceives: %s\n", observation)
	err = agent.PerceiveAndReact(observation, currentTime)
	if err != nil {
		fmt.Println("Error during perception and reaction:", err)
		return
	}

	// Simulate agent perceiving a new observation.
	observation = "Klaus sees a squirrel climbing a tree."
	fmt.Printf("\nAgent perceives: %s\n", observation)
	err = agent.PerceiveAndReact(observation, currentTime)
	if err != nil {
		fmt.Println("Error during perception and reaction:", err)
		return
	}

	// Simulate agent perceiving a new observation.
	observation = "Klaus' little sister ran into the living room'."
	fmt.Printf("\nAgent perceives: %s\n", observation)
	err = agent.PerceiveAndReact(observation, currentTime)
	if err != nil {
		fmt.Println("Error during perception and reaction:", err)
		return
	}

	// Print out the agent's memories after updates (including perception and action execution).
	fmt.Println("\nAgent's memories after updates:")
	for _, mem := range agent.Memory.Memories {
		fmt.Printf("- %s (Importance: %.1f)\n", mem.Description, mem.Importance)
	}
}
