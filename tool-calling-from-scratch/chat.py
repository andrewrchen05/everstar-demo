# Example Usage
from core import Agent, Message, Role
from tools.detect_bounding_box import DetectBoundingBox
from tools.draw_bounding_box import DrawBoundingBox
from prompt.system_prompt import SYSTEM_PROMPT

def chat():
    """
    Basic chatbot interface.
    """
    # Initialize agent with optional system prompt
    detect_bbox_tool = DetectBoundingBox()
    draw_bbox_tool = DrawBoundingBox()
    
    agent = Agent(
        tools=[detect_bbox_tool, draw_bbox_tool],
        system_prompt=SYSTEM_PROMPT
    )
    
    print("Chatbot initialized. Type 'exit' to quit.\n")
    
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Goodbye!")
                break
            
            if not user_input.strip():
                continue
            
            # Create user message
            user_message = Message(role=Role.USER, content=user_input)
            
            # Get agent response
            response = agent.run(messages=[user_message])
            
            # Display response
            print(f"AI: {response.content}\n")
    finally:
        # Save conversation history before exiting
        agent.logger.save_conversation()
        if agent.logger.current_conversation_id:
            print(f"\nConversation saved to: conversation_history/{agent.logger.current_conversation_id}.json")


if __name__ == "__main__":
    chat()
