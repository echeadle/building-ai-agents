"""
AugmentedLLM with structured output validation.

This example demonstrates how to use JSON Schema validation to ensure
Claude's responses match a specific structure. This is essential for
building reliable, programmatic agent workflows.

Chapter 14: Building the Complete Augmented LLM
"""

import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify API key
if not os.getenv("ANTHROPIC_API_KEY"):
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

from augmented_llm import AugmentedLLM, AugmentedLLMConfig


def sentiment_analysis_example():
    """Demonstrate structured output for sentiment analysis."""
    
    print("\n--- Sentiment Analysis Example ---")
    
    # Define the expected response schema
    sentiment_schema = {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"]
            },
            "confidence": {
                "type": "number"
            },
            "reasoning": {
                "type": "string"
            },
            "key_phrases": {
                "type": "array"
            }
        },
        "required": ["sentiment", "confidence", "reasoning"]
    }
    
    # Create LLM with structured output
    config = AugmentedLLMConfig(
        system_prompt="""You are a sentiment analysis assistant.

Analyze the sentiment of user messages and respond with a JSON object containing:
- sentiment: "positive", "negative", or "neutral"
- confidence: A number from 0.0 to 1.0 indicating your confidence
- reasoning: A brief explanation of your analysis
- key_phrases: Array of phrases that influenced your analysis (optional)

IMPORTANT: Respond ONLY with the JSON object. No other text, no markdown formatting.""",
        response_schema=sentiment_schema
    )
    
    llm = AugmentedLLM(config=config)
    
    # Analyze some sample texts
    texts = [
        "I absolutely love this product! Best purchase I've ever made!",
        "The service was okay. Nothing special, but not terrible either.",
        "Terrible experience. The product broke after one day and customer service was unhelpful.",
        "Just received my order. It's a standard item that does what it's supposed to do."
    ]
    
    for text in texts:
        print(f"\nText: \"{text[:60]}...\"" if len(text) > 60 else f"\nText: \"{text}\"")
        
        try:
            response = llm.run(f"Analyze this text: {text}")
            result = json.loads(response)
            
            print(f"  Sentiment: {result['sentiment']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  Reasoning: {result['reasoning']}")
            if 'key_phrases' in result and result['key_phrases']:
                print(f"  Key phrases: {', '.join(result['key_phrases'])}")
        except Exception as e:
            print(f"  Error: {e}")
        
        llm.clear_history()


def entity_extraction_example():
    """Demonstrate structured output for entity extraction."""
    
    print("\n--- Entity Extraction Example ---")
    
    # Define schema for extracted entities
    entity_schema = {
        "type": "object",
        "properties": {
            "people": {
                "type": "array"
            },
            "organizations": {
                "type": "array"
            },
            "locations": {
                "type": "array"
            },
            "dates": {
                "type": "array"
            },
            "summary": {
                "type": "string"
            }
        },
        "required": ["people", "organizations", "locations", "dates", "summary"]
    }
    
    config = AugmentedLLMConfig(
        system_prompt="""You are an entity extraction assistant.

Extract named entities from text and respond with a JSON object containing:
- people: Array of person names mentioned
- organizations: Array of company/organization names
- locations: Array of places mentioned  
- dates: Array of dates or time references
- summary: One sentence summary of the text

If no entities of a type are found, return an empty array.

IMPORTANT: Respond ONLY with the JSON object. No other text.""",
        response_schema=entity_schema
    )
    
    llm = AugmentedLLM(config=config)
    
    # Sample texts for extraction
    texts = [
        "Apple CEO Tim Cook announced new products at the Cupertino headquarters on September 12, 2024.",
        "Dr. Sarah Johnson from MIT published her research findings in the Nature journal last week.",
    ]
    
    for text in texts:
        print(f"\nText: \"{text}\"")
        
        try:
            response = llm.run(f"Extract entities from: {text}")
            result = json.loads(response)
            
            print(f"  People: {result['people']}")
            print(f"  Organizations: {result['organizations']}")
            print(f"  Locations: {result['locations']}")
            print(f"  Dates: {result['dates']}")
            print(f"  Summary: {result['summary']}")
        except Exception as e:
            print(f"  Error: {e}")
        
        llm.clear_history()


def task_classification_example():
    """Demonstrate structured output for task classification."""
    
    print("\n--- Task Classification Example ---")
    
    classification_schema = {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["question", "request", "complaint", "feedback", "other"]
            },
            "priority": {
                "type": "string",
                "enum": ["low", "medium", "high", "urgent"]
            },
            "requires_human": {
                "type": "boolean"
            },
            "suggested_action": {
                "type": "string"
            }
        },
        "required": ["category", "priority", "requires_human", "suggested_action"]
    }
    
    config = AugmentedLLMConfig(
        system_prompt="""You are a customer service classifier.

Classify incoming messages and respond with a JSON object containing:
- category: One of "question", "request", "complaint", "feedback", "other"
- priority: One of "low", "medium", "high", "urgent"
- requires_human: true if a human should handle this, false if AI can respond
- suggested_action: Brief description of how to handle this message

Classification guidelines:
- Questions about products/services are usually "low" or "medium" priority
- Complaints about billing or broken items are "high" priority
- Safety issues or threats are "urgent" priority
- Complaints and urgent matters usually require_human = true

IMPORTANT: Respond ONLY with the JSON object. No other text.""",
        response_schema=classification_schema
    )
    
    llm = AugmentedLLM(config=config)
    
    messages = [
        "What are your store hours?",
        "I was charged twice for my order and I need a refund immediately!",
        "Just wanted to say your customer service team is amazing!",
        "My child swallowed a small part from your product.",
    ]
    
    for message in messages:
        print(f"\nMessage: \"{message}\"")
        
        try:
            response = llm.run(f"Classify this customer message: {message}")
            result = json.loads(response)
            
            print(f"  Category: {result['category']}")
            print(f"  Priority: {result['priority']}")
            print(f"  Requires human: {result['requires_human']}")
            print(f"  Suggested action: {result['suggested_action']}")
        except Exception as e:
            print(f"  Error: {e}")
        
        llm.clear_history()


def main():
    """Run all structured output examples."""
    
    print("AugmentedLLM with Structured Output")
    print("=" * 50)
    
    sentiment_analysis_example()
    entity_extraction_example()
    task_classification_example()
    
    print("\n" + "=" * 50)
    print("All examples complete!")


if __name__ == "__main__":
    main()
