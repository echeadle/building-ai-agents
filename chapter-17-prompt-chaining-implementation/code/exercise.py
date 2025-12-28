"""
Exercise Solution: Content Repurposing Chain

Build a content repurposing chain that takes a blog post topic and produces
content for multiple platforms.

Requirements:
1. Step 1: Generate a detailed blog post outline (5-7 sections)
2. Step 2: Write the full blog post based on the outline
3. Step 3: Create a LinkedIn post summarizing the blog (under 300 characters)
4. Step 4: Generate 3 tweet variations based on the blog post

Quality Gates:
- Blog outline must have at least 5 sections
- Full blog post must be at least 500 words
- LinkedIn post must be under 300 characters
- Each tweet must be under 280 characters

Bonus: LLM-based quality scoring

Chapter 17: Prompt Chaining - Implementation
"""

import os
import time
import re
from dotenv import load_dotenv
import anthropic
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic()
MODEL_NAME = "claude-sonnet-4-20250514"


@dataclass
class QualityCheck:
    """Result of a quality gate check."""
    passed: bool
    message: str
    score: Optional[float] = None
    details: dict = field(default_factory=dict)


@dataclass
class StepResult:
    """Result from a chain step."""
    name: str
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    quality_check: Optional[QualityCheck] = None
    attempts: int = 0
    duration_seconds: float = 0.0


@dataclass
class ContentRepurposingResult:
    """Complete result from the content repurposing chain."""
    topic: str
    success: bool
    outline: Optional[str] = None
    blog_post: Optional[str] = None
    linkedin_post: Optional[str] = None
    tweets: Optional[list[str]] = None
    quality_score: Optional[float] = None
    steps: list[StepResult] = field(default_factory=list)
    total_duration_seconds: float = 0.0


# ============================================================================
# Quality Gate Functions
# ============================================================================

def validate_outline(outline: str) -> QualityCheck:
    """
    Validate that the outline has at least 5 sections.
    
    Looks for numbered sections (1., 2., etc.) or headers (##, ###).
    """
    # Count numbered sections
    numbered_sections = len(re.findall(r'^\d+\.', outline, re.MULTILINE))
    
    # Count markdown headers
    header_sections = len(re.findall(r'^#{2,3}\s+', outline, re.MULTILINE))
    
    # Count bullet points that might be section indicators
    bullet_sections = len(re.findall(r'^[-*]\s+\*\*', outline, re.MULTILINE))
    
    section_count = max(numbered_sections, header_sections, bullet_sections)
    
    if section_count >= 5:
        return QualityCheck(
            passed=True,
            message=f"Outline has {section_count} sections",
            details={"section_count": section_count}
        )
    else:
        return QualityCheck(
            passed=False,
            message=f"Outline has only {section_count} sections (need at least 5)",
            details={"section_count": section_count}
        )


def validate_blog_post(blog_post: str) -> QualityCheck:
    """
    Validate that the blog post is at least 500 words.
    """
    word_count = len(blog_post.split())
    
    if word_count >= 500:
        return QualityCheck(
            passed=True,
            message=f"Blog post has {word_count} words",
            details={"word_count": word_count}
        )
    else:
        return QualityCheck(
            passed=False,
            message=f"Blog post has only {word_count} words (need at least 500)",
            details={"word_count": word_count}
        )


def validate_linkedin_post(linkedin_post: str) -> QualityCheck:
    """
    Validate that the LinkedIn post is under 300 characters.
    """
    char_count = len(linkedin_post)
    
    if char_count <= 300:
        return QualityCheck(
            passed=True,
            message=f"LinkedIn post is {char_count} characters",
            details={"char_count": char_count}
        )
    else:
        return QualityCheck(
            passed=False,
            message=f"LinkedIn post is {char_count} characters (max 300)",
            details={"char_count": char_count}
        )


def validate_tweets(tweets: list[str]) -> QualityCheck:
    """
    Validate that we have 3 tweets, each under 280 characters.
    """
    if len(tweets) < 3:
        return QualityCheck(
            passed=False,
            message=f"Only {len(tweets)} tweets generated (need 3)",
            details={"tweet_count": len(tweets)}
        )
    
    char_counts = [len(t) for t in tweets]
    over_limit = [(i+1, c) for i, c in enumerate(char_counts) if c > 280]
    
    if over_limit:
        details = {f"tweet_{num}_chars": chars for num, chars in over_limit}
        return QualityCheck(
            passed=False,
            message=f"Tweets {[n for n,_ in over_limit]} exceed 280 characters",
            details=details
        )
    
    return QualityCheck(
        passed=True,
        message=f"All 3 tweets are within character limits",
        details={"char_counts": char_counts}
    )


def evaluate_overall_quality(
    topic: str,
    blog_post: str,
    linkedin_post: str,
    tweets: list[str]
) -> QualityCheck:
    """
    BONUS: Use LLM to evaluate overall content quality.
    
    Returns a score from 1-10 and detailed feedback.
    """
    tweets_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(tweets)])
    
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""Evaluate this content suite for the topic "{topic}".

BLOG POST (excerpt - first 500 chars):
{blog_post[:500]}...

LINKEDIN POST:
{linkedin_post}

TWEETS:
{tweets_text}

Rate the overall quality from 1-10 considering:
- Message consistency across formats
- Engagement potential
- Professional tone
- Clear calls to action

Respond in this exact format:
SCORE: [1-10]
PASSED: [YES/NO] (YES if score >= 7)
STRENGTHS: [1-2 sentence summary]
IMPROVEMENTS: [1-2 sentence summary]"""
        }]
    )
    
    result_text = response.content[0].text
    
    try:
        lines = result_text.strip().split("\n")
        score_line = [l for l in lines if l.startswith("SCORE:")][0]
        passed_line = [l for l in lines if l.startswith("PASSED:")][0]
        strengths_line = [l for l in lines if l.startswith("STRENGTHS:")][0]
        improvements_line = [l for l in lines if l.startswith("IMPROVEMENTS:")][0]
        
        score = float(score_line.split(":")[1].strip())
        passed = "YES" in passed_line.upper()
        strengths = strengths_line.split(":", 1)[1].strip()
        improvements = improvements_line.split(":", 1)[1].strip()
        
        return QualityCheck(
            passed=passed,
            message=f"Quality score: {score}/10",
            score=score,
            details={
                "strengths": strengths,
                "improvements": improvements
            }
        )
    except (IndexError, ValueError):
        return QualityCheck(
            passed=True,
            message="Could not parse quality evaluation",
            score=None
        )


# ============================================================================
# Chain Step Functions
# ============================================================================

def generate_outline(topic: str) -> str:
    """Step 1: Generate a blog post outline."""
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Create a detailed blog post outline for the topic: "{topic}"

Requirements:
- Include 5-7 main sections
- Each section should have a clear heading
- Include 2-3 bullet points per section describing what to cover
- Start with an introduction and end with a conclusion
- Make it comprehensive but focused

Format as a numbered list with sub-bullets."""
        }]
    )
    return response.content[0].text


def write_blog_post(topic: str, outline: str) -> str:
    """Step 2: Write the full blog post from the outline."""
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Write a complete blog post based on this outline.

TOPIC: {topic}

OUTLINE:
{outline}

Requirements:
- Write at least 500 words (aim for 600-800)
- Follow the outline structure
- Use engaging, professional language
- Include a compelling introduction
- End with a clear call to action
- Use markdown formatting for headers"""
        }]
    )
    return response.content[0].text


def create_linkedin_post(topic: str, blog_post: str) -> str:
    """Step 3: Create a LinkedIn summary."""
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"""Create a LinkedIn post summarizing this blog post.

TOPIC: {topic}

BLOG POST (excerpt):
{blog_post[:1500]}

Requirements:
- MUST be under 300 characters (this is critical!)
- Capture the key value proposition
- Include a hook to drive engagement
- Professional but conversational tone
- End with a call to action or question

Count your characters carefully. Under 300 is mandatory."""
        }]
    )
    return response.content[0].text.strip()


def generate_tweets(topic: str, blog_post: str) -> list[str]:
    """Step 4: Generate 3 tweet variations."""
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""Create 3 different tweets promoting this blog post.

TOPIC: {topic}

BLOG POST (excerpt):
{blog_post[:1000]}

Requirements:
- Each tweet MUST be under 280 characters (critical!)
- Each should take a different angle:
  1. Hook with a surprising fact or question
  2. Focus on the main benefit
  3. Use a quote or key insight
- Make them engaging and shareable
- Include relevant emojis sparingly

Format your response as:
TWEET 1: [tweet text]
TWEET 2: [tweet text]  
TWEET 3: [tweet text]

Count characters carefully!"""
        }]
    )
    
    # Parse tweets from response
    result_text = response.content[0].text
    tweets = []
    
    for line in result_text.split("\n"):
        if line.strip().startswith("TWEET"):
            tweet = line.split(":", 1)[1].strip() if ":" in line else ""
            if tweet:
                tweets.append(tweet)
    
    # Fallback: if parsing fails, split by double newlines
    if len(tweets) < 3:
        tweets = [t.strip() for t in result_text.split("\n\n") if t.strip()][:3]
    
    return tweets[:3]


# ============================================================================
# Main Chain Function
# ============================================================================

def run_content_repurposing_chain(
    topic: str,
    max_retries: int = 2,
    require_quality_score: float = 7.0
) -> ContentRepurposingResult:
    """
    Execute the complete content repurposing chain.
    
    Args:
        topic: The blog post topic
        max_retries: Retries per step for quality failures
        require_quality_score: Minimum acceptable quality score (0 to disable)
    
    Returns:
        ContentRepurposingResult with all generated content
    """
    start_time = time.time()
    result = ContentRepurposingResult(topic=topic, success=False)
    
    # Step 1: Generate Outline
    print("\n[1/4] Generating blog post outline...")
    step_start = time.time()
    
    for attempt in range(max_retries + 1):
        outline = generate_outline(topic)
        quality = validate_outline(outline)
        
        if quality.passed:
            print(f"  ✓ {quality.message}")
            result.outline = outline
            result.steps.append(StepResult(
                name="Generate Outline",
                success=True,
                output=outline,
                quality_check=quality,
                attempts=attempt + 1,
                duration_seconds=time.time() - step_start
            ))
            break
        else:
            print(f"  ✗ {quality.message} (attempt {attempt + 1})")
            if attempt == max_retries:
                result.steps.append(StepResult(
                    name="Generate Outline",
                    success=False,
                    error=quality.message,
                    quality_check=quality,
                    attempts=attempt + 1
                ))
                return result
    
    # Step 2: Write Blog Post
    print("\n[2/4] Writing full blog post...")
    step_start = time.time()
    
    for attempt in range(max_retries + 1):
        blog_post = write_blog_post(topic, result.outline)
        quality = validate_blog_post(blog_post)
        
        if quality.passed:
            print(f"  ✓ {quality.message}")
            result.blog_post = blog_post
            result.steps.append(StepResult(
                name="Write Blog Post",
                success=True,
                output=blog_post,
                quality_check=quality,
                attempts=attempt + 1,
                duration_seconds=time.time() - step_start
            ))
            break
        else:
            print(f"  ✗ {quality.message} (attempt {attempt + 1})")
            if attempt == max_retries:
                result.steps.append(StepResult(
                    name="Write Blog Post",
                    success=False,
                    error=quality.message,
                    quality_check=quality,
                    attempts=attempt + 1
                ))
                return result
    
    # Step 3: Create LinkedIn Post
    print("\n[3/4] Creating LinkedIn post...")
    step_start = time.time()
    
    for attempt in range(max_retries + 1):
        linkedin_post = create_linkedin_post(topic, result.blog_post)
        quality = validate_linkedin_post(linkedin_post)
        
        if quality.passed:
            print(f"  ✓ {quality.message}")
            result.linkedin_post = linkedin_post
            result.steps.append(StepResult(
                name="Create LinkedIn Post",
                success=True,
                output=linkedin_post,
                quality_check=quality,
                attempts=attempt + 1,
                duration_seconds=time.time() - step_start
            ))
            break
        else:
            print(f"  ✗ {quality.message} (attempt {attempt + 1})")
            if attempt == max_retries:
                result.steps.append(StepResult(
                    name="Create LinkedIn Post",
                    success=False,
                    error=quality.message,
                    quality_check=quality,
                    attempts=attempt + 1
                ))
                return result
    
    # Step 4: Generate Tweets
    print("\n[4/4] Generating tweets...")
    step_start = time.time()
    
    for attempt in range(max_retries + 1):
        tweets = generate_tweets(topic, result.blog_post)
        quality = validate_tweets(tweets)
        
        if quality.passed:
            print(f"  ✓ {quality.message}")
            result.tweets = tweets
            result.steps.append(StepResult(
                name="Generate Tweets",
                success=True,
                output="\n".join(tweets),
                quality_check=quality,
                attempts=attempt + 1,
                duration_seconds=time.time() - step_start
            ))
            break
        else:
            print(f"  ✗ {quality.message} (attempt {attempt + 1})")
            if attempt == max_retries:
                result.steps.append(StepResult(
                    name="Generate Tweets",
                    success=False,
                    error=quality.message,
                    quality_check=quality,
                    attempts=attempt + 1
                ))
                return result
    
    # BONUS: Overall Quality Evaluation
    if require_quality_score > 0:
        print("\n[BONUS] Evaluating overall content quality...")
        quality_eval = evaluate_overall_quality(
            topic, result.blog_post, result.linkedin_post, result.tweets
        )
        result.quality_score = quality_eval.score
        
        if quality_eval.score:
            print(f"  Quality Score: {quality_eval.score}/10")
            if quality_eval.details.get("strengths"):
                print(f"  Strengths: {quality_eval.details['strengths']}")
            if quality_eval.details.get("improvements"):
                print(f"  Improvements: {quality_eval.details['improvements']}")
            
            if quality_eval.score < require_quality_score:
                print(f"  ⚠ Score below threshold ({require_quality_score})")
    
    result.success = True
    result.total_duration_seconds = time.time() - start_time
    
    return result


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("CONTENT REPURPOSING CHAIN")
    print("="*60)
    
    # Run the chain
    result = run_content_repurposing_chain(
        topic="How AI is Transforming Small Business Marketing in 2025",
        max_retries=2,
        require_quality_score=7.0
    )
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"\nTopic: {result.topic}")
    print(f"Success: {result.success}")
    print(f"Total Duration: {result.total_duration_seconds:.2f}s")
    
    if result.quality_score:
        print(f"Quality Score: {result.quality_score}/10")
    
    print("\n--- Step Summary ---")
    for step in result.steps:
        status = "✓" if step.success else "✗"
        print(f"{status} {step.name}: {step.attempts} attempt(s), {step.duration_seconds:.2f}s")
    
    if result.success:
        print("\n--- Blog Post Outline ---")
        print(result.outline[:500] + "..." if len(result.outline) > 500 else result.outline)
        
        print("\n--- Blog Post (excerpt) ---")
        print(result.blog_post[:800] + "..." if len(result.blog_post) > 800 else result.blog_post)
        
        print("\n--- LinkedIn Post ---")
        print(f"({len(result.linkedin_post)} chars)")
        print(result.linkedin_post)
        
        print("\n--- Tweets ---")
        for i, tweet in enumerate(result.tweets, 1):
            print(f"\nTweet {i} ({len(tweet)} chars):")
            print(tweet)
    else:
        print("\n❌ Chain failed. Check step errors above.")
