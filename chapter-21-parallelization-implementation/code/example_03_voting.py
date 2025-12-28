"""
Voting pattern: multiple perspectives on the same task.

Chapter 21: Parallelization - Implementation

Voting runs the same task multiple times (possibly with different
prompts or configurations) and aggregates the results to build
consensus and increase confidence.
"""

import asyncio
import os
from collections import Counter
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")


@dataclass
class Voter:
    """
    Defines a single voter configuration.
    
    Attributes:
        name: Identifier for this voter
        system_prompt: The perspective this voter brings
        temperature: Sampling temperature (higher = more creative)
    """
    name: str
    system_prompt: str
    temperature: float = 1.0


@dataclass
class Vote:
    """
    A single vote from one voter.
    
    Attributes:
        voter_name: Which voter cast this vote
        decision: The choice made
        confidence: Self-reported confidence level
        reasoning: Explanation for the decision
        success: Whether the vote was collected successfully
        error: Error message if failed
    """
    voter_name: str
    decision: str
    confidence: str | None = None
    reasoning: str | None = None
    success: bool = True
    error: str | None = None


@dataclass
class VotingResult:
    """
    Aggregated voting results.
    
    Attributes:
        votes: List of individual votes
        winner: The winning decision
        vote_counts: How many votes each option received
        consensus: Whether consensus threshold was met
        consensus_threshold: The threshold used
        execution_time: Total time for voting
    """
    votes: list[Vote]
    winner: str | None
    vote_counts: dict[str, int]
    consensus: bool
    consensus_threshold: float
    execution_time: float


class VotingWorkflow:
    """
    Implements the voting pattern for parallel consensus.
    
    Runs the same task with multiple voters (different perspectives)
    and aggregates their decisions to determine a consensus result.
    
    Example usage:
        workflow = VotingWorkflow(consensus_threshold=0.7)
        voters = [
            Voter("conservative", "You are risk-averse..."),
            Voter("moderate", "You are balanced..."),
            Voter("aggressive", "You favor bold moves...")
        ]
        result = await workflow.run(
            voters,
            "Should we invest in this startup?",
            options=["yes", "no", "maybe"]
        )
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        consensus_threshold: float = 0.6
    ):
        """
        Initialize the voting workflow.
        
        Args:
            model: The Claude model to use
            max_tokens: Maximum tokens per voter response
            consensus_threshold: Fraction of votes needed for consensus
        """
        self.model = model
        self.max_tokens = max_tokens
        self.consensus_threshold = consensus_threshold
        self.async_client = anthropic.AsyncAnthropic()
    
    async def _get_vote(
        self,
        voter: Voter,
        prompt: str,
        options: list[str] | None = None
    ) -> Vote:
        """
        Get a single vote from a voter.
        
        Args:
            voter: The voter configuration
            prompt: The decision prompt
            options: Optional list of valid choices
            
        Returns:
            Vote with the voter's decision
        """
        try:
            # Build the voting prompt
            if options:
                options_str = ", ".join(options)
                full_prompt = f"""{prompt}

You must choose exactly one of these options: {options_str}

Respond in this exact format:
DECISION: [your choice - must be one of the options]
CONFIDENCE: [high/medium/low]
REASONING: [brief explanation in 1-2 sentences]"""
            else:
                full_prompt = f"""{prompt}

Respond in this exact format:
DECISION: [your choice]
CONFIDENCE: [high/medium/low]
REASONING: [brief explanation in 1-2 sentences]"""
            
            response = await self.async_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=voter.temperature,
                system=voter.system_prompt,
                messages=[{"role": "user", "content": full_prompt}]
            )
            
            # Parse the response
            text = response.content[0].text
            decision = self._extract_field(text, "DECISION")
            confidence = self._extract_field(text, "CONFIDENCE")
            reasoning = self._extract_field(text, "REASONING")
            
            # Normalize decision to match options if provided
            if options and decision:
                decision = self._normalize_to_options(decision, options)
            
            return Vote(
                voter_name=voter.name,
                decision=decision or "UNKNOWN",
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            return Vote(
                voter_name=voter.name,
                decision="ERROR",
                success=False,
                error=str(e)
            )
    
    def _extract_field(self, text: str, field: str) -> str | None:
        """Extract a field value from formatted response."""
        for line in text.split("\n"):
            line = line.strip()
            if line.upper().startswith(f"{field}:"):
                return line.split(":", 1)[1].strip()
        return None
    
    def _normalize_to_options(
        self, 
        decision: str, 
        options: list[str]
    ) -> str:
        """
        Normalize a decision to match one of the valid options.
        
        Handles cases where the model includes extra text or
        slight variations in the response.
        """
        decision_lower = decision.lower().strip()
        
        # Try exact match first
        for option in options:
            if option.lower() == decision_lower:
                return option
        
        # Try partial match
        for option in options:
            if option.lower() in decision_lower:
                return option
            if decision_lower in option.lower():
                return option
        
        # Return as-is if no match found
        return decision
    
    def _aggregate_votes(
        self, 
        votes: list[Vote]
    ) -> tuple[str | None, dict[str, int], bool]:
        """
        Aggregate votes and determine consensus.
        
        Returns:
            (winner, vote_counts, has_consensus)
        """
        # Filter successful votes with valid decisions
        valid_votes = [
            v for v in votes 
            if v.success and v.decision not in ("UNKNOWN", "ERROR")
        ]
        
        if not valid_votes:
            return None, {}, False
        
        # Count votes
        decisions = [v.decision for v in valid_votes]
        vote_counts = dict(Counter(decisions))
        
        # Find winner (most common decision)
        winner = max(vote_counts, key=vote_counts.get)
        winner_count = vote_counts[winner]
        
        # Check if consensus threshold is met
        consensus = (winner_count / len(valid_votes)) >= self.consensus_threshold
        
        return winner, vote_counts, consensus
    
    async def run(
        self,
        voters: list[Voter],
        prompt: str,
        options: list[str] | None = None
    ) -> VotingResult:
        """
        Run voting with all voters in parallel.
        
        Args:
            voters: List of Voter configurations
            prompt: The decision prompt
            options: Optional list of valid choices
            
        Returns:
            VotingResult with individual votes and consensus
        """
        import time
        start = time.time()
        
        # Get all votes in parallel
        votes = await asyncio.gather(*[
            self._get_vote(voter, prompt, options)
            for voter in voters
        ])
        
        # Aggregate results
        winner, vote_counts, consensus = self._aggregate_votes(votes)
        
        return VotingResult(
            votes=list(votes),
            winner=winner,
            vote_counts=vote_counts,
            consensus=consensus,
            consensus_threshold=self.consensus_threshold,
            execution_time=time.time() - start
        )


# =============================================================================
# Example: Content Moderation with Voting
# =============================================================================

async def content_moderation_example():
    """
    Use voting to classify content with high confidence.
    
    Multiple moderators with different perspectives review
    content and vote on its classification.
    """
    workflow = VotingWorkflow(consensus_threshold=0.6)
    
    # Define diverse voters with different perspectives
    voters = [
        Voter(
            name="strict_moderator",
            system_prompt="""You are a strict content moderator. 
You err on the side of caution and flag anything that could 
potentially be problematic, including mild negativity or complaints.""",
            temperature=0.3
        ),
        Voter(
            name="balanced_moderator", 
            system_prompt="""You are a balanced content moderator.
You carefully weigh context and intent when making decisions.
You distinguish between venting frustration and actual harmful content.""",
            temperature=0.5
        ),
        Voter(
            name="lenient_moderator",
            system_prompt="""You are a lenient content moderator.
You focus on clear violations and give benefit of the doubt
for ambiguous content. Complaints and criticism are acceptable.""",
            temperature=0.3
        ),
        Voter(
            name="context_analyst",
            system_prompt="""You analyze content with deep attention to context.
Consider the full picture before making a judgment.
Distinguish between criticism of policies and personal attacks.""",
            temperature=0.7
        ),
        Voter(
            name="policy_expert",
            system_prompt="""You are an expert in content policies.
You apply rules consistently and fairly based on:
- Acceptable: criticism, complaints, disagreement
- Needs review: borderline language, unclear intent
- Rejected: hate speech, threats, harassment""",
            temperature=0.3
        )
    ]
    
    # Test content samples
    test_cases = [
        {
            "content": """I can't believe how stupid the new policy is. 
The people who made it must have rocks for brains. We should organize a protest!""",
            "description": "Frustrated complaint with hyperbole"
        },
        {
            "content": """This product is absolutely terrible. 
The company should be ashamed. I'm telling everyone I know to avoid it.""",
            "description": "Negative review"
        },
        {
            "content": """Great article! Really helped me understand the topic better. 
Thanks for sharing this valuable information.""",
            "description": "Positive comment"
        }
    ]
    
    options = ["acceptable", "needs_review", "rejected"]
    
    print("=" * 60)
    print("CONTENT MODERATION WITH VOTING")
    print("=" * 60)
    
    for test in test_cases:
        print(f"\n{'-' * 60}")
        print(f"Content: \"{test['content'][:60]}...\"")
        print(f"Type: {test['description']}")
        print("-" * 60)
        
        prompt = f"""Classify the following user content:

{test['content']}

Is this content acceptable, needs_review, or should be rejected?"""
        
        result = await workflow.run(voters, prompt, options)
        
        print(f"\nExecution time: {result.execution_time:.2f}s")
        print("\nIndividual votes:")
        for vote in result.votes:
            status = "✓" if vote.success else "✗"
            confidence = f" ({vote.confidence})" if vote.confidence else ""
            print(f"  {status} {vote.voter_name}: {vote.decision}{confidence}")
        
        print(f"\nVote counts: {result.vote_counts}")
        print(f"Winner: {result.winner}")
        consensus_status = "YES ✓" if result.consensus else "NO ✗"
        print(f"Consensus reached: {consensus_status}")


# =============================================================================
# Example: Investment Decision with Voting
# =============================================================================

async def investment_decision_example():
    """
    Use voting for investment decisions with different risk profiles.
    
    Different "investors" with varying risk tolerances vote on
    whether to make an investment.
    """
    workflow = VotingWorkflow(consensus_threshold=0.5)
    
    voters = [
        Voter(
            name="conservative_investor",
            system_prompt="""You are a conservative investor focused on capital preservation.
You prefer stable, proven investments with low risk.
You're skeptical of new ventures and high-growth claims.""",
            temperature=0.3
        ),
        Voter(
            name="moderate_investor",
            system_prompt="""You are a moderate investor seeking balanced returns.
You consider both risk and reward carefully.
You like diversification and reasonable growth potential.""",
            temperature=0.5
        ),
        Voter(
            name="aggressive_investor",
            system_prompt="""You are an aggressive investor seeking high returns.
You're willing to accept higher risk for potential gains.
You're excited by innovation and growth opportunities.""",
            temperature=0.7
        ),
        Voter(
            name="value_investor",
            system_prompt="""You are a value investor focused on fundamentals.
You look for undervalued assets with strong financials.
You're patient and focus on long-term value.""",
            temperature=0.3
        ),
        Voter(
            name="trend_analyst",
            system_prompt="""You analyze market trends and momentum.
You consider market sentiment and timing.
You look at what sectors are gaining attention.""",
            temperature=0.5
        )
    ]
    
    # Investment opportunity
    opportunity = """
    A 3-year-old AI startup is seeking Series B funding at a $500M valuation.
    
    Positives:
    - Revenue grew 300% last year to $20M ARR
    - Strong technical team from top tech companies
    - Patent portfolio with 12 granted patents
    - Major enterprise clients including 3 Fortune 500 companies
    
    Concerns:
    - Still not profitable (burning $5M/month)
    - Highly competitive market with well-funded rivals
    - Key technical lead recently departed
    - Valuation is 25x revenue (high for the sector)
    """
    
    prompt = f"""Evaluate this investment opportunity:

{opportunity}

Should we invest in this startup?"""
    
    options = ["invest", "pass", "more_info_needed"]
    
    print("\n" + "=" * 60)
    print("INVESTMENT DECISION WITH VOTING")
    print("=" * 60)
    print(f"\nOpportunity: AI Startup Series B")
    print("-" * 60)
    
    result = await workflow.run(voters, prompt, options)
    
    print(f"\nExecution time: {result.execution_time:.2f}s")
    print("\nIndividual votes:")
    for vote in result.votes:
        status = "✓" if vote.success else "✗"
        print(f"\n  {status} {vote.voter_name}: {vote.decision}")
        if vote.reasoning:
            # Truncate long reasoning
            reasoning = vote.reasoning[:100] + "..." if len(vote.reasoning) > 100 else vote.reasoning
            print(f"      Reasoning: {reasoning}")
    
    print(f"\n{'-' * 60}")
    print(f"Vote counts: {result.vote_counts}")
    print(f"Decision: {result.winner}")
    consensus_status = "YES ✓" if result.consensus else "NO ✗"
    print(f"Consensus reached: {consensus_status}")


# =============================================================================
# Helper Functions: Aggregation Strategies
# =============================================================================

def majority_vote(votes: list[str]) -> str:
    """
    Return the most common vote.
    
    Simple majority wins - the option with the most votes.
    """
    counts = Counter(votes)
    return counts.most_common(1)[0][0]


def consensus_vote(
    votes: list[str], 
    threshold: float = 0.7
) -> tuple[str | None, bool]:
    """
    Return winner only if it exceeds the threshold.
    
    Useful when you need high confidence in the decision.
    
    Returns:
        (winner, reached_consensus)
    """
    counts = Counter(votes)
    total = len(votes)
    
    winner, count = counts.most_common(1)[0]
    ratio = count / total
    
    if ratio >= threshold:
        return winner, True
    return winner, False


def weighted_vote(
    votes: list[tuple[str, float]]
) -> tuple[str, dict[str, float]]:
    """
    Aggregate votes with weights.
    
    Useful when some voters should have more influence.
    
    Args:
        votes: List of (decision, weight) tuples
        
    Returns:
        (winner, weighted_scores)
    """
    scores: dict[str, float] = {}
    for decision, weight in votes:
        scores[decision] = scores.get(decision, 0) + weight
    
    winner = max(scores, key=scores.get)
    return winner, scores


async def main():
    """Run all voting examples."""
    await content_moderation_example()
    await investment_decision_example()


if __name__ == "__main__":
    asyncio.run(main())
