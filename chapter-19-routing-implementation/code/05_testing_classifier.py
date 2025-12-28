"""
Testing classification accuracy for the customer service router.

This module demonstrates how to create test datasets and measure
classifier accuracy - a critical step before deploying any routing system.

Chapter 19: Routing - Implementation
"""

import os
from typing import Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env file
load_dotenv()

# Verify API key is available
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

# Initialize the Anthropic client
client = anthropic.Anthropic()


# ============================================================
# Classifier (same as in other examples)
# ============================================================

CLASSIFICATION_PROMPT = """You are a customer service query classifier. Analyze the message and classify it into exactly ONE category.

Categories:
- BILLING: Charges, invoices, payments, refunds, pricing, subscription costs
- TECHNICAL: Product issues, bugs, errors, how-to questions, setup problems
- ACCOUNT: Password resets, login issues, profile changes, account settings
- GENERAL: General inquiries, feedback, compliments, complaints, other

Rules:
1. Respond with ONLY the category name (BILLING, TECHNICAL, ACCOUNT, or GENERAL)
2. Choose the BEST fit if the query could belong to multiple categories
3. If truly ambiguous, choose GENERAL

Message: {message}

Category:"""

VALID_CATEGORIES = {"BILLING", "TECHNICAL", "ACCOUNT", "GENERAL"}


def classify_query(message: str) -> str:
    """Classify a customer query into a category."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        messages=[
            {"role": "user", "content": CLASSIFICATION_PROMPT.format(message=message)}
        ]
    )
    
    category = response.content[0].text.strip().upper()
    
    for valid_cat in VALID_CATEGORIES:
        if valid_cat in category:
            return valid_cat
    
    return "GENERAL"


# ============================================================
# Test Dataset
# ============================================================
# A good test dataset has:
# 1. Representative samples for each category
# 2. Edge cases that could belong to multiple categories
# 3. Varying phrasings and styles

# Format: (query, expected_category)
TEST_DATASET: list[Tuple[str, str]] = [
    # ========================================
    # BILLING - Clear cases
    # ========================================
    ("Why was I charged $50 yesterday?", "BILLING"),
    ("I need a refund for my last purchase", "BILLING"),
    ("What's the difference between the basic and pro plans?", "BILLING"),
    ("My credit card was declined when trying to pay", "BILLING"),
    ("Can I get an invoice for tax purposes?", "BILLING"),
    ("How do I update my payment method?", "BILLING"),
    ("I was billed twice for the same subscription", "BILLING"),
    
    # ========================================
    # TECHNICAL - Clear cases
    # ========================================
    ("The app crashes when I click the submit button", "TECHNICAL"),
    ("How do I enable dark mode?", "TECHNICAL"),
    ("I'm getting an error message: 'Connection timeout'", "TECHNICAL"),
    ("The search feature isn't returning any results", "TECHNICAL"),
    ("How do I integrate with your API?", "TECHNICAL"),
    ("The page won't load on Safari browser", "TECHNICAL"),
    ("Is there a keyboard shortcut for saving?", "TECHNICAL"),
    
    # ========================================
    # ACCOUNT - Clear cases
    # ========================================
    ("I forgot my password", "ACCOUNT"),
    ("How do I change my username?", "ACCOUNT"),
    ("I can't log in to my account", "ACCOUNT"),
    ("I want to delete my account permanently", "ACCOUNT"),
    ("How do I enable two-factor authentication?", "ACCOUNT"),
    ("I need to update my email address", "ACCOUNT"),
    ("Someone else may have accessed my account", "ACCOUNT"),
    
    # ========================================
    # GENERAL - Clear cases
    # ========================================
    ("What are your office hours?", "GENERAL"),
    ("I love your product!", "GENERAL"),
    ("Do you have a mobile app?", "GENERAL"),
    ("I want to provide some feedback on the new design", "GENERAL"),
    ("Where is your company located?", "GENERAL"),
    ("Are you hiring?", "GENERAL"),
    ("What's your refund policy?", "GENERAL"),
    
    # ========================================
    # Edge cases - These are trickier
    # ========================================
    # Billing + fulfillment
    ("I was charged but never received access to my account", "BILLING"),
    
    # Account + technical
    ("My password reset email never arrived in my inbox", "ACCOUNT"),
    
    # Billing + technical
    ("The pricing page shows different prices than my invoice", "BILLING"),
    
    # Account + billing
    ("I can't access my premium features even though I paid", "BILLING"),
    
    # Technical + general
    ("When will the new dashboard feature be available?", "GENERAL"),
    
    # Complaint (general) but about billing
    ("I'm very frustrated with all these charges", "BILLING"),
    
    # Technical phrased as general question
    ("Why is everything so slow today?", "TECHNICAL"),
]


# ============================================================
# Testing Functions
# ============================================================

@dataclass
class TestResult:
    """Result of testing a single query."""
    query: str
    expected: str
    predicted: str
    correct: bool


@dataclass
class AccuracyReport:
    """Complete accuracy report."""
    overall_accuracy: float
    correct_count: int
    total_count: int
    category_accuracy: dict[str, float]
    category_counts: dict[str, dict[str, int]]
    results: list[TestResult]
    misclassifications: list[TestResult]


def run_accuracy_test(
    dataset: list[Tuple[str, str]] = TEST_DATASET,
    verbose: bool = True
) -> AccuracyReport:
    """
    Run the classifier against a test dataset and compute accuracy.
    
    Args:
        dataset: List of (query, expected_category) tuples
        verbose: If True, print progress and errors
        
    Returns:
        AccuracyReport with detailed metrics
    """
    results: list[TestResult] = []
    correct_count = 0
    
    # Track per-category performance
    category_counts: dict[str, dict[str, int]] = {
        cat: {"correct": 0, "total": 0} 
        for cat in VALID_CATEGORIES
    }
    
    if verbose:
        print("Running classification tests...")
        print(f"Total test cases: {len(dataset)}\n")
    
    for i, (query, expected) in enumerate(dataset):
        # Classify the query
        predicted = classify_query(query)
        is_correct = predicted == expected
        
        # Update counts
        if is_correct:
            correct_count += 1
            category_counts[expected]["correct"] += 1
        
        category_counts[expected]["total"] += 1
        
        # Store result
        result = TestResult(
            query=query,
            expected=expected,
            predicted=predicted,
            correct=is_correct,
        )
        results.append(result)
        
        # Show progress
        if verbose:
            status = "✓" if is_correct else "✗"
            print(f"[{i+1}/{len(dataset)}] {status}", end="")
            
            if not is_correct:
                print(f" - Expected {expected}, got {predicted}")
                print(f"    Query: {query[:60]}...")
            else:
                print()
    
    # Calculate accuracies
    overall_accuracy = (correct_count / len(dataset)) * 100
    
    category_accuracy = {}
    for cat, counts in category_counts.items():
        if counts["total"] > 0:
            category_accuracy[cat] = (counts["correct"] / counts["total"]) * 100
        else:
            category_accuracy[cat] = 0.0
    
    # Collect misclassifications
    misclassifications = [r for r in results if not r.correct]
    
    return AccuracyReport(
        overall_accuracy=overall_accuracy,
        correct_count=correct_count,
        total_count=len(dataset),
        category_accuracy=category_accuracy,
        category_counts=category_counts,
        results=results,
        misclassifications=misclassifications,
    )


def print_report(report: AccuracyReport) -> None:
    """Print a formatted accuracy report."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION ACCURACY REPORT")
    print("=" * 60)
    
    # Overall accuracy
    print(f"\n📊 Overall Accuracy: {report.overall_accuracy:.1f}%")
    print(f"   Correct: {report.correct_count} / {report.total_count}")
    
    # Accuracy threshold check
    if report.overall_accuracy >= 90:
        print("   ✅ Meets production threshold (90%+)")
    elif report.overall_accuracy >= 80:
        print("   ⚠️  Acceptable but could be improved (80-90%)")
    else:
        print("   ❌ Below acceptable threshold (<80%)")
    
    # Per-category breakdown
    print("\n📁 Per-Category Accuracy:")
    print("-" * 40)
    for category in sorted(report.category_accuracy.keys()):
        accuracy = report.category_accuracy[category]
        counts = report.category_counts[category]
        bar = "█" * int(accuracy / 5) + "░" * (20 - int(accuracy / 5))
        print(f"   {category:12} [{bar}] {accuracy:5.1f}% ({counts['correct']}/{counts['total']})")
    
    # Misclassifications
    if report.misclassifications:
        print(f"\n❌ Misclassifications ({len(report.misclassifications)}):")
        print("-" * 40)
        for r in report.misclassifications:
            print(f"   • \"{r.query[:50]}...\"")
            print(f"     Expected: {r.expected} → Got: {r.predicted}")
    else:
        print("\n✅ All queries classified correctly!")
    
    # Recommendations
    print("\n💡 Recommendations:")
    print("-" * 40)
    
    # Find weakest category
    weakest = min(report.category_accuracy.items(), key=lambda x: x[1])
    if weakest[1] < 100:
        print(f"   • Improve {weakest[0]} classification (currently {weakest[1]:.0f}%)")
        print("     Consider adding more examples to the prompt")
    
    if report.overall_accuracy < 90:
        print("   • Add more few-shot examples for commonly confused categories")
        print("   • Review misclassified queries for patterns")
        print("   • Consider rule-based pre-routing for obvious cases")
    
    print("\n" + "=" * 60)


def create_confusion_matrix(report: AccuracyReport) -> dict[str, dict[str, int]]:
    """
    Create a confusion matrix from test results.
    
    Returns:
        Dictionary where matrix[expected][predicted] = count
    """
    matrix: dict[str, dict[str, int]] = {
        cat: {c: 0 for c in VALID_CATEGORIES}
        for cat in VALID_CATEGORIES
    }
    
    for result in report.results:
        matrix[result.expected][result.predicted] += 1
    
    return matrix


def print_confusion_matrix(matrix: dict[str, dict[str, int]]) -> None:
    """Print a formatted confusion matrix."""
    categories = sorted(matrix.keys())
    
    print("\n📊 Confusion Matrix:")
    print("-" * 50)
    
    # Header
    header = "Expected\\Predicted"
    print(f"{header:20}", end="")
    for cat in categories:
        print(f"{cat[:8]:>10}", end="")
    print()
    
    # Rows
    for expected in categories:
        print(f"{expected:20}", end="")
        for predicted in categories:
            count = matrix[expected][predicted]
            if expected == predicted:
                print(f"{'[' + str(count) + ']':>10}", end="")  # Highlight diagonal
            else:
                print(f"{count:>10}", end="")
        print()


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Classification Accuracy Testing")
    print("=" * 60)
    
    # Run the test
    report = run_accuracy_test(verbose=True)
    
    # Print detailed report
    print_report(report)
    
    # Print confusion matrix
    matrix = create_confusion_matrix(report)
    print_confusion_matrix(matrix)
    
    # Summary
    print("\n" + "=" * 60)
    print("Testing Complete")
    print("=" * 60)
    print(f"\nFinal Accuracy: {report.overall_accuracy:.1f}%")
    
    if report.overall_accuracy >= 90:
        print("✅ Ready for production!")
    else:
        print("⚠️  Consider improving classification before production use.")
