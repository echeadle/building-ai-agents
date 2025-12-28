# Evaluator-Optimizer Pattern Diagrams

These diagrams can be rendered using Mermaid (supported in GitHub, VS Code, and many other tools).

## Basic Flow Diagram

```mermaid
flowchart TD
    A[Task Input] --> B[Generator]
    B --> C[Generated Content]
    C --> D[Evaluator]
    D --> E{Approved?}
    E -->|Yes| F[Final Output]
    E -->|No| G{Max Iterations?}
    G -->|No| H[Feedback]
    H --> B
    G -->|Yes| I[Best Effort Output]
```

## Detailed Architecture

```mermaid
flowchart TB
    subgraph Input
        T[Task Description]
        CR[Criteria + Rubrics]
        CFG[Config: max_iter, threshold]
    end
    
    subgraph Generator
        G[Generator LLM]
        GSP[System Prompt: Expert Role]
    end
    
    subgraph Evaluator
        E[Evaluator LLM]
        ESP[System Prompt: Reviewer Role]
        RUB[Rubrics]
    end
    
    subgraph Control
        LOOP[Iteration Loop]
        BEST[Best Tracker]
        HIST[History]
    end
    
    subgraph Stopping
        S1[Quality Met?]
        S2[Max Iterations?]
        S3[Diminishing Returns?]
        S4[Feedback Converged?]
    end
    
    subgraph Output
        OUT[Final Content]
        META[Metadata]
        LOG[Iteration Log]
    end
    
    T --> G
    CR --> E
    CFG --> LOOP
    
    G --> |content| E
    E --> |scores + feedback| LOOP
    LOOP --> |feedback| G
    
    LOOP --> S1 & S2 & S3 & S4
    S1 & S2 & S3 & S4 --> OUT
    
    LOOP --> BEST
    LOOP --> HIST
    BEST --> OUT
    HIST --> LOG
```

## Iteration Flow

```mermaid
sequenceDiagram
    participant U as User
    participant O as Optimizer
    participant G as Generator
    participant E as Evaluator
    
    U->>O: optimize(task)
    
    loop Until approved or max iterations
        O->>G: generate(task, prev_output, feedback)
        G-->>O: content
        
        O->>E: evaluate(content)
        E-->>O: scores + feedback + decision
        
        alt Approved
            O-->>U: OptimizationResult(approved)
        else Not Approved & More Iterations
            Note over O: Store feedback for next iteration
        else Max Iterations Reached
            O-->>U: OptimizationResult(best_effort)
        end
    end
```

## Stopping Conditions Decision Tree

```mermaid
flowchart TD
    CHECK[Check Stopping Conditions]
    
    CHECK --> Q1{All scores >= threshold?}
    Q1 -->|Yes| APPROVED[Return: Approved]
    Q1 -->|No| Q2{iteration >= max?}
    
    Q2 -->|Yes| BEST[Return: Best Effort]
    Q2 -->|No| Q3{improvement < min?}
    
    Q3 -->|Yes| BEST
    Q3 -->|No| Q4{feedback == prev_feedback?}
    
    Q4 -->|Yes| BEST
    Q4 -->|No| CONTINUE[Continue to Next Iteration]
```

## Multi-Evaluator Variation

```mermaid
flowchart LR
    G[Generator] --> C[Content]
    
    C --> E1[Technical Evaluator]
    C --> E2[Style Evaluator]
    C --> E3[Compliance Evaluator]
    
    E1 --> AGG[Aggregated Feedback]
    E2 --> AGG
    E3 --> AGG
    
    AGG --> |if not approved| G
    AGG --> |if approved| OUT[Final Output]
```

## Usage

1. **GitHub**: Mermaid diagrams render automatically in markdown files
2. **VS Code**: Install the "Markdown Preview Mermaid Support" extension
3. **Online**: Use https://mermaid.live/ to render and export
4. **Obsidian**: Native mermaid support in notes

These diagrams complement the ASCII diagrams in the chapter and can be used for presentations or documentation.
