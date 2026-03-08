# Response Format

**Your response should contain:**

1. **Thinking** (3-5 sentences, REQUIRED)
   - If an Evolution Log is available, reference specific insights from it
   - If no Evolution Log is available: for merge, analyze parent solutions' strengths/weaknesses; for mutate, analyze the parent's bottleneck; for draft, analyze the task requirements
   - Identify the bottleneck you're addressing
   - Explain why your proposed change should help
   - **DO NOT** use generic descriptions like "The dataset has X rows..."

2. **A brief outline** (3-5 sentences) of your proposed solution
   - Explain the key idea and approach
   - Highlight the main components or strategies
   - Avoid excessive exploratory data analysis (EDA)

3. **A single markdown code block** (````python...```) implementing this solution
   - The code should be self-contained and executable as-is
   - All necessary imports should be included
   - The code should be well-structured and readable

**CRITICAL Requirements:**

- **DO NOT** include additional headings or explanations outside the code block
- **DO NOT** add multiple code blocks or fragments
- **DO** ensure the code is a single-file Python program
- **DO** make the code executable without modifications

**Example Thinking (with Evolution Log):**

> "Based on Evolution Log Step 8's insight that airport features improved RMSE by 12%,
> I will explore additional location-based features. The current bottleneck is
> long-distance prediction accuracy. I propose adding zone-based target encoding
> to capture neighborhood-level fare patterns."

**Example Thinking (without Evolution Log, mutate):**

> "The parent solution shows signs of overfitting (train=0.95, val=0.82).
> The MODEL block uses no regularization. I will add Dropout(0.3) and L2
> regularization to improve generalization."
