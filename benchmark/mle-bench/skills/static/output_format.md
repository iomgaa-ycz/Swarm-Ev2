# Response Format

**Your response should contain:**

1. **Thinking** (2-3 sentences, REQUIRED)
   - Briefly state your analysis of the problem/data
   - Identify the key insight driving your approach

2. **A brief outline** (3-5 sentences) of your proposed solution
   - Explain the key idea and approach
   - Highlight the main components or strategies
   - Avoid excessive exploratory data analysis (EDA)

3. **A single markdown code block** (````python...```) implementing this solution
   - The code should be self-contained and executable as-is
   - All necessary imports should be included
   - The code should be well-structured and readable

**CRITICAL Requirements:**

- ❌ **DO NOT** include additional headings or explanations outside the code block
- ❌ **DO NOT** add multiple code blocks or fragments
- ✅ **DO** ensure the code is a single-file Python program
- ✅ **DO** make the code executable without modifications

**Example Structure:**

```
**Thinking**: The dataset has [X rows] with [key features]. The target is [type].
Key insight: [what pattern/approach should work well and why].

I propose a solution using [technique] to address [problem]. The approach consists of:
1) [preprocessing step], 2) [model selection], 3) [validation strategy]. This should
achieve good performance because [reasoning].

​```python
import pandas as pd
# ... (implementation)
​```
```
