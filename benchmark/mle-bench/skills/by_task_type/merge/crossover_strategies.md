# Genetic Crossover Strategies

When merging two parent solutions, the goal is to **combine their strengths** while preserving functional coherence.

## Crossover Methods

### 1. Uniform Crossover (Recommended)
- **Strategy**: Randomly select each gene block from parent A or B according to the gene plan
- **Advantage**: Explores diverse combinations, preserves successful components from both parents
- **Implementation**:
  ```
  For each gene block G:
    if gene_plan[G] == "A": use parent_a.G
    else: use parent_b.G
  ```

### 2. Single-Point Crossover
- **Strategy**: Choose a split point, take blocks before from A, blocks after from B
- **Advantage**: Maintains larger contiguous segments from each parent
- **Use case**: When gene blocks have strong dependencies

### 3. Best-of-Both Selection
- **Strategy**: Analyze each gene block's contribution to fitness, select the better one
- **Advantage**: Maximizes theoretical fitness by choosing best components
- **Risk**: May break synergies between blocks

## Key Principles

### Preserve Working Components
- ✅ **DO** retain high-performing gene blocks unchanged when possible
- ✅ **DO** respect the gene plan provided (it's designed to balance exploration)
- ❌ **DON'T** arbitrarily modify blocks from high-fitness parents

### Maintain Functional Coherence
- Ensure the merged solution is **executable and logically consistent**
- Example: If parent A uses CNN (image input) and parent B uses tabular preprocessing:
  - Adapt preprocessing to generate image-compatible tensors
  - Or adapt model to accept tabular features

### Handle Dependencies
- **DATA ↔ MODEL**: Model architecture must match data format
- **MODEL ↔ LOSS**: Loss function must align with model output (e.g., binary vs. multi-class)
- **OPTIMIZER ↔ MODEL**: Learning rate and optimizer type should match model complexity

## Conflict Resolution Examples

### Example 1: Incompatible DATA and MODEL
**Parent A**: Tabular data → XGBoost
**Parent B**: Image data → CNN

**Gene Plan**: `{"DATA": "B", "MODEL": "A"}`

**Solution**: Adapt MODEL to accept image input:
```python
# Original XGBoost doesn't support images directly
# → Use CNN feature extractor + XGBoost on extracted features
cnn_features = extract_features_with_cnn(image_data)
model = XGBoost().fit(cnn_features, labels)
```

### Example 2: Mismatched LOSS and MODEL
**Parent A**: Binary classification (sigmoid output)
**Parent B**: Multi-class classification (softmax output)

**Gene Plan**: `{"MODEL": "A", "LOSS": "B"}`

**Solution**: Adapt LOSS to binary case:
```python
# Parent B's loss is categorical_crossentropy
# → Replace with binary_crossentropy for Parent A's model
loss = tf.keras.losses.BinaryCrossentropy()
```

## Validation Checklist

After merging, ensure:
- [ ] All gene blocks are present and non-empty
- [ ] DATA block produces data in the format expected by MODEL
- [ ] MODEL output shape matches LOSS function requirements
- [ ] OPTIMIZER parameters are compatible with MODEL
- [ ] Code is syntactically correct and executable
- [ ] No undefined variables or missing imports
