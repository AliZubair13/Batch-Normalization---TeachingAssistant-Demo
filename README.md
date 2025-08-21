# Batch-Normalization---TeachingAssistant-Demo

# Submission Notes

**Name:** Zubair Ali L 
**Email:** zl5749@nyu.edu 

## Implementation Choice

**Topic Selected:** BatchNorm

**Why I chose this topic:**
Batch Normalization is a fundamental technique that improves the training stability and speed of deep neural networks. I chose this topic because it is widely used in practice, yet implementing it from scratch helps deepen understanding of the underlying mathematical operations and their effect on model training.

## Time Breakdown

- **Setup & Research:** 4 hours
- **Core Implementation:** 6 hours
- **Testing & Debugging:** 3 hours
- **Teaching Materials:** 2 hours
- **Documentation & Polish:** 2 hours
- **Total Time:** 17 hours

## Implementation Details

### Key Design Decisions

1. **Manual computation of batch statistics in the forward pass**
   - What: Implemented calculation of batch mean and variance within the forward function to normalize inputs.
   - Why: To closely mimic the official BatchNorm behavior and understand how statistics impact normalization.
   - Trade-offs: Balancing numerical stability with simplicity; added a small epsilon for variance to avoid division by zero.

2. **Caching intermediate values for backward pass**
   - What: Stored variables such as normalized inputs, variance, mean, and inverse standard deviation during forward pass.
   - Why: Necessary for gradient calculation during the backward pass without recomputation
   - Trade-offs: Additional memory use versus performance and clarity in gradient derivation.

### Assumptions Made

1. The input to BatchNorm1D is always a 2D tensor with shape (batch_size, num_features). This matches the typical use case in fully connected layers
2. The momentum parameter for running mean/variance uses exponential moving average for stable inference statistics.
3. The batch size is always greater than zero to prevent invalid operations during mean and variance calculations.

### Challenges Faced

1. **Challenge:** Ensuring numerical stability during variance normalization.
   **Solution:** Added a small epsilon value (1e-5) inside the square root calculation.
   **Learning:** This small detail is crucial to prevent NaNs and unstable training behavior.

2. **Challenge:** Implementing the backward pass with correct gradient flow through normalization and scaling parameters.
   **Solution:** Carefully derived the gradient formulas step-by-step, stored necessary cached values, and verified shapes and values via tests.
   **Learning:** Understanding the chain rule in normalization contexts and caching intermediate results is vital for custom autograd implementations.

## Testing Approach

**Test Coverage:** 84%

**Testing Strategy:**
#### Unit tests for: 
- BatchNorm1D.forward(), backward(), train(), and eval() methods.
- SimpleMLP.forward() under various conditions (training/eval mode, single sample, large batch).
- Integration test for main() execution.
#### Edge cases tested 
- Input with mismatched feature dimensions.
- Non-2D tensor input.
- Empty batch input (batch_size=0).
- Batch size of 1.
- Input with zero variance (e.g., constant input)
- Calling backward() before any forward() call
#### Performance benchmarks: [Brief results]
- All 18 unit tests passed (pytest tests/ -v)
- High test coverage: 99% for src/implementation.py, 84% total coverage
- All tests completed in ~2.4 seconds, showing efficient execution
- Functional correctness verified under --cov with no failures or slowdowns
![alt text](<teaching/Diagrams/Training Performance - 1.png>)
![Training Performance - 2](teaching/Diagrams/Training%20Performance%20-2.png)

## Teaching Materials

**Tutorial Structure: Batch Normalization**
1. Overview - this part covers learning objectives, prerequisites and setup
2. Part 1: Understanding the concept - intro to batch norm, need of it, and mathematical foundation of batch norm
3. Part 2: Implementation Deep Dive - Segmenting the implementation.py into a more understandable chunks
4. Part 3: Experiments and Visualization - Experimenting with the basic functionality, Training vs inference mode, and Impact on training dynamics
5. Part 4: Common Pitfalls and Debugging - Discussing some of common pitfalls, with how to avoid them and fix it.
6. Part 5: Practical Applications: Applying our implementation to the real world application, and utilizing our BatchNorm1D on a simulated tabular classification task to check with the test loss and accuracy.
7. Part 6: Performance Analysis: It discusses How does our implementation compare to existing libraries?
8. Part 7: Exercises for Students: Its an hands on exercises for students to strengthen their understanding on the concept.
9. Summary and Key Takeaways: It discusses about key insights and objectives of the concept
10. FAQs - Frequently Asked Questions

**Key Visualizations:**
- Pre Normalization vs Post Normalization: This chart shows how Batch Normalization transforms data distributions.
- Basic Functionality Test: It compares Mean, var for before batch Norm and after Batch norm
- Training loss comparison for with batch norm and for without batch norm

**Anticipated Student Questions:**
1. Q: Why do we maintain running mean and variance during training in BatchNorm1D?
   A: Running statistics are required for evaluation mode where we no longer compute batch-wise statistics. This ensures deterministic outputs during inference.

2. Q: What happens when the batch size is 1 in BatchNorm? Is it still useful?
   A: When batch_size = 1, variance becomes zero, and normalization may become unstable. Our implementation handles it gracefully using eps to avoid division by zero, but performance gains are limited in such cases.

3. Q: Why use gamma and beta in BatchNorm when we already normalize the input?
   A: Normalization may erase the original feature distribution. gamma and beta reintroduce the model’s capacity to learn optimal scale and shift parameters after normalization.

## AI Tool Usage Declaration

**I used AI tools for:**
- Improving docstring clarity
- Generating edge case test ideas
- Verifying PyTorch equivalence with my custom implementation
- Ideas to structure the teaching material

**I did NOT use AI tools for:**
- Core algorithm implementation
- Architecture design decisions
- Teaching content creation
- Test case design

## Reflection

**What went well:**
- Forward and backward pass implementations for BatchNorm1D passed all unit and integration tests.
- PyTorch vs. custom model comparison showed comparable results.
- Full coverage was achieved for critical branches including edge cases

**What was challenging:**
- Implementing stable backward propagation logic, especially derivative calculations with respect to mean and variance.
- Managing shape consistency across batch and feature dimensions.
- Designing comprehensive test cases to hit 99%+ code coverage.

**What I would do differently:**
- Start with smaller helper functions for backward logic to improve readability.
- Add numerical gradient checks early to catch gradient bugs.
- Use better logging or visualization to understand running stats behavior during training.

**Key learnings:**
- Gained a solid understanding of how normalization techniques impact model training and performance.
- Learned the significance of managing training vs. evaluation behavior in model components.
- Developed skills in writing effective tests that cover both standard usage and edge cases.

## Future Improvements

If I had more time, I would:
- In-depth understanding of how normalization affects convergence and gradient flow.
- Importance of correctly switching between training and evaluation modes.
- How to structure robust unit tests for both functionality and corner cases.

## Submission Checklist

✅ Core algorithm implemented and working
✅ All tests passing with good coverage
✅ Teaching notebook is clear and runnable
✅ Code is well-documented with docstrings
✅ Visualizations are included and helpful
✅ Git history shows iterative development
✅ SUBMISSION.md is complete
✅ All files are pushed to GitHub

---

**Declaration:** I confirm that this submission is my own work, completed in accordance with the academic integrity guidelines.

**Signature:** ZUBAIR ALI L 
**Date:** 07/27/2025
