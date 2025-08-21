# Frequently Asked Questions - Batch Normalizations

## General Understanding

**Q: What is the intuition behind Batch Normalization?**
A: Batch Normalization (BatchNorm) normalizes the activations of each mini-batch to have zero mean and unit variance, followed by a learnable scaling and shifting. Intuitively, it reduces internal covariate shift, allowing each layer to learn on a more stable distribution of inputs. It also smooths the loss landscape, leading to faster and more stable training.

**Q: When should I use Batch Normalization instead of alternatives like LayerNorm or GroupNorm?**
A: Use BatchNorm when training deep networks with large mini-batches (e.g., CNNs for image classification). Use LayerNorm for small-batch or sequential data (e.g., NLP with Transformers), and GroupNorm for cases where batch size is small or inconsistent (e.g., object detection). BatchNorm is less effective when the batch size is very small or variable.

**Q: What are the main hyperparameters and how do I tune them?** 
A:
- Momentum (usually ~0.1): Controls how quickly running statistics (mean/var) are updated; higher values adapt faster to new data.
- Epsilon (e.g., 1e-5): A small constant for numerical stability; usually not tuned.
- Learning Rate: Often can be increased when using BatchNorm.
- Affine Parameters (γ and β): Learnable scale and shift; not tuned directly but affect learning.

## Implementation Details

**Q: Why do we need the learnable parameters γ (gamma) and β (beta)?**
A: Without γ and β, the normalization step would constrain all outputs to a fixed distribution (zero mean, unit variance), which could limit the model's capacity. γ and β allow the network to undo or adjust the normalization as needed for expressivity.

**Q: What happens if I forget to set the model to .train() or .eval()?**
A: BatchNorm behaves differently in training and evaluation modes. During training, it uses batch statistics; during evaluation, it uses running estimates. Forgetting to set the mode will result in inconsistent behavior and degraded performance, especially during inference.

**Q: How can I make this more efficient?**
A:
- Fuse BatchNorm with preceding linear/conv layers during inference (many frameworks do this automatically).
- Use synchronized BatchNorm in multi-GPU training to get accurate statistics across devices.
- Reduce batch size sensitivity with GroupNorm or LayerNorm if needed.

## Common Errors

**Q: I'm getting error Expected more than 1 value per channel when training, got input size..., what does it mean?**
A: This error occurs when BatchNorm receives a batch size of 1 or a dimension with only one value, which prevents computation of meaningful batch statistics. Either increase the batch size or switch to a normalization method that doesn't depend on the batch (e.g., LayerNorm or InstanceNorm).

**Q: My results don't match the expected output.**
A: Check these common issues:
- Mode mismatch: Ensure .train() and .eval() modes are correctly used.
- Incorrect placement: BatchNorm should typically be used before the activation function (e.g., ReLU).
- Small batch sizes: If batches are too small, statistics will be noisy — consider GroupNorm or LayerNorm instead.

## Advanced Topics

**Q: How does this relate to LayerNorm or WeightNorm?**
A: While BatchNorm normalizes across the batch dimension, LayerNorm normalizes across feature dimensions and is independent of batch size—making it more suitable for RNNs and transformers. WeightNorm reparameterizes weights rather than activations and doesn't rely on data distribution.

**Q: Can this be extended to other use cases like GANs or reinforcement learning?**
A: Yes, but with caveats. In GANs, BatchNorm can introduce instability due to different statistics in generator/discriminator. Alternatives like InstanceNorm or SpectralNorm are often preferred. In RL, batch statistics may be inconsistent across episodes, so other normalization strategies may work better.

## Resources

**Q: Where can I learn more?**

A: Here are some recommended resources:
- 📘 Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- 📗 Neural Networks and Deep Learning by Michael Nielsen
- 📙 Pattern Recognition and Machine Learning by Christopher M. Bishop
