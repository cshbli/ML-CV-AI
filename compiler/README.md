# Deep Learning Compiler
## TVM与XLA的对比

XLA的整体设计理念，跟TVM存在一些比较明显的区别（以开源部分为例）：

1). XLA目前打击的是非计算密集算子，TVM打击的是计算密集算子，优化目标不一样，导致其作法存在明显的差异；

2). XLA强调自动化，所以给定计算图，XLA会直接codegen出可执行码(XLA社区最近也有一些妥协，开始考虑让用户在计算图上标记XLA的compilation scope，从而应对缓解XLA可用性的问题，虽然我觉得这种作法有一些倒退，但如果配合上系统层面的设计，倒不失为一个务实的解法)，而TVM则把一部分优化工作转移到用户身上（TVM开发者），这个设计理念的区别，我认为其影响还是比较深远的，因为涉及到了整个优化任务复杂性的拆解，XLA想在系统层面完成更多工作，TVM则认为可以把部分工作offload给用户，我认为这跟两个工具假定的目标用户存在差异有关。XLA假定的用户更倾向于是普通算法建模用户，而TVM目前的用户更多是DL引擎开发者（随着时间推移，我的观察和判断TVM会更多推广让普通算法建模用户的使用，但目前的用户还更多是具备引擎系统经验的开发者），这两类用户对于使用接口的容忍度存在比较明显的差异；

3). XLA整体的工程系统设计更为考究，也更为厚重，但不容易拆解出来以模块化的方式为外部使用，而TVM的设计相对更为轻巧，也比较容易以松耦合的方式被外部使用（比如TVM离线gen的kernel被集成到其他DL引擎框架里）；

4). XLA在图优化方面，会有更为复杂专注的实现逻辑，而TVM在图优化方面的实现则相对简单得多；

5). 为了支持更为复杂的图优化，加上自动codegen可执行码的理念，XLA的codegen部分实现逻辑是比较复杂的，相比较而言TVM的codegen部分其实比较朴素直接，如果用技术语言来描述一下的话，TVM的codegen部分，更像是一个纯粹的1-1 mapping性质的visitor实现，而XLA的codegen则除了对IR DAG遍历以外，涉及到针对不同计算pattern的inter-op的codegen逻辑拼接，以及数据存取index的推导计算和复用优化等等。当然TVM的codegen也可能针对不同硬件，加入一些inter-op的graph pattern的处理逻辑，但并不影响主体的界定；

6).TVM是一个经典的machine learning-based system，在完成schedule/computation抽象以外，整个优化空间探索，转换成了一个data-driven的机器学习优化问题，这是一个轻巧，但也一力降十会的作法。XLA在这方面，因为是纯system guy的工作，所以比较实在，是以纯系统的方式来解决优化问题。但是除了机器学习的方式以外，改成heuristics的方式来进行优化空间探索是不是也可能获得相近的效果呢？我觉得这还是一个open的question。不过把历史数据使用起来，辅助指导优化过程的探索寻优，这个原则我是buy in的。

## References
* [深度学习系统杂谈](https://jackwish.net/2019/on-deep-learning-system.html)
* [Learning to Optimize Tensor Programs](https://arxiv.org/pdf/1805.08166.pdf)
* [Boost Quantization Inference Performance](https://jackwish.net/2019/boost-quant-perf.html)
* [Introducing TFLite Parser Python Package](https://jackwish.net/2020/introducing-tflite-parser-package.html)
* [深度学习加速：算法、编译器、体系结构与硬件设计](https://zhuanlan.zhihu.com/p/101544149)
* [也谈TVM和深度学习编译器](https://zhuanlan.zhihu.com/p/87664838)
* [手把手带你遨游TVM](https://zhuanlan.zhihu.com/p/50529704)
* [TVM: Deep Learning模型的优化编译器(强烈推荐, 附踩坑记录)](https://zhuanlan.zhihu.com/p/58918363)
* [深度学习推理引擎的一些思考](https://zhuanlan.zhihu.com/p/87392811)
* [AutoTVM：让AI来编译优化AI系统底层算子](https://zhuanlan.zhihu.com/p/37181530)
* [TVM教程1 — 用TVM编译深度学习模型](https://zhuanlan.zhihu.com/p/111842386)
* [TensorFlow官方发布剪枝优化工具：参数减少80%，精度几乎不变](https://zhuanlan.zhihu.com/p/65846143)
* [神经网络优化算法：Dropout、梯度消失/爆炸、Adam优化算法，一篇就够了！](https://zhuanlan.zhihu.com/p/78854514)
* [模型压缩一半，精度几乎无损，TensorFlow推出半精度浮点量化工具包，还有在线Demo](https://zhuanlan.zhihu.com/p/76872595)
