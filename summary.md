##  项目目标  
最开始只有ai生成的需求：让我自己根据所学来拟合$y = x^2$。就是用mlp来实现  
## 实现过程  
先用dataset存数据，然后划分训练集和测试集，构建mlp模型：两层linear和一层relu
然后进行模型训练：清空梯度，进行预测，算出loss，用loss反向传播，然后用optimizer更新，一直这样循环。最后用plot画出$y = x^2$ 和预测曲线进行对比
##  问题与 Bug  
 **tensor可能不能用来绘图，需要转换成numpy**

**linear不能接收整数，需要float()转化为浮点数**

**直接拿出来的x_test是无序的，画图很丑陋，需要排序**

**样本的取值是-2到2，如果用超出范围的数据预测误差会很大**


##  项目总结 / 收获  
### 学会了不少api使用：
```python
torch.linspace(-2, 2, 200).reshape(-1, 1)
```
在-2到2均匀地取200个样本点

```python
train_dataset, test_dataset = torch.utils.data.random_split(  
    dataset,  
    [train_size, test_size]  
)
```
用pytorch的方法划分训练集和测试集。这里生成的两个set都是**Subset对象** 它们都是原dataset的子集，`Subset` 保存了原始 Dataset 和对应的索引，因此可以通过下标访问其中的样本。

```python
x_test = torch.stack([item[0] for item in test_dataset])
```
从Subset里面取数据。stack:把多个 shape 相同的 Tensor，在一个新的维度上堆叠起来，形成一个新的 Tensor。本来是40个shape为(1)的，变成一个shape为(40,1)的tensor

```python
sort_idx = torch.argsort(x_test, dim=0).squeeze()
```
其中，**argsort**：返回排序后的**下标**
	   **dim=0** :按照x的顺序来排序，xtest是(40,1)，第 0 维表示样本维度
	   **.squeeze()**：由于得到的还是(40,1)，压缩成(40)，即去除大小为1的维度

```python
loss_fn2 = torch.nn.MSELoss(reduction='none')
```
得到每一个样本点的误差，而不是总体的平均误差

```python
x_test_np = x_test_sorted.detach().numpy()
```
转换带有梯度追踪的tensor为numpy，方便画图。对于**要求梯度的 Tensor**，需要用detach()，因为 NumPy 不参与 PyTorch 的 autograd。


### 体会了一个完整的训练流程
处理数据->选择模型->模型训练->再次整理数据并画图
## 后续改进
发现一个问题：为什么加一个RELU就可以拟合抛物线？

最后的拟合函数可以写成这样的形式：

$$
f(x)=\sum_{i=1}^{8}a_i\operatorname{ReLU}(w_i x+b_i)+c
$$

RELU在临界点可以改变拟合线的斜率，多个就可以分多个段了
分段函数的和是分段区间更细的分段函数