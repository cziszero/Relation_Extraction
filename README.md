# Relation_Extraction
Relation Classification via Convolutional Deep Neural Network

The code is an implementation of the paper http://www.aclweb.org/anthology/C14-1220 using tensorflow.

##Algorithm
- I almost followed the technique used in the paper mentioned above, only tweaking with some parameters such as dimensions of word vector, position vectors, optimization function and so on.
- Basic architecture is a convolution layer, max pool and final softamx layer. We can always add/delete the number of conv and max-pool layers b/w the input layer and the final softmax layer. I used only 1 conv and 1 max pool.

##Files
 - **text_cnn.py** - It is a class which implements the architecture of the model. So it accepts the input, contains all the layers such as **conv2d**(convolution layer), **max_pool** etc. which process the input vector and finally gives the output in terms of predictions for each class.
 - **data_helpers.py** - It is a generic script which contains helpers such as generating batches, loading the training data etc 载入训练数据，生成批次等.
 - **train.py** - This module creates the input vector from the training data, and finally trains the model on the data and saves it on the disk.输入训练数据，训练模型然后存在磁盘上
 - **temp.py** - This is a pyspark code used to fetch data from the **HBase** table and predict the class of each row using the trained model. 这是pyspark代码，用来从HBase中获取数据并用训练好的模型来预测每一行的类别。
 
 ##Challenges
 - My training data is around 7K rows. Due to this, the accuracy is around **70.34%** on the test set. So as the training set grows, I'm sure the model will perform much better.训练数据为7K行，测试集上准确率大概为70.34%，训练集增长时，模型一定会表现的更好。
 - My data set consists of inter sentencial entities with entities linked with a cause-effect relationship. However, this model can extended to a n class problem.
 
 ##TODO
 - Use RNNs maybe LSTMS for the training.
 - Fine tuning the model.
 
 ###PYSPARK can now be used with TensorFlow for online training and testing. In my case I am using pysark for online testing.