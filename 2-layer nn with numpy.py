#!/usr/bin/env python
# coding: utf-8

# In[332]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import time


# In[333]:


class nn(object):
    def __init__(self,inputnode,hiddennode,outputnode,lr):
        #nodes
        self.input = inputnode
        self.hidden = hiddennode
        self.output = outputnode
        
        #learning rate
        self.lr = lr
        
        #weight matrix
        self.w_i_h = np.random.normal(0,pow(self.hidden,-0.5),(self.input,self.hidden))
        self.w_h_o = np.random.normal(0,pow(self.output,-0.5),(self.hidden,self.output))
        
        #active function
        
        
    
    def active_function(self,x):
        return 1/(1+np.exp(-x))
    
    def train(self,input_list,targets):
        inputs = np.array(input_list,ndmin=2).T
        targets = np.array(targets,ndmin=2).T   #targets denotes the label
        
        hidden_inputs = np.matmul(self.w_i_h.T,inputs)
        hidden_outputs = self.active_function(hidden_inputs) #(100,1)
        
        output_inputs = np.matmul(self.w_h_o.T,hidden_outputs)
        output_outputs = self.active_function(output_inputs) #(10,1)
        
        output_error = targets - output_outputs  #(10,1)
        hidden_error = np.matmul(self.w_h_o,output_error)  #row vector(1,100)
        
        
        #update w_h_o weights
        self.w_h_o += self.lr * np.matmul(output_error*output_outputs*(1-output_outputs),hidden_outputs.T).T
        #[(10,1).(1,100)].T=(10,100).T=(100,10)
        
        #update w_i_h weights
        self.w_i_h +=self.lr *np.matmul(hidden_error*hidden_outputs*(1-hidden_outputs),inputs.T).T
        
        self.loss = [0]
        self.loss +=  np.linalg.norm(output_error)
        
        pass
        
    def test(self,input_list):
        inputs = np.array(input_list,ndmin=2).T
        hidden_inputs = np.dot(self.w_i_h.T,inputs)
        hidden_outputs = self.active_function(hidden_inputs)
        output_inputs = np.dot(self.w_h_o.T,hidden_outputs)
        output_outputs = self.active_function(output_inputs)
        
        return output_outputs


# In[334]:


'''input_list=np.array([1,2,3])
targets=np.array([3,4])
nn=nn(3,4,2,0.3)
nn.train(input_list,targets)
nn.loss
'''


# In[335]:


data_file = open("mnist_train.csv",'r')
data_list = data_file.readlines()
data_file.close()
#len(data_list)
data_list[1]


# In[336]:


all_values = data_list[1].split(',')
image_array = np.asfarray(all_values[1:]).reshape(28,28)
plt.imshow(image_array)


# In[337]:


input_nodes = 784
hidden_nodes = 100
output_nodes = 10
lr = 0.01
nn = nn(input_nodes,hidden_nodes,output_nodes,lr)
training_set = open("mnist_train.csv",'r')
training_list = training_set.readlines()
training_set.close()


# In[338]:


#training
loss_record=[]
time_record=0
epoch = 10
for iter in range(epoch):
    loss = 0
    start = time.time()
    for record in training_list:
        all_values = record.split(',')
        #scaled
        inputs = np.asfarray(all_values[1:])/255*0.99+0.01
    
        targets = np.zeros(output_nodes)+0.01
        targets[int(all_values[0])] = 0.99  #all_values[0] is the target label
    
        nn.train(inputs,targets)
        loss += nn.loss
        
    loss /= len(training_list)
    print("the loss of epoch "+str(iter+1)+" is: "+str(loss))
    end = time.time()
    print("spending time of epoch "+str(iter+1)+" is: "+str(end-start))
    time_record += end-start
    loss_record.append(loss)
    
print("The training is ending.")
print("The total time spending is: "+str(time_record))


# In[339]:


x = np.arange(epoch)+1
y = loss_record
plt.plot(x,y,'r:')
plt.show()


# In[340]:


nn.w_i_h


# In[341]:


nn.w_h_o


# In[342]:


#testing
test_set = open("mnist_test.csv",'r')
test_list = test_set.readlines()
test_set.close()

correct_num = 0

for record in test_list:
    all_values = record.split(',')
    correct_label = int(all_values[0])
    input_list = np.asfarray(all_values[1:])/255*0.99+0.01
    outputs = nn.test(input_list)
    pre_label = np.argmax(outputs)
    
    if(pre_label==correct_label):
        correct_num += 1
    
correct_rate = correct_num/len(test_list)
print("correct rate is: "+str(correct_rate))


# In[343]:


lr_correct[0.01]=0.9656


# In[300]:


lr_correct[0.3]=0.9428


# In[380]:


lr_correct


# In[409]:


lr_list = sorted(lr_correct.keys())
lr_list.pop(0)
acc=np.zeros(5)

for i in range(len(lr_list)):
    acc_rate = lr_correct[lr_list[i]]
    acc[i] = acc_rate  #此处有个很有趣的现象：用循环赋值给非零列表，结果列表元素全部归0；另外，append()函数参数不能放表达式
    
plt.plot(lr_list,acc,'ro')
plt.show()
print(acc)


# In[ ]:




