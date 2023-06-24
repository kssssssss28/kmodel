from data.loadData import get_data_split
from createModel.createModel import createModel, createSingleModel
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import time

def train(bz, delta, p, m, n, e, lamuda):
    model = createSingleModel(lamuda)
    DNA, labels = get_data_split(0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=p, patience=m)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=n, verbose=1)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()
    train_loss_metric = tf.keras.metrics.Mean()
    train_accuracy_metric = tf.keras.metrics.MeanSquaredError()
    
    batch_size = 256
    num_batches = len(DNA) // batch_size
    
    train_data = DNA
    train_labels = labels
    
    num_epochs = 200  # 指定训练的epoch数量

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        
        for batch in range(num_batches):
            start = batch * batch_size
            end = start + batch_size

            # 获取当前批次的训练数据和标签
            batch_data = train_data[start:end]
            batch_labels = train_labels[start:end]
            
            with tf.GradientTape() as tape:
                # 前向传播
                logits = model(batch_data)
                # 计算损失
                loss_value = loss_fn(batch_labels, logits)

            # 计算梯度
            grads = tape.gradient(loss_value, model.trainable_variables)
            # 更新模型参数
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # 更新训练指标
            train_loss_metric(loss_value)
            train_accuracy_metric(batch_labels, logits)

            # 打印当前批次的训练指标
            if batch % 100 == 0:
                print('Batch {}/{} - Loss: {:.4f} - Accuracy: {:.4f}'.format(
                    batch, num_batches, train_loss_metric.result(), train_accuracy_metric.result()
                ))

            predictions = model(batch_data)
            print('Predictions:', predictions)


        # 打印每个epoch结束时的训练指标
        print('Epoch {} - Loss: {:.4f} - Accuracy: {:.4f}'.format(
            epoch + 1, train_loss_metric.result(), train_accuracy_metric.result()
        ))

    # 最后一批次的训练指标
    print('Final Batch - Loss: {:.4f} - Accuracy: {:.4f}'.format(
        train_loss_metric.result(), train_accuracy_metric.result()
    ))
    
train(32, 0.0001, 0.01, 0, 30, 20, 0.01)
